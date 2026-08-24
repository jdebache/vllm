# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    BatchedPrepareAndFinalize,
    make_moe_prepare_and_finalize_naive_dp_ep,
    make_moe_prepare_and_finalize_no_dp_ep,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_gin import (
    FlashInferGinPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_one_sided import (  # noqa: E501
    FlashInferNVLinkOneSidedPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_two_sided import (  # noqa: E501
    FlashInferNVLinkTwoSidedPrepareAndFinalize,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_cutedsl_moe_nvfp4,
    has_flashinfer_trtllm_fused_moe,
)
from vllm.utils.import_utils import (
    has_deep_ep,
    has_deep_ep_v2,
    has_mori,
    has_nixl_ep,
)

logger = init_logger(__name__)


def _validate_flashinfer_gin_moe_backend(
    moe_backend: str, dispatch_quantization: str
) -> None:
    if moe_backend == "flashinfer_trtllm":
        if not has_flashinfer_trtllm_fused_moe():
            raise ValueError("flashinfer_gin requires FlashInfer TRT-LLM fused MoE.")
        return

    if dispatch_quantization == "nvfp4" and moe_backend == "flashinfer_cutedsl":
        if not has_flashinfer_cutedsl_moe_nvfp4():
            raise ValueError("flashinfer_gin requires FlashInfer CuteDSL NVFP4 MoE.")
        return

    required_backend = (
        "flashinfer_trtllm or flashinfer_cutedsl"
        if dispatch_quantization == "nvfp4"
        else "flashinfer_trtllm"
    )
    raise ValueError(
        f"flashinfer_gin {dispatch_quantization} dispatch requires "
        f"moe_backend={required_backend}."
    )


def _validate_flashinfer_gin_expert_layout(moe: FusedMoEConfig, ep_size: int) -> int:
    num_physical_experts = moe.num_experts
    if num_physical_experts % ep_size != 0:
        raise ValueError(
            "flashinfer_gin requires experts to divide evenly across EP ranks."
        )
    expected_local_experts = num_physical_experts // ep_size
    if moe.num_local_experts != expected_local_experts:
        raise ValueError(
            "flashinfer_gin expected "
            f"{expected_local_experts} physical experts per EP rank, got "
            f"{moe.num_local_experts}; fused shared expert slots are not "
            "supported."
        )
    return num_physical_experts


if current_platform.is_cuda_alike():
    if has_deep_ep():
        from .prepare_finalize.deepep_ht import DeepEPHTPrepareAndFinalize
        from .prepare_finalize.deepep_ll import (
            DEEPEP_QUANT_BLOCK_SHAPE,
            DeepEPLLPrepareAndFinalize,
        )
    if has_deep_ep_v2():
        from .prepare_finalize.deepep_v2 import DeepEPV2PrepareAndFinalize
    if has_mori():
        from .prepare_finalize.mori import MoriPrepareAndFinalize
    if has_nixl_ep():
        from .prepare_finalize.nixl_ep import (
            NIXL_EP_QUANT_BLOCK_SHAPE,
            NixlEPPrepareAndFinalize,
        )


def get_ep_all2all_manager(eep_stage: bool = False) -> Any:
    if eep_stage:
        from vllm.distributed.elastic_ep.standby_state import get_standby_ep_group

        ep_group = get_standby_ep_group()
        assert ep_group is not None
        device_communicator = ep_group.device_communicator
    else:
        device_communicator = get_ep_group().device_communicator

    assert device_communicator is not None
    all2all_manager = device_communicator.all2all_manager
    assert all2all_manager is not None
    return all2all_manager


def maybe_roundup_layer_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    moe_parallel_config: FusedMoEParallelConfig,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        moe_parallel_config: Fused MoE parallelization strategy configuration.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs
        and all2all backend.
        Original hidden size otherwise.
    """
    if moe_parallel_config.use_deepep_ht_kernels:
        hidden_size = DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size, act_dtype
        )

    if moe_parallel_config.use_deepep_ll_kernels:
        hidden_size = DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size
        )

    if moe_parallel_config.use_deepep_v2_kernels:
        hidden_size = DeepEPV2PrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size, act_dtype
        )

    if moe_parallel_config.use_nixl_ep_kernels:
        hidden_size = NixlEPPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size
        )

    return hidden_size


def maybe_make_prepare_finalize(
    moe: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig | None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    allow_new_interface: bool = False,
    use_monolithic: bool = False,
    eep_stage: bool = False,
) -> FusedMoEPrepareAndFinalize | None:
    if not moe.moe_parallel_config.use_all2all_kernels:
        if not allow_new_interface:
            return None

        # Opt-in XPU batched path: reorganize tokens into E x T x K locally
        # (no all-to-all) so BatchedTritonExperts (moe_mmk TD) can run.
        if current_platform.is_xpu() and moe.moe_backend == "batched_triton":
            return BatchedPrepareAndFinalize(
                max_num_tokens=moe.max_num_tokens,
                num_local_experts=moe.num_local_experts,
                num_dispatchers=1,
                rank=moe.moe_parallel_config.ep_rank,
            )

        # For DP/TP case, fall back to naive P/F.
        if moe.moe_parallel_config.dp_size > 1:
            logger.info_once(
                "Detected DP deployment with no --enable-expert-parallel. "
                "Falling back to AllGather+ReduceScatter dispatch/combine."
            )
            all2all_manager = get_ep_all2all_manager(eep_stage)
            return make_moe_prepare_and_finalize_naive_dp_ep(
                is_sequence_parallel=moe.moe_parallel_config.is_sequence_parallel,
                num_dispatchers=all2all_manager.world_size,
                use_monolithic=use_monolithic,
            )
        else:
            return make_moe_prepare_and_finalize_no_dp_ep(use_monolithic)

    all2all_manager = get_ep_all2all_manager(eep_stage)

    prepare_finalize: FusedMoEPrepareAndFinalize | None = None

    if moe.use_deepep_ht_kernels:
        assert moe.dp_size == all2all_manager.dp_world_size

        all_to_all_args: dict[str, Any] = dict()
        handle = all2all_manager.get_handle(all_to_all_args)
        prepare_finalize = DeepEPHTPrepareAndFinalize(
            handle,
            num_dispatchers=all2all_manager.world_size,
            dp_size=all2all_manager.dp_world_size,
            rank_expert_offset=all2all_manager.rank * moe.num_local_experts,
        )

    elif moe.use_deepep_ll_kernels:
        assert quant_config is not None
        global_to_physical = physical_to_global = local_expert_global_ids = None
        if routing_tables is not None:
            (
                global_to_physical,
                physical_to_global,
                local_expert_global_ids,
            ) = routing_tables
        all_to_all_args = dict(
            max_num_tokens_per_dp_rank=moe.max_num_tokens,
            token_hidden_size=moe.hidden_dim,
            num_ep_ranks=all2all_manager.world_size,
            num_global_experts=moe.num_experts,
            num_local_experts=moe.num_experts // all2all_manager.world_size,
        )
        handle = all2all_manager.get_handle(all_to_all_args)

        # Note: We may want to use FP8 dispatch just to reduce
        # data movement.
        use_fp8_dispatch = (
            quant_config.quant_dtype == current_platform.fp8_dtype()
            and quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE
        )

        prepare_finalize = DeepEPLLPrepareAndFinalize(
            handle,
            max_tokens_per_rank=moe.max_num_tokens,
            num_dispatchers=all2all_manager.world_size,
            use_fp8_dispatch=use_fp8_dispatch,
            global_to_physical=global_to_physical,
            physical_to_global=physical_to_global,
            local_expert_global_ids=local_expert_global_ids,
        )
    elif moe.use_deepep_v2_kernels:
        assert moe.dp_size == all2all_manager.dp_world_size

        use_fp8_dispatch = (
            quant_config is not None
            and quant_config.quant_dtype == current_platform.fp8_dtype()
            and quant_config.is_block_quantized
        )
        all_to_all_args = dict(
            num_max_tokens_per_rank=moe.max_num_tokens,
            hidden=moe.hidden_dim,
            num_topk=moe.experts_per_token,
            num_experts=moe.num_experts,
            use_fp8_dispatch=use_fp8_dispatch,
        )
        handle = all2all_manager.get_handle(all_to_all_args)
        vllm_config = get_current_vllm_config()
        use_cudagraph = not vllm_config.model_config.enforce_eager

        prepare_finalize = DeepEPV2PrepareAndFinalize(
            buffer=handle,
            num_dispatchers=all2all_manager.world_size,
            dp_size=all2all_manager.dp_world_size,
            rank_expert_offset=all2all_manager.rank * moe.num_local_experts,
            num_experts=moe.num_experts,
            num_topk=moe.experts_per_token,
            use_fp8_dispatch=use_fp8_dispatch,
            use_cudagraph=use_cudagraph,
        )

    elif moe.use_mori_kernels:
        assert quant_config is not None

        # Note: We may want to use FP8 dispatch just to reduce
        # data movement.
        use_fp8_dispatch = (
            quant_config.is_per_act_token or quant_config.is_block_quantized
        )
        if use_fp8_dispatch:
            # For PTPC (per token per channel) quant, scale dim is 1
            # For 1x128 quant, scale dim is hidden_dim // 128
            quant_dtype = quant_config.quant_dtype
            scale_dim = 1 if quant_config.is_per_act_token else moe.hidden_dim // 128
        else:
            # Unquantized dispatch (e.g. AITER with defer_input_quant):
            # dispatch raw BF16/FP16 data, no scales needed.
            quant_dtype = moe.in_dtype
            scale_dim = 0
        all_to_all_args = dict(
            rank=all2all_manager.rank,
            num_ep_ranks=all2all_manager.world_size,
            quant_dtype=quant_dtype,
            token_hidden_size=moe.hidden_dim,
            scale_dim=scale_dim,
            scale_type_size=0 if scale_dim == 0 else torch.float32.itemsize,
            max_num_tokens_per_dp_rank=moe.max_num_tokens,
            input_dtype=moe.in_dtype,
            num_local_experts=moe.num_experts // all2all_manager.world_size,
            num_experts_per_token=moe.experts_per_token,
        )
        handle = all2all_manager.get_handle(all_to_all_args)

        prepare_finalize = MoriPrepareAndFinalize(
            handle,
            max_tokens_per_rank=moe.max_num_tokens,
            num_dispatchers=all2all_manager.world_size,
            use_fp8_dispatch=use_fp8_dispatch,
        )

    elif moe.use_fi_nvl_two_sided_kernels:
        assert quant_config is not None
        prepare_finalize = FlashInferNVLinkTwoSidedPrepareAndFinalize(
            num_dispatchers=all2all_manager.world_size,
        )

    elif moe.use_fi_nvl_one_sided_kernels:
        assert quant_config is not None
        max_num_tokens = (
            get_current_vllm_config().scheduler_config.max_num_batched_tokens
        )
        if quant_config.quant_dtype is None:
            dispatch_dtype_bytes_per_elem = 2
            dispatch_scale_bytes_per_token = 0
        elif quant_config.quant_dtype == "nvfp4":
            dispatch_dtype_bytes_per_elem = 0
            dispatch_scale_bytes_per_token = moe.hidden_dim // 16
        elif quant_config.quant_dtype == "mxfp8":
            dispatch_dtype_bytes_per_elem = 1
            align = quant_config.mx_alignment
            if align > 0:
                padded_k = ((moe.hidden_dim + align - 1) // align) * align
            else:
                padded_k = moe.hidden_dim
            dispatch_scale_bytes_per_token = padded_k // 32
        elif quant_config.quant_dtype == current_platform.fp8_dtype():
            # Plain fp8 (e4m3) activations, 1 byte/elem. Only per-tensor-static
            # scaling is supported here: the scale is a rank-invariant constant
            # held on the expert, so nothing extra is dispatched with the
            # tokens. Per-act-token / block-quant fp8 would need a genuine
            # per-token fp32 scale to travel alongside; that path is not wired
            # yet (the one-sided scale bookkeeping assumes 1-byte scale elems,
            # see FlashInferNVLinkOneSidedPrepareAndFinalize.prepare).
            if quant_config.is_per_act_token or quant_config.is_block_quantized:
                raise NotImplementedError(
                    "flashinfer_nvlink_one_sided fp8 dispatch currently supports "
                    "per-tensor-static scaling only; got "
                    f"per_act_token={quant_config.is_per_act_token}, "
                    f"block_shape={quant_config.block_shape}"
                )
            dispatch_dtype_bytes_per_elem = 1
            dispatch_scale_bytes_per_token = 0
        else:
            raise NotImplementedError(
                "flashinfer_nvlink_one_sided dispatch supports nvfp4, mxfp8, "
                "fp8 (e4m3), and bf16 (quant_dtype=None) today; got "
                f"quant_dtype={quant_config.quant_dtype!r}"
            )
        prepare_finalize = FlashInferNVLinkOneSidedPrepareAndFinalize(
            max_num_tokens=max_num_tokens,
            top_k=moe.experts_per_token,
            num_experts=moe.num_experts,
            hidden_size=moe.hidden_dim,
            num_dispatchers=all2all_manager.world_size,
            dispatch_dtype_bytes_per_elem=dispatch_dtype_bytes_per_elem,
            dispatch_scale_bytes_per_token=dispatch_scale_bytes_per_token,
        )

    elif moe.use_fi_gin_kernels:
        if quant_config is None or not (
            quant_config.use_nvfp4_w4a4 or quant_config.use_fp8_w8a8
        ):
            raise ValueError(
                "flashinfer_gin requires static NVFP4 or static per-tensor FP8 MoE."
            )
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        moe_parallel_config = moe.moe_parallel_config
        if (
            moe_parallel_config.tp_size != 1
            or moe_parallel_config.pcp_size != 1
            or moe_parallel_config.sp_size != 1
            or moe_parallel_config.dp_size != moe_parallel_config.ep_size
        ):
            raise ValueError(
                "flashinfer_gin requires TP=PCP=SP=1 and DP=EP; got "
                f"TP={moe_parallel_config.tp_size}, "
                f"PCP={moe_parallel_config.pcp_size}, "
                f"SP={moe_parallel_config.sp_size}, "
                f"DP={moe_parallel_config.dp_size}, "
                f"EP={moe_parallel_config.ep_size}."
            )
        if moe_parallel_config.ep_size != all2all_manager.world_size:
            raise ValueError("flashinfer_gin manager must span the complete EP group.")
        if (
            moe_parallel_config.ep_size < 8
            or moe_parallel_config.ep_size > 32
            or moe_parallel_config.ep_size % 8 != 0
        ):
            raise ValueError(
                "flashinfer_gin requires EP to be a multiple of 8 in [8, 32]."
            )
        if parallel_config.expert_placement_strategy != "linear":
            raise ValueError("flashinfer_gin requires linear expert placement.")
        # EPLB maps logical routes to fixed physical expert slots before
        # dispatch. GIN specializes on that physical count and its linear
        # ownership, which remain stable across rebalances.
        num_physical_experts = _validate_flashinfer_gin_expert_layout(
            moe, moe_parallel_config.ep_size
        )
        if (
            moe.experts_per_token < 2
            or moe.experts_per_token > 8
            or moe.experts_per_token % 2
        ):
            raise ValueError("flashinfer_gin requires even top_k in [2, 8].")
        if moe.hidden_dim % 64 != 0:
            raise ValueError(
                "flashinfer_gin requires a hidden dimension divisible by 64."
            )
        if moe.in_dtype != torch.bfloat16:
            raise ValueError("flashinfer_gin currently requires BF16 activations.")

        validated = None
        validation_error = None
        try:
            candidate = FlashInferGinPrepareAndFinalize.validate_quant_config(
                quant_config
            )
            _validate_flashinfer_gin_moe_backend(
                moe.moe_backend,
                candidate[0],
            )
            validated = candidate
        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
        all2all_manager.collective_error_check(  # type: ignore[attr-defined]
            "MoE configuration validation",
            validation_error,
        )
        if validated is None:
            raise RuntimeError(
                "flashinfer_gin validation did not produce an activation scale."
            )
        dispatch_quantization, validated_gscale = validated

        max_num_tokens = (
            parallel_config.flashinfer_gin_max_num_tokens or moe.max_num_tokens
        )
        handle = all2all_manager.get_handle(
            dict(
                max_num_tokens=max_num_tokens,
                top_k=moe.experts_per_token,
                hidden=moe.hidden_dim,
                num_experts=num_physical_experts,
                combine_quant=int(parallel_config.flashinfer_gin_combine_quant),
                zero_copy_combine=int(parallel_config.flashinfer_gin_zero_copy_combine),
                # Load-bearing: an NVFP4 main model and an FP8 draft model agree
                # on every other key field, so without the format the draft would
                # silently reuse the main model's NVFP4 handle.
                dispatch_quantization=dispatch_quantization,
            )
        )
        prepare_finalize = FlashInferGinPrepareAndFinalize(
            handle,
            quant_config,
            num_experts=num_physical_experts,
            gscale=validated_gscale,
            dispatch_quantization=dispatch_quantization,
            zero_copy_combine=parallel_config.flashinfer_gin_zero_copy_combine,
        )

    elif moe.use_ag_rs_all2all_kernels and allow_new_interface:
        prepare_finalize = make_moe_prepare_and_finalize_naive_dp_ep(
            use_monolithic=use_monolithic,
            is_sequence_parallel=moe.moe_parallel_config.is_sequence_parallel,
            num_dispatchers=all2all_manager.world_size,
        )

    elif moe.use_nixl_ep_kernels:
        assert quant_config is not None
        global_to_physical = physical_to_global = local_expert_global_ids = None
        if routing_tables is not None:
            (
                global_to_physical,
                physical_to_global,
                local_expert_global_ids,
            ) = routing_tables
        all_to_all_args = dict(
            max_num_tokens_per_dp_rank=moe.max_num_tokens,
            token_hidden_size=moe.hidden_dim,
            num_ep_ranks=all2all_manager.world_size,
            num_global_experts=moe.num_experts,
            num_local_experts=moe.num_experts // all2all_manager.world_size,
            stage=eep_stage,
        )
        handle = all2all_manager.get_handle(all_to_all_args)

        # Note: We may want to use FP8 dispatch just to reduce
        # data movement.
        use_fp8_dispatch = (
            quant_config.quant_dtype == current_platform.fp8_dtype()
            and quant_config.block_shape == NIXL_EP_QUANT_BLOCK_SHAPE
        )

        prepare_finalize = NixlEPPrepareAndFinalize(
            handle,
            max_tokens_per_rank=moe.max_num_tokens,
            num_dispatchers=all2all_manager.world_size,
            use_fp8_dispatch=use_fp8_dispatch,
            global_to_physical=global_to_physical,
            physical_to_global=physical_to_global,
            local_expert_global_ids=local_expert_global_ids,
        )

    return prepare_finalize
