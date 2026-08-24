# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep_cutedsl import (
    FlashInferMoeEpCutedsl,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    is_global_sf_supported_for_nvfp4_backend,
    make_nvfp4_moe_kernel,
    make_nvfp4_moe_quant_config,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform

_FLASHINFER_MOE_EP_CUTEDSL = "flashinfer_moe_ep_cutedsl"

# NVFP4 backends that have been verified to support EPLB for this quantization recipe.
_EPLB_SUPPORTED_NVFP4_BACKENDS = frozenset(
    {
        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED,
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
    }
)

logger = init_logger(__name__)


def _validate_flashinfer_moe_ep_cutedsl_config(
    moe: FusedMoEConfig,
    *,
    use_a16: bool,
) -> None:
    vllm_config = get_current_vllm_config()
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    capability = current_platform.get_device_capability()

    unsupported: list[str] = []
    if model is None or model.architectures != ["MistralLarge3ForCausalLM"]:
        unsupported.append("model architectures other than MistralLarge3ForCausalLM")
    if use_a16:
        unsupported.append("A16 activations")
    if model is None or model.enable_sleep_mode:
        unsupported.append("sleep mode")
    if vllm_config.lora_config is not None or moe.is_lora_enabled:
        unsupported.append("LoRA")
    if vllm_config.weight_transfer_config is not None:
        unsupported.append("runtime weight transfer")
    if parallel.tensor_parallel_size != 1:
        unsupported.append("tensor parallelism")
    if parallel.pipeline_parallel_size != 1:
        unsupported.append("pipeline parallelism")
    if parallel.prefill_context_parallel_size != 1:
        unsupported.append("prefill context parallelism")
    if parallel.data_parallel_size != 4:
        unsupported.append("data parallel size other than 4")
    if not parallel.enable_expert_parallel:
        unsupported.append("expert parallelism disabled")
    if parallel.enable_eplb or parallel.eplb_config.num_redundant_experts != 0:
        unsupported.append("EPLB or redundant experts")
    if parallel.enable_dbo:
        unsupported.append("dual batch overlap")
    if parallel.enable_elastic_ep:
        unsupported.append("elastic expert parallelism")
    if parallel.use_sequence_parallel_moe:
        unsupported.append("sequence parallel MoE")
    if parallel.expert_placement_strategy != "linear":
        unsupported.append("non-linear expert placement")
    if not vllm_config.kernel_config.enable_cutedsl_warmup:
        unsupported.append("disabled CuTeDSL warmup")
    if capability is None or capability.to_int() != 100:
        unsupported.append("GPU architectures other than SM100")

    mp = moe.moe_parallel_config
    if not mp.use_ep or mp.ep_size != 4 or mp.dp_size != 4:
        unsupported.append("MoE parallel configuration other than DP4/EP4")
    if mp.tp_size != 1 or mp.pcp_size != 1 or mp.sp_size != 1:
        unsupported.append("sharded MoE compute")
    if moe.skip_final_all_reduce:
        unsupported.append("skip_final_all_reduce")

    expected = {
        "hidden_dim": 7168,
        "intermediate_size": 4096,
        "num_experts": 128,
        "num_local_experts": 32,
        "num_logical_experts": 128,
        "experts_per_token": 4,
    }
    actual = {name: getattr(moe, name) for name in expected}
    if actual != expected:
        unsupported.append(f"non-ML3 MoE geometry {actual}")
    if moe.in_dtype != torch.bfloat16:
        unsupported.append(f"activation dtype {moe.in_dtype}")
    if moe.activation is not MoEActivation.SILU:
        unsupported.append(f"activation {moe.activation.value}")
    if moe.has_bias:
        unsupported.append("expert bias")
    if moe.swiglu_limit is not None:
        unsupported.append("SwiGLU clamping")

    if unsupported:
        raise ValueError(
            f"{_FLASHINFER_MOE_EP_CUTEDSL} only supports ML3 TP1/DP4/EP4 "
            f"prefill deployments; unsupported: {', '.join(unsupported)}"
        )


def _require_finite_positive(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all() or not (value > 0).all():
        raise ValueError(f"{name} must contain finite positive scales")


def _require_exact_shard_match(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"{name} must have two logical w1/w3 scale shards")
    _require_finite_positive(name, value)
    if not torch.equal(value[:, 0], value[:, 1]):
        raise ValueError(f"{name} must match exactly between w1 and w3")
    return value[:, 0].contiguous()


def _global_exact_scale(local_scales: torch.Tensor) -> float:
    local_min, local_max = torch.aminmax(local_scales)
    bounds = torch.stack((local_max, -local_min))
    torch.distributed.all_reduce(
        bounds,
        op=torch.distributed.ReduceOp.MAX,
        group=get_ep_group().device_group,
    )
    global_max = bounds[0]
    global_min = -bounds[1]
    if not torch.equal(global_min, global_max):
        raise ValueError(
            "w13_input_global_scale must match exactly across all routed experts"
        )
    return float(global_min.item())


class CompressedTensorsW4A4Nvfp4MoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
        use_a16: bool = False,
    ):
        super().__init__(moe)
        self.group_size = 16
        self.use_flashinfer_moe_ep_cutedsl = (
            self.moe.moe_backend == _FLASHINFER_MOE_EP_CUTEDSL
        )
        self.load_input_scales_by_shard = self.use_flashinfer_moe_ep_cutedsl
        self._flashinfer_moe_ep_cutedsl: FlashInferMoeEpCutedsl | None = None

        if self.use_flashinfer_moe_ep_cutedsl:
            _validate_flashinfer_moe_ep_cutedsl_config(moe, use_a16=use_a16)
            self.nvfp4_backend = None
            self.experts_cls = None
            self.use_global_sf = False
            return

        # Select experts implementation.
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=None if use_a16 else kNvfp4Dynamic,
        )

        self.use_global_sf = is_global_sf_supported_for_nvfp4_backend(
            self.nvfp4_backend
        )

    @property
    def supports_eplb(self) -> bool:
        if self.use_flashinfer_moe_ep_cutedsl:
            return False
        return self.nvfp4_backend in _EPLB_SUPPORTED_NVFP4_BACKENDS

    @property
    def supports_internal_mk(self) -> bool:
        return self.use_flashinfer_moe_ep_cutedsl or super().supports_internal_mk

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        if self.use_flashinfer_moe_ep_cutedsl:
            return False
        return super().mk_can_overlap_shared_experts

    @property
    def output_is_reduced(self) -> bool:
        return self.use_flashinfer_moe_ep_cutedsl or super().output_is_reduced

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        if self.use_flashinfer_moe_ep_cutedsl:
            return torch.int32
        return super().topk_indices_dtype

    @property
    def is_monolithic(self) -> bool:
        if self.use_flashinfer_moe_ep_cutedsl:
            return False
        return super().is_monolithic

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight Global Scales
        w13_weight_global_data = (
            torch.full(
                (num_experts, w13_num_shards),
                torch.nan,
                dtype=torch.float32,
            )
            if self.use_flashinfer_moe_ep_cutedsl
            else torch.empty(
                num_experts,
                w13_num_shards,
                dtype=torch.float32,
            )
        )
        w13_weight_scale_2 = torch.nn.Parameter(
            w13_weight_global_data,
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_global_data = (
            torch.full((num_experts,), torch.nan, dtype=torch.float32)
            if self.use_flashinfer_moe_ep_cutedsl
            else torch.empty(num_experts, dtype=torch.float32)
        )
        w2_weight_scale_2 = torch.nn.Parameter(
            w2_weight_global_data,
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input Global Scales
        w13_input_data = (
            torch.full(
                (num_experts, w13_num_shards),
                torch.nan,
                dtype=torch.float32,
            )
            if self.use_flashinfer_moe_ep_cutedsl
            else torch.empty(
                num_experts,
                w13_num_shards,
                dtype=torch.float32,
            )
        )
        w13_input_scale = torch.nn.Parameter(
            w13_input_data,
            requires_grad=False,
        )
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_data = (
            torch.full((num_experts,), torch.nan, dtype=torch.float32)
            if self.use_flashinfer_moe_ep_cutedsl
            else torch.empty(num_experts, dtype=torch.float32)
        )
        w2_input_scale = torch.nn.Parameter(
            w2_input_data,
            requires_grad=False,
        )
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """
        Convert NVFP4 MoE weights into kernel format and setup the kernel.
        """
        if self.use_flashinfer_moe_ep_cutedsl:
            self._process_flashinfer_moe_ep_cutedsl_weights(layer)
            return

        # NOTE(rob): wN_weight_packed -> wN_weight is because ModularKernelMethod
        # requires this naming convention. However, the name change breaks
        # reloading because the state dict no longer matches disk. Once we
        # remove MKM, we should revert this change to ensure compatibility.
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        # Use a single gscale for w13.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected.",
            )
        w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()

        # Shuffle weights into the NvFp4 kernel format.
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=(1.0 / w13_weight_global_scale),
            a13_scale=(1.0 / layer.w13_input_global_scale),
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=(1.0 / layer.w2_weight_global_scale),
            a2_scale=(1.0 / layer.w2_input_global_scale),
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        replace_parameter(layer, "w13_weight_scale_2", w13_scale_2)
        replace_parameter(layer, "w2_weight_scale_2", w2_scale_2)
        layer.w13_input_scale = a13_scale
        layer.w2_input_scale = a2_scale

        # Setup modular kernel.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.nvfp4_backend,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def _process_flashinfer_moe_ep_cutedsl_weights(
        self,
        layer: RoutedExperts,
    ) -> None:
        w13_weight_global = _require_exact_shard_match(
            "w13_weight_global_scale",
            layer.w13_weight_global_scale,
        )
        w13_input_global = _require_exact_shard_match(
            "w13_input_global_scale",
            layer.w13_input_global_scale,
        )
        _require_finite_positive(
            "w2_weight_global_scale",
            layer.w2_weight_global_scale,
        )
        _require_finite_positive(
            "w2_input_global_scale",
            layer.w2_input_global_scale,
        )

        expert_map = layer.expert_map
        if expert_map is None:
            raise ValueError(
                f"{_FLASHINFER_MOE_EP_CUTEDSL} requires an EP expert map"
            )
        local_global_experts = torch.where(expert_map >= 0)[0]
        first_expert = self.moe.ep_rank * self.moe.num_local_experts
        expected_experts = torch.arange(
            first_expert,
            first_expert + self.moe.num_local_experts,
            device=local_global_experts.device,
        )
        expected_local_slots = torch.arange(
            self.moe.num_local_experts,
            dtype=expert_map.dtype,
            device=expert_map.device,
        )
        if not torch.equal(local_global_experts, expected_experts) or not torch.equal(
            expert_map[expected_experts], expected_local_slots
        ):
            raise ValueError(
                f"{_FLASHINFER_MOE_EP_CUTEDSL} requires contiguous linear "
                "expert placement"
            )

        # CT stores the FI norm constants directly: q * stored_sf reconstructs
        # x * norm_const, and alpha cancels the activation and weight constants.
        input_norm_const = _global_exact_scale(w13_input_global)
        fc1_alpha = torch.reciprocal(
            w13_weight_global * w13_input_global
        ).contiguous()
        fc1_norm_const = layer.w2_input_global_scale.contiguous()
        fc2_alpha = torch.reciprocal(
            layer.w2_weight_global_scale * layer.w2_input_global_scale
        ).contiguous()

        adapter = FlashInferMoeEpCutedsl(
            layer,
            self.moe,
            input_norm_const=input_norm_const,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
        )
        self._flashinfer_moe_ep_cutedsl = adapter
        self.moe_quant_config = FusedMoEQuantConfig.make(
            "nvfp4",
            weight_dtype="nvfp4",
        )

        for name in (
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_global_scale",
            "w2_weight_global_scale",
            "w13_input_global_scale",
            "w2_input_global_scale",
        ):
            delattr(layer, name)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        if self.use_flashinfer_moe_ep_cutedsl:
            assert self.moe_quant_config is not None
            return self.moe_quant_config
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            swiglu_alpha=getattr(layer, "swiglu_alpha", None),
            swiglu_beta=getattr(layer, "swiglu_beta", None),
            layer=layer,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_flashinfer_moe_ep_cutedsl:
            raise RuntimeError(
                f"{_FLASHINFER_MOE_EP_CUTEDSL} is not a monolithic MoE backend"
            )
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_flashinfer_moe_ep_cutedsl:
            adapter = self._flashinfer_moe_ep_cutedsl
            if adapter is None:
                raise RuntimeError(
                    f"{_FLASHINFER_MOE_EP_CUTEDSL} weights were not finalized"
                )
            return adapter(x, topk_ids, topk_weights)
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
