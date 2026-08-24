# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlashInfer GIN prepare/finalize adapter for multi-node expert parallelism."""

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)


def _validate_rank_invariant_scalar(scale: torch.Tensor, name: str) -> torch.Tensor:
    """Reduce a per-expert activation scale tensor to one validated fp32 scalar.

    The GIN dispatch quantizes with a single device-resident scalar, so a scheme
    whose activation scale varies per expert cannot be folded into the wire
    format.
    """
    scale = scale.detach().reshape(-1)
    if scale.device.type != "cuda":
        raise ValueError(f"flashinfer_gin {name} must be on CUDA.")
    if scale.numel() == 0 or not torch.is_floating_point(scale):
        raise ValueError(f"flashinfer_gin {name} must be a floating-point scalar.")
    scale = scale.to(dtype=torch.float32)
    if not bool(torch.isfinite(scale).all().item()):
        raise ValueError(f"flashinfer_gin {name} must be finite.")
    if not bool((scale > 0).all().item()):
        raise ValueError(f"flashinfer_gin {name} must be positive.")
    if not bool((scale == scale[0]).all().item()):
        raise ValueError(
            f"flashinfer_gin requires one {name} shared by all experts."
        )
    return scale[:1].clone().contiguous()


class FlashInferGinPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Run a fused BF16-to-{NVFP4,FP8} GIN dispatch and GIN combine.

    The dispatch activation format is inferred from the MoE quantization scheme
    and fixed for the lifetime of the underlying GIN handle. The combine wire is
    unaffected by that choice.
    """

    def __init__(
        self,
        a2a: Any,
        quant_config: FusedMoEQuantConfig,
        num_experts: int,
        gscale: torch.Tensor | None = None,
        dispatch_quantization: str | None = None,
        zero_copy_combine: bool = False,
    ) -> None:
        super().__init__()
        if gscale is None or dispatch_quantization is None:
            dispatch_quantization, gscale = self.validate_quant_config(quant_config)

        self.a2a = a2a
        self.num_experts = num_experts
        self._quant_config = quant_config
        self._dispatch_quantization = dispatch_quantization
        self._zero_copy_combine = zero_copy_combine
        # NVFP4 reads this as the a1_gscale MULTIPLIER, FP8 as the a1_scale
        # DIVISOR. Both are rank-invariant scalars the dispatch kernel re-reads
        # on every graph replay.
        self._gscale_dev = gscale
        self._cscale_dev = torch.ones_like(self._gscale_dev)
        self._last_runtime_max_tokens: int | None = None
        self._registered_combine_output: torch.Tensor | None = None

    @staticmethod
    def validate_quant_config(
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[str, torch.Tensor]:
        """Pick the dispatch format and its scalar scale from the MoE scheme.

        Returns (dispatch_quantization, scale) before collective handle creation
        so a misconfiguration fails on every rank rather than mid-dispatch.
        """
        if quant_config.use_nvfp4_w4a4:
            if getattr(quant_config, "per_act_token_quant", False):
                raise ValueError(
                    "flashinfer_gin does not support per-token NVFP4 activation "
                    "scales."
                )
            if quant_config.a1_gscale is None:
                raise ValueError("flashinfer_gin requires an NVFP4 activation gscale.")
            if quant_config.is_scale_swizzled:
                raise ValueError(
                    "flashinfer_gin requires linear activation scale factors."
                )
            return "nvfp4", _validate_rank_invariant_scalar(
                quant_config.a1_gscale, "activation gscale"
            )

        if quant_config.use_fp8_w8a8:
            # Only the static per-tensor scheme folds into a scalar wire divisor.
            # Per-token and blockwise FP8 would need a per-token scale payload.
            if getattr(quant_config, "per_act_token_quant", False):
                raise ValueError(
                    "flashinfer_gin does not support per-token FP8 activation scales."
                )
            if quant_config.is_block_quantized:
                raise ValueError(
                    "flashinfer_gin does not support blockwise FP8 activation scales."
                )
            if not quant_config.is_per_tensor:
                raise ValueError(
                    "flashinfer_gin requires per-tensor FP8 activation scales."
                )
            if quant_config.a1_scale is None:
                raise ValueError("flashinfer_gin requires an FP8 activation scale.")
            return "fp8_per_tensor", _validate_rank_invariant_scalar(
                quant_config.a1_scale, "activation scale"
            )

        raise ValueError(
            "flashinfer_gin requires static NVFP4 or static per-tensor FP8 MoE."
        )

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts) -> None:
        from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
            TrtLlmFp8ExpertsModular,
        )
        from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
            TrtLlmNvFp4ExpertsModular,
        )

        if self._dispatch_quantization == "fp8_per_tensor":
            if not isinstance(fused_experts, TrtLlmFp8ExpertsModular):
                raise ValueError(
                    "flashinfer_gin FP8 dispatch requires "
                    "TrtLlmFp8ExpertsModular; set moe_backend=flashinfer_trtllm."
                )
            # The per-tensor kernel folds a1_scale into its output scalars and
            # takes pre-quantized hidden states, which is what GIN delivers.
            if not self._quant_config.is_per_tensor:
                raise ValueError(
                    "flashinfer_gin requires per-tensor FP8 activation scales."
                )
            return

        from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (  # noqa: E501
            FlashInferCuteDSLExperts,
        )

        if not isinstance(
            fused_experts,
            (TrtLlmNvFp4ExpertsModular, FlashInferCuteDSLExperts),
        ):
            raise ValueError(
                "flashinfer_gin currently requires "
                "TrtLlmNvFp4ExpertsModular or FlashInferCuteDSLExperts."
            )
        if fused_experts.expects_unquantized_inputs:
            raise ValueError(
                "flashinfer_gin does not support per-token NVFP4 activation scales."
            )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def num_dispatchers(self) -> int:
        return self.a2a.ep_size

    def output_is_reduced(self) -> bool:
        return True

    def fused_expert_output_buffer(self) -> torch.Tensor | None:
        if not self._zero_copy_combine or self.a2a.combine_quant:
            return None
        runtime_max_tokens = self._last_runtime_max_tokens
        if runtime_max_tokens is None:
            raise RuntimeError(
                "flashinfer_gin requested an expert output buffer before prepare"
            )
        registered = self.a2a.get_registered_combine_input(runtime_max_tokens)
        expected_shape = (
            self.a2a.ep_size,
            runtime_max_tokens,
            self.a2a.hidden,
        )
        if (
            not isinstance(registered, torch.Tensor)
            or registered.shape != expected_shape
            or registered.dtype != torch.bfloat16
            or registered.device.type != "cuda"
            or not registered.is_contiguous()
        ):
            raise RuntimeError(
                "flashinfer_gin returned an invalid registered combine input"
            )
        self._registered_combine_output = registered.view(
            self.a2a.ep_size * runtime_max_tokens,
            self.a2a.hidden,
        )
        return self._registered_combine_output

    def _runtime_max_tokens(self, local_num_tokens: int) -> int:
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is None:
            raise RuntimeError("flashinfer_gin requires DP forward metadata.")
        local_sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        if local_sizes is None:
            raise RuntimeError("flashinfer_gin requires per-DP-rank token counts.")
        runtime_max_tokens = max(local_sizes)
        if local_num_tokens > runtime_max_tokens:
            raise RuntimeError(
                "local token count exceeds the synchronized DP token bound"
            )
        return runtime_max_tokens

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if defer_input_quant:
            raise ValueError("flashinfer_gin always quantizes during dispatch.")
        if apply_router_weight_on_input:
            raise ValueError(
                "flashinfer_gin does not support apply_router_weight_on_input."
            )
        if num_experts != self.num_experts:
            raise ValueError(
                f"expected {self.num_experts} experts, got {num_experts}"
            )
        if quant_config is not self._quant_config:
            raise ValueError("flashinfer_gin quantization config changed after setup.")
        if (
            a1.device.type != "cuda"
            or a1.dtype != torch.bfloat16
            or not a1.is_contiguous()
        ):
            raise ValueError("flashinfer_gin requires contiguous BF16 activations.")
        if a1.ndim != 2 or a1.shape[1] != self.a2a.hidden:
            raise ValueError(
                f"expected activations shaped [tokens, {self.a2a.hidden}]"
            )
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("flashinfer_gin requires contiguous int32 expert IDs.")
        if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
            raise ValueError(
                "flashinfer_gin requires contiguous float32 router weights."
            )
        if topk_ids.shape != topk_weights.shape:
            raise ValueError("expert IDs and router weights must have the same shape.")
        if topk_ids.shape[0] != a1.shape[0]:
            raise ValueError(
                "flashinfer_gin routing rows must match the activation rows."
            )
        if topk_ids.device != a1.device or topk_weights.device != a1.device:
            raise ValueError("flashinfer_gin inputs must share one CUDA device.")
        if topk_ids.ndim != 2 or topk_ids.shape[1] != self.a2a.top_k:
            raise ValueError(f"flashinfer_gin expects top_k={self.a2a.top_k}.")
        if expert_map is not None and (
            expert_map.dtype != torch.int32
            or expert_map.numel() < self.num_experts
        ):
            raise ValueError("flashinfer_gin received an invalid linear expert map.")

        runtime_max_tokens = self._runtime_max_tokens(a1.shape[0])
        self._last_runtime_max_tokens = runtime_max_tokens
        self._registered_combine_output = None
        (activation, scale_factors, expert_ids, router_weights), _ = (
            self.a2a.dispatch_async(
                topk_ids,
                a1,
                self._gscale_dev,
                topk_weights,
                runtime_max_tokens,
                invalid_token_expert_id=-1,
            )
        )
        rows = self.a2a.ep_size * runtime_max_tokens
        return (
            activation.reshape(rows, -1),
            # None under per-tensor FP8: no per-token scale ships, and the expert
            # backend already folded a1_scale into its output scalars.
            None if scale_factors is None else scale_factors.reshape(rows, -1),
            None,
            expert_ids.reshape(rows, -1),
            router_weights.reshape(rows, -1),
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if apply_router_weight_on_input:
            raise ValueError(
                "flashinfer_gin does not support apply_router_weight_on_input."
            )
        if not isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP):
            raise ValueError(
                "flashinfer_gin requires experts that weight and reduce locally."
            )
        runtime_max_tokens = self._last_runtime_max_tokens
        if runtime_max_tokens is None:
            raise RuntimeError("flashinfer_gin finalize called before prepare.")
        if (
            output.device.type != "cuda"
            or output.dtype != torch.bfloat16
            or not output.is_contiguous()
        ):
            raise ValueError("flashinfer_gin requires a contiguous BF16 output.")
        if (
            fused_expert_output.device != output.device
            or fused_expert_output.dtype != torch.bfloat16
            or not fused_expert_output.is_contiguous()
        ):
            raise ValueError(
                "flashinfer_gin requires contiguous BF16 expert output."
            )

        payload = fused_expert_output.reshape(
            self.a2a.ep_size,
            runtime_max_tokens,
            self.a2a.hidden,
        )
        if self._zero_copy_combine and not self.a2a.combine_quant:
            registered = self._registered_combine_output
            if (
                registered is None
                or fused_expert_output.shape != registered.shape
                or fused_expert_output.data_ptr() != registered.data_ptr()
            ):
                raise RuntimeError(
                    "flashinfer_gin raw zero-copy combine requires the exact "
                    "registered fused-expert output buffer"
                )
        if self.a2a.combine_quant:
            self._cscale_dev.copy_(
                (448.0 * 6.0)
                / payload.detach().abs().amax().clamp_min(1e-6).to(torch.float32)
            )
        combine = (
            self.a2a.combine_zero_copy_async
            if self._zero_copy_combine
            else self.a2a.combine_async
        )
        combine(
            payload,
            output,
            output.shape[0],
            runtime_max_tokens,
            self._cscale_dev,
        )
