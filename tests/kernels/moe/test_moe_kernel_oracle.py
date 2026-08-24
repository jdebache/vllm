# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoE kernel oracle construction and delegation.

The unquantized test contains a canonical demonstration that
`UnquantizedMoEKernelOracle` methods delegate one-to-one to the
existing module-level functions in `oracle/unquantized.py`. Each method
on `UnquantizedMoEKernelOracle` follows the same `return module_fn(args)`
pattern, so verifying delegation for one method (`make_kernel`) gives
high confidence in the rest.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    TrtLlmFp8ExpertsModular,
)
from vllm.model_executor.layers.fused_moe.oracle import UnquantizedMoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    make_fp8_moe_kernel,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)


class TestUnquantizedDelegation:
    """UnquantizedMoEKernelOracle methods must delegate to the existing
    module-level functions; behaviour is bit-identical."""

    def test_make_kernel_delegates(self) -> None:
        quant_config = object()
        moe_config = object()
        experts_cls = TritonExperts
        sentinel_kernel = object()

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized."
            "make_unquantized_moe_kernel",
            return_value=sentinel_kernel,
        ) as mocked:
            out = UnquantizedMoEKernelOracle().make_kernel(
                quant_config,
                moe_config,
                UnquantizedMoeBackend.TRITON,
                experts_cls,
            )

        mocked.assert_called_once_with(
            quant_config,
            moe_config,
            UnquantizedMoeBackend.TRITON,
            experts_cls,
            None,  # routing_tables default
        )
        assert out is sentinel_kernel


def test_fp8_trtllm_kernel_processes_layer_weights() -> None:
    layer = torch.nn.Module()
    experts = Mock()
    prepare_finalize = SimpleNamespace(
        activation_format=mk.FusedMoEActivationFormat.Standard
    )
    sentinel_kernel = object()

    with (
        patch.object(TrtLlmFp8ExpertsModular, "__new__", return_value=experts),
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.fp8."
            "maybe_make_prepare_finalize",
            return_value=prepare_finalize,
        ),
        patch.object(mk, "FusedMoEKernel", return_value=sentinel_kernel),
    ):
        out = make_fp8_moe_kernel(
            moe_quant_config=Mock(),
            moe_config=Mock(),
            experts_cls=TrtLlmFp8ExpertsModular,
            fp8_backend=Fp8MoeBackend.FLASHINFER_TRTLLM,
            layer=layer,
        )

    experts.process_weights_after_loading.assert_called_once_with(layer)
    assert out is sentinel_kernel
