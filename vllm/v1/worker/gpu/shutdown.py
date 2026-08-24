# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def free_before_shutdown(vllm_config: VllmConfig) -> None:
    if vllm_config.parallel_config.all2all_backend == "flashinfer_gin":
        from vllm.compilation.breakable_cudagraph import (
            BreakableCUDAGraphWrapper,
        )
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        # GIN context pointers may be embedded in these graphs. Ensure their
        # destructors run before the EP communicator begins collective teardown.
        CUDAGraphWrapper.clear_all_graphs()
        BreakableCUDAGraphWrapper.clear_all_graphs()
        gc.collect()
        torch.accelerator.synchronize()

    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT
    from vllm.v1.worker.workspace import reset_workspace_manager

    cache_config = vllm_config.cache_config
    cache_config.num_gpu_blocks = None

    compilation_config = vllm_config.compilation_config
    compilation_config.static_forward_context.clear()

    _ROPE_DICT.clear()
    reset_workspace_manager()
