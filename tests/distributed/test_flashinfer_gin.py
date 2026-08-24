# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from types import SimpleNamespace

import pytest
import torch

import vllm.config.parallel as parallel_config_module
import vllm.distributed.device_communicators.all2all as all2all_module
import vllm.model_executor.layers.fused_moe.all2all_utils as all2all_utils_module
import vllm.model_executor.layers.fused_moe.modular_kernel as mk_module
import vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_gin as pf_module
from vllm.config import EPLBConfig, ParallelConfig
from vllm.distributed.device_communicators.all2all import (
    FlashInferGinAll2AllManager,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_gin import (
    FlashInferGinPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)


class _FakeGinHandle:
    def __init__(self, key, destroy_error=None):
        self.key = key
        self.destroy_error = destroy_error
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1
        if self.destroy_error is not None:
            raise self.destroy_error


def _make_manager() -> FlashInferGinAll2AllManager:
    manager = FlashInferGinAll2AllManager.__new__(FlashInferGinAll2AllManager)
    manager.rank = 3
    manager.world_size = 16
    manager.cpu_group = object()
    manager.gpus_per_node = 8
    manager._handles = {}
    manager._lock = threading.RLock()
    manager._destroy_started = False
    manager._test_module = object()
    manager._preload_module = lambda top_k, hidden, num_experts: manager._test_module
    manager._new_uid = lambda module: torch.arange(128, dtype=torch.uint8)
    manager.collective_error_check = lambda phase, error: None
    return manager


def _handle_args(**overrides):
    args = dict(
        max_num_tokens=128,
        top_k=4,
        hidden=7168,
        num_experts=128,
        combine_quant=0,
        zero_copy_combine=0,
        dispatch_quantization="nvfp4",
    )
    args.update(overrides)
    return args


def test_manager_strongly_caches_exact_config(monkeypatch):
    allocation_calls = []
    create_calls = []

    class _FakeGinMoeAlltoAll:
        @classmethod
        def allocate_recv_views(cls, *args):
            allocation_calls.append(args)
            return ("recv_views", args)

        @classmethod
        def create(cls, **kwargs):
            create_calls.append(kwargs)
            return _FakeGinHandle(len(create_calls))

    monkeypatch.setattr(
        all2all_module,
        "GinMoeAlltoAll",
        _FakeGinMoeAlltoAll,
        raising=False,
    )
    manager = _make_manager()
    preload_calls = []
    modules = {}
    uid_modules = []

    def _preload(top_k, hidden, num_experts):
        key = (top_k, hidden, num_experts)
        preload_calls.append(key)
        return modules.setdefault(key, object())

    manager._preload_module = _preload
    manager._new_uid = lambda module: uid_modules.append(module) or torch.arange(
        128, dtype=torch.uint8
    )

    first = manager.get_handle(_handle_args())
    again = manager.get_handle(_handle_args())
    second = manager.get_handle(_handle_args(hidden=4096))

    assert again is first
    assert second is not first
    assert len(create_calls) == 2
    assert create_calls[0]["uid"].shape == (128,)
    assert create_calls[0]["module"] is modules[(4, 7168, 128)]
    assert create_calls[1]["module"] is modules[(4, 4096, 128)]
    assert create_calls[0]["rank"] == 3
    assert create_calls[0]["ep_size"] == 16
    assert create_calls[0]["gpus_per_node"] == 8
    assert create_calls[0]["use_lsa"] == 1
    assert "dispatch_transport" not in create_calls[0]
    assert create_calls[0]["recv_views"] == (
        "recv_views",
        (16, 128, 4, 7168, "nvfp4"),
    )
    assert preload_calls == [
        (4, 7168, 128),
        (4, 4096, 128),
    ]
    assert allocation_calls == [
        (16, 128, 4, 7168, "nvfp4"),
        (16, 128, 4, 4096, "nvfp4"),
    ]
    assert uid_modules == [
        modules[(4, 7168, 128)],
        modules[(4, 4096, 128)],
    ]
    assert list(manager._handles) == [
        (128, 4, 7168, 128, 8, 0, 0, "nvfp4"),
        (128, 4, 4096, 128, 8, 0, 0, "nvfp4"),
    ]


def test_manager_supports_rail_relay_only_flashinfer_create(monkeypatch):
    class _RailRelayOnlyGinMoeAlltoAll:
        @classmethod
        def allocate_recv_views(cls, *args):
            return ("recv_views", args)

        @classmethod
        def create(
            cls,
            uid,
            rank,
            ep_size,
            max_num_tokens,
            top_k,
            hidden,
            num_experts,
            gpus_per_node=8,
            use_lsa=1,
            combine_quant=0,
            dispatch_quantization="nvfp4",
            recv_views=None,
            enable_zero_copy_combine=False,
            *,
            module=None,
        ):
            return _FakeGinHandle("rail_relay")

    monkeypatch.setattr(
        all2all_module,
        "GinMoeAlltoAll",
        _RailRelayOnlyGinMoeAlltoAll,
        raising=False,
    )

    handle = _make_manager().get_handle(_handle_args())

    assert handle.key == "rail_relay"


def test_manager_requests_rail_relay_from_legacy_flashinfer(monkeypatch):
    transports = []

    class _LegacyGinMoeAlltoAll:
        @classmethod
        def allocate_recv_views(cls, *args):
            return ("recv_views", args)

        @classmethod
        def create(cls, *, dispatch_transport="direct", **kwargs):
            transports.append(dispatch_transport)
            return _FakeGinHandle(dispatch_transport)

    monkeypatch.setattr(
        all2all_module,
        "GinMoeAlltoAll",
        _LegacyGinMoeAlltoAll,
        raising=False,
    )

    handle = _make_manager().get_handle(_handle_args())

    assert transports == ["rail_relay"]
    assert handle.key == "rail_relay"


def test_manager_caches_zero_copy_contexts_separately(monkeypatch):
    create_calls = []

    class _FakeGinMoeAlltoAll:
        def combine_zero_copy_async(self):
            pass

        def get_registered_combine_input(self):
            pass

        @classmethod
        def allocate_recv_views(cls, *args):
            return ("recv_views", args)

        @classmethod
        def create(cls, *, enable_zero_copy_combine=False, **kwargs):
            create_calls.append(
                dict(
                    kwargs,
                    enable_zero_copy_combine=enable_zero_copy_combine,
                )
            )
            return _FakeGinHandle(len(create_calls))

    monkeypatch.setattr(
        all2all_module,
        "GinMoeAlltoAll",
        _FakeGinMoeAlltoAll,
        raising=False,
    )
    manager = _make_manager()

    staged = manager.get_handle(_handle_args())
    zero_copy = manager.get_handle(_handle_args(zero_copy_combine=1))
    zero_copy_again = manager.get_handle(_handle_args(zero_copy_combine=1))
    quantized = manager.get_handle(_handle_args(combine_quant=1, zero_copy_combine=1))

    assert staged is not zero_copy
    assert zero_copy_again is zero_copy
    assert quantized is not zero_copy
    assert len(create_calls) == 3
    assert [call["enable_zero_copy_combine"] for call in create_calls] == [
        False,
        True,
        False,
    ]


@pytest.mark.parametrize(
    ("combine_quant", "expected_missing"),
    [
        (0, "get_registered_combine_input"),
        (1, "combine_zero_copy_async"),
    ],
)
def test_manager_rejects_missing_zero_copy_api_before_allocation(
    monkeypatch, combine_quant, expected_missing
):
    events = []

    class _OldGinMoeAlltoAll:
        @classmethod
        def allocate_recv_views(cls, *args):
            events.append(("allocate", args))
            raise AssertionError("allocation must not run")

        @classmethod
        def create(cls, *, enable_zero_copy_combine=False, **kwargs):
            raise AssertionError("create must not run")

    monkeypatch.setattr(
        all2all_module,
        "GinMoeAlltoAll",
        _OldGinMoeAlltoAll,
        raising=False,
    )
    manager = _make_manager()

    with pytest.raises(RuntimeError, match=expected_missing):
        manager.get_handle(
            _handle_args(
                combine_quant=combine_quant,
                zero_copy_combine=1,
            )
        )

    assert events == []


def test_manager_coordinates_receive_view_allocation_before_native_setup(
    monkeypatch,
):
    events = []

    class _FakeGinMoeAlltoAll:
        @classmethod
        def allocate_recv_views(cls, *args):
            events.append(("allocate", args))
            raise torch.OutOfMemoryError("receive views")

        @classmethod
        def create(cls, **kwargs):
            events.append(("create", kwargs))
            raise AssertionError("native setup must not run")

    monkeypatch.setattr(
        all2all_module,
        "GinMoeAlltoAll",
        _FakeGinMoeAlltoAll,
        raising=False,
    )
    manager = _make_manager()

    def _collective_error_check(phase, error):
        events.append(("check", phase, error))
        raise RuntimeError(error)

    manager.collective_error_check = _collective_error_check

    with pytest.raises(RuntimeError, match="OutOfMemoryError: receive views"):
        manager.get_handle(_handle_args())

    assert events == [
        ("allocate", (16, 128, 4, 7168, "nvfp4")),
        (
            "check",
            "receive-view allocation",
            "OutOfMemoryError: receive views",
        ),
    ]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"max_num_tokens": 0}, "must be positive"),
        ({"top_k": 3}, "even top_k"),
        ({"hidden": 65}, "multiple of 64"),
        ({"num_experts": 143}, "divide evenly"),
        ({"combine_quant": 2}, "must be 0 or 1"),
        ({"zero_copy_combine": 2}, "must be 0 or 1"),
        ({"max_num_tokens": 65535}, "max_num_tokens < 65535"),
        ({"dispatch_quantization": "fp8_per_token"}, "dispatch_quantization"),
    ],
)
def test_manager_rejects_invalid_native_configs_before_collective_setup(
    overrides,
    match,
):
    manager = _make_manager()

    with pytest.raises(ValueError, match=match):
        manager.get_handle(_handle_args(**overrides))


def test_manager_destroys_every_context_once_in_creation_order(monkeypatch):
    manager = _make_manager()
    destroyed = []

    class _OrderedHandle(_FakeGinHandle):
        def destroy(self):
            super().destroy()
            destroyed.append(self.key)

    first = _OrderedHandle("first")
    second = _OrderedHandle("second")
    manager._handles = {("first",): first, ("second",): second}
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    manager.destroy()
    manager.destroy()

    assert destroyed == ["first", "second"]
    assert first.destroy_calls == 1
    assert second.destroy_calls == 1
    assert manager._handles == {}


def test_manager_retains_buffers_and_does_not_retry_after_teardown_failure(
    monkeypatch,
):
    manager = _make_manager()
    failure = RuntimeError("collective teardown failed")
    failed = _FakeGinHandle("failed", destroy_error=failure)
    later = _FakeGinHandle("later")
    manager._handles = {("failed",): failed, ("later",): later}
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    with pytest.raises(RuntimeError, match="failed to shut down"):
        manager.destroy()
    manager.destroy()

    assert failed.destroy_calls == 1
    assert later.destroy_calls == 1
    assert manager._handles


def test_uid_is_generated_only_on_ep_root_and_broadcast_within_group(monkeypatch):
    manager = _make_manager()
    del manager._new_uid
    manager.rank = 0
    uid = torch.arange(128, dtype=torch.uint8)
    module = object()
    broadcasts = []
    checks = []
    uid_modules = []

    def _check(phase, error):
        checks.append((phase, error))

    manager.collective_error_check = _check
    monkeypatch.setattr(
        all2all_module,
        "gin_moe_get_unique_id",
        lambda *, module: uid_modules.append(module) or uid,
        raising=False,
    )
    monkeypatch.setattr(
        all2all_module.dist,
        "get_process_group_ranks",
        lambda group: [8 + rank for rank in range(16)],
    )
    monkeypatch.setattr(
        all2all_module.dist,
        "broadcast",
        lambda tensor, src, group: broadcasts.append((tensor, src, group)),
    )

    result = manager._new_uid(module)

    assert result is uid
    assert uid_modules == [module]
    assert checks == [("unique-ID generation", None)]
    assert broadcasts == [(uid, 8, manager.cpu_group)]


def test_shape_specialized_module_is_preloaded_collectively(monkeypatch):
    manager = _make_manager()
    del manager._preload_module
    module = object()
    preload_calls = []
    checks = []

    class _FakeGinMoeAlltoAll:
        @classmethod
        def preload(cls, **kwargs):
            preload_calls.append(kwargs)
            return module

    monkeypatch.setattr(
        all2all_module,
        "GinMoeAlltoAll",
        _FakeGinMoeAlltoAll,
        raising=False,
    )
    manager.collective_error_check = lambda phase, error: checks.append((phase, error))

    result = manager._preload_module(4, 7168, 128)

    assert result is module
    assert preload_calls == [
        {
            "rank": 3,
            "ep_size": 16,
            "gpus_per_node": 8,
            "top_k": 4,
            "hidden": 7168,
            "num_experts": 128,
        }
    ]
    assert checks == [("JIT preload", None)]


def test_none_preload_is_reported_as_local_error_before_exchange(monkeypatch):
    manager = _make_manager()
    del manager._preload_module
    checks = []

    class _FakeGinMoeAlltoAll:
        @classmethod
        def preload(cls, **kwargs):
            return None

        @classmethod
        def allocate_recv_views(cls, *args):
            raise AssertionError("receive allocation must not run")

        @classmethod
        def create(cls, **kwargs):
            raise AssertionError("native setup must not run")

    monkeypatch.setattr(
        all2all_module,
        "GinMoeAlltoAll",
        _FakeGinMoeAlltoAll,
        raising=False,
    )
    manager.collective_error_check = lambda phase, error: checks.append((phase, error))

    def _unexpected_uid(_module):
        raise AssertionError("UID generation must not run")

    manager._new_uid = _unexpected_uid

    with pytest.raises(RuntimeError, match="JIT preload returned no module"):
        manager.get_handle(_handle_args())

    assert checks == [
        (
            "JIT preload",
            "RuntimeError: flashinfer_gin JIT preload returned no module.",
        )
    ]


def test_collective_error_check_reports_every_rank_before_native_setup(
    monkeypatch,
):
    manager = _make_manager()
    del manager.collective_error_check

    def _all_gather_object(output, value, group):
        assert value is None
        assert group is manager.cpu_group
        output[:] = [None] * manager.world_size
        output[5] = "RuntimeError: JIT failed"

    monkeypatch.setattr(
        all2all_module.dist,
        "all_gather_object",
        _all_gather_object,
    )

    with pytest.raises(RuntimeError, match="rank 5: RuntimeError: JIT failed"):
        manager.collective_error_check("JIT preload", None)


def test_topology_discovery_accepts_only_full_node_major_eight_gpu_nodes(
    monkeypatch,
):
    import vllm.distributed.parallel_state as parallel_state

    manager = _make_manager()
    node_ranks = {
        0: list(range(8)),
        8: list(range(8, 16)),
    }
    calls = []

    def _in_the_same_node_as(group, source_rank):
        assert group is manager.cpu_group
        calls.append(source_rank)
        return [rank in node_ranks[source_rank] for rank in range(16)]

    monkeypatch.setattr(
        parallel_state,
        "in_the_same_node_as",
        _in_the_same_node_as,
    )

    assert manager._validate_rank_topology() == 8
    assert calls == [0, 8]

    node_ranks[0] = list(range(7)) + [8]
    with pytest.raises(ValueError, match="node-major"):
        manager._validate_rank_topology()


@pytest.mark.parametrize("enable_eplb", [False, True])
def test_parallel_config_selects_flashinfer_gin_kernels(enable_eplb):
    config = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=16,
        ep_size=16,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="flashinfer_gin",
        enable_eplb=enable_eplb,
    )

    assert config.use_all2all_kernels
    assert config.use_fi_gin_kernels
    assert not config.use_fi_nvl_one_sided_kernels
    assert config.enable_eplb is enable_eplb


def _make_flashinfer_gin_eplb_parallel_config(**overrides):
    kwargs = dict(
        data_parallel_size=16,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config=EPLBConfig(
            num_redundant_experts=16,
            communicator="torch_gloo",
        ),
        all2all_backend="flashinfer_gin",
        expert_placement_strategy="linear",
    )
    kwargs.update(overrides)
    return ParallelConfig(**kwargs)


def test_parallel_config_allows_flashinfer_gin_with_eplb(monkeypatch):
    monkeypatch.setattr(
        parallel_config_module.current_platform,
        "is_cuda_alike",
        lambda: True,
    )

    config = _make_flashinfer_gin_eplb_parallel_config()

    assert config.enable_eplb
    assert config.eplb_config.num_redundant_experts == 16
    assert config.eplb_config.communicator == "torch_gloo"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"expert_placement_strategy": "round_robin"}, "requires linear"),
        ({"enable_elastic_ep": True}, "does not support elastic EP"),
    ],
)
def test_parallel_config_rejects_unsupported_flashinfer_gin_eplb_modes(
    monkeypatch, overrides, match
):
    monkeypatch.setattr(
        parallel_config_module.current_platform,
        "is_cuda_alike",
        lambda: True,
    )

    with pytest.raises(ValueError, match=match):
        _make_flashinfer_gin_eplb_parallel_config(**overrides)


def test_flashinfer_gin_eplb_uses_linear_physical_expert_ownership():
    moe = SimpleNamespace(num_experts=144, num_local_experts=9)
    assert all2all_utils_module._validate_flashinfer_gin_expert_layout(moe, 16) == 144

    for ep_rank in range(16):
        local_count, expert_map, _ = determine_expert_map(
            ep_size=16,
            ep_rank=ep_rank,
            global_num_experts=144,
            expert_placement_strategy="linear",
        )

        assert local_count == 9
        assert expert_map is not None
        physical_ids = torch.where(expert_map >= 0)[0]
        assert torch.equal(
            physical_ids,
            torch.arange(ep_rank * 9, (ep_rank + 1) * 9),
        )
        assert torch.equal(
            expert_map[physical_ids],
            torch.arange(9, dtype=torch.int32),
        )


@pytest.mark.parametrize(
    ("num_experts", "num_local_experts", "match"),
    [
        (143, 9, "divide evenly"),
        (144, 8, "expected 9 physical experts per EP rank, got 8"),
        (144, 10, "fused shared expert slots are not supported"),
    ],
)
def test_flashinfer_gin_rejects_inconsistent_physical_expert_layout(
    num_experts, num_local_experts, match
):
    moe = SimpleNamespace(
        num_experts=num_experts,
        num_local_experts=num_local_experts,
    )

    with pytest.raises(ValueError, match=match):
        all2all_utils_module._validate_flashinfer_gin_expert_layout(moe, 16)


@pytest.mark.parametrize(
    ("moe_backend", "dispatch_quantization"),
    [
        ("flashinfer_trtllm", "nvfp4"),
        ("flashinfer_trtllm", "fp8_per_tensor"),
        ("flashinfer_cutedsl", "nvfp4"),
    ],
)
def test_flashinfer_gin_accepts_compatible_moe_backends(
    monkeypatch, moe_backend, dispatch_quantization
):
    monkeypatch.setattr(
        all2all_utils_module,
        "has_flashinfer_trtllm_fused_moe",
        lambda: True,
    )
    monkeypatch.setattr(
        all2all_utils_module,
        "has_flashinfer_cutedsl_moe_nvfp4",
        lambda: True,
    )

    all2all_utils_module._validate_flashinfer_gin_moe_backend(
        moe_backend,
        dispatch_quantization,
    )


@pytest.mark.parametrize(
    ("moe_backend", "dispatch_quantization", "match"),
    [
        ("flashinfer_cutedsl", "fp8_per_tensor", "flashinfer_trtllm"),
        ("flashinfer_cutlass", "nvfp4", "flashinfer_cutedsl"),
    ],
)
def test_flashinfer_gin_rejects_incompatible_moe_backends(
    monkeypatch, moe_backend, dispatch_quantization, match
):
    monkeypatch.setattr(
        all2all_utils_module,
        "has_flashinfer_trtllm_fused_moe",
        lambda: True,
    )
    monkeypatch.setattr(
        all2all_utils_module,
        "has_flashinfer_cutedsl_moe_nvfp4",
        lambda: True,
    )

    with pytest.raises(ValueError, match=match):
        all2all_utils_module._validate_flashinfer_gin_moe_backend(
            moe_backend,
            dispatch_quantization,
        )


def test_flashinfer_gin_checks_selected_backend_capability(monkeypatch):
    monkeypatch.setattr(
        all2all_utils_module,
        "has_flashinfer_cutedsl_moe_nvfp4",
        lambda: False,
    )

    with pytest.raises(ValueError, match="CuteDSL NVFP4 MoE"):
        all2all_utils_module._validate_flashinfer_gin_moe_backend(
            "flashinfer_cutedsl",
            "nvfp4",
        )


def test_flashinfer_gin_validates_the_model_scoped_moe_backend(monkeypatch):
    """EPLB draft validation uses its backend and physical expert count."""
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        tp_rank=0,
        pcp_size=1,
        pcp_rank=0,
        dp_size=16,
        dp_rank=0,
        ep_size=16,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="flashinfer_gin",
        enable_eplb=True,
    )
    moe = FusedMoEConfig(
        num_experts=144,
        experts_per_token=4,
        hidden_dim=64,
        intermediate_size=128,
        num_local_experts=9,
        num_logical_experts=128,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        moe_backend="flashinfer_trtllm",
        max_num_tokens=32,
    )
    outer_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="flashinfer_cutedsl"),
        parallel_config=SimpleNamespace(
            expert_placement_strategy="linear",
            flashinfer_gin_max_num_tokens=None,
            flashinfer_gin_combine_quant=False,
            flashinfer_gin_zero_copy_combine=True,
        ),
    )
    manager = SimpleNamespace(
        world_size=16,
        get_handle=lambda config: ("handle", config),
        collective_error_check=lambda phase, error: None,
    )
    validated_backends = []

    class _FakePrepareFinalize:
        @staticmethod
        def validate_quant_config(quant_config):
            return "fp8_per_tensor", torch.tensor(0.5)

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(
        all2all_utils_module,
        "get_current_vllm_config",
        lambda: outer_config,
    )
    monkeypatch.setattr(
        all2all_utils_module,
        "get_ep_all2all_manager",
        lambda eep_stage=False: manager,
    )
    monkeypatch.setattr(
        all2all_utils_module,
        "FlashInferGinPrepareAndFinalize",
        _FakePrepareFinalize,
    )
    monkeypatch.setattr(
        all2all_utils_module,
        "_validate_flashinfer_gin_moe_backend",
        lambda backend, quantization: validated_backends.append(
            (backend, quantization)
        ),
    )

    result = all2all_utils_module.maybe_make_prepare_finalize(
        moe,
        SimpleNamespace(use_nvfp4_w4a4=False, use_fp8_w8a8=True),
        allow_new_interface=True,
    )

    assert isinstance(result, _FakePrepareFinalize)
    assert validated_backends == [("flashinfer_trtllm", "fp8_per_tensor")]
    assert result.args[0][1]["num_experts"] == 144
    assert moe.num_local_experts == result.args[0][1]["num_experts"] // 16 == 9
    assert result.args[0][1]["zero_copy_combine"] == 1
    assert "dispatch_transport" not in result.args[0][1]
    assert result.kwargs["num_experts"] == 144
    assert result.kwargs["zero_copy_combine"] is True


def test_capability_requires_nvcc_even_with_flashinfer_cubins(monkeypatch):
    import vllm.utils.flashinfer as flashinfer_utils

    flashinfer_utils.has_flashinfer_gin.cache_clear()
    monkeypatch.setattr(flashinfer_utils, "has_flashinfer_comm", lambda: True)
    monkeypatch.setattr(flashinfer_utils.shutil, "which", lambda command: None)

    assert not flashinfer_utils.has_flashinfer_gin()
    flashinfer_utils.has_flashinfer_gin.cache_clear()


@pytest.mark.parametrize(
    ("quant_config", "match"),
    [
        (
            SimpleNamespace(use_nvfp4_w4a4=False, use_fp8_w8a8=False),
            "static NVFP4 or static per-tensor FP8",
        ),
        (
            SimpleNamespace(
                use_nvfp4_w4a4=False,
                use_fp8_w8a8=True,
                per_act_token_quant=True,
            ),
            "per-token FP8",
        ),
        (
            SimpleNamespace(
                use_nvfp4_w4a4=False,
                use_fp8_w8a8=True,
                per_act_token_quant=False,
                is_block_quantized=True,
            ),
            "blockwise FP8",
        ),
        (
            SimpleNamespace(
                use_nvfp4_w4a4=False,
                use_fp8_w8a8=True,
                per_act_token_quant=False,
                is_block_quantized=False,
                is_per_tensor=False,
            ),
            "per-tensor FP8",
        ),
        (
            SimpleNamespace(
                use_nvfp4_w4a4=False,
                use_fp8_w8a8=True,
                per_act_token_quant=False,
                is_block_quantized=False,
                is_per_tensor=True,
                a1_scale=None,
            ),
            "FP8 activation scale",
        ),
        (
            SimpleNamespace(
                use_nvfp4_w4a4=True,
                a1_gscale=None,
                is_scale_swizzled=False,
            ),
            "activation gscale",
        ),
        (
            SimpleNamespace(
                use_nvfp4_w4a4=True,
                a1_gscale=torch.ones(1),
                is_scale_swizzled=True,
            ),
            "linear activation scale factors",
        ),
        (
            SimpleNamespace(
                use_nvfp4_w4a4=True,
                a1_gscale=torch.ones(1),
                is_scale_swizzled=False,
            ),
            "must be on CUDA",
        ),
    ],
)
def test_prepare_finalize_rejects_unsupported_quantization(quant_config, match):
    with pytest.raises(ValueError, match=match):
        FlashInferGinPrepareAndFinalize(
            SimpleNamespace(),
            quant_config,
            num_experts=16,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([1.0, float("nan")], "finite"),
        ([1.0, 0.0], "positive"),
        ([1.0, 2.0], "shared by all experts"),
    ],
)
def test_prepare_finalize_rejects_invalid_activation_gscale(values, match):
    quant_config = SimpleNamespace(
        use_nvfp4_w4a4=True,
        per_act_token_quant=False,
        a1_gscale=torch.tensor(values, dtype=torch.float32, device="cuda"),
        is_scale_swizzled=False,
    )

    with pytest.raises(ValueError, match=match):
        FlashInferGinPrepareAndFinalize.validate_quant_config(quant_config)


def _fp8_quant_config(values=(0.25,)):
    return SimpleNamespace(
        use_nvfp4_w4a4=False,
        use_fp8_w8a8=True,
        per_act_token_quant=False,
        is_block_quantized=False,
        is_per_tensor=True,
        a1_scale=torch.tensor(values, dtype=torch.float32, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([1.0, float("inf")], "finite"),
        ([1.0, -1.0], "positive"),
        ([1.0, 2.0], "shared by all experts"),
    ],
)
def test_prepare_finalize_rejects_invalid_fp8_activation_scale(values, match):
    with pytest.raises(ValueError, match=match):
        FlashInferGinPrepareAndFinalize.validate_quant_config(_fp8_quant_config(values))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_static_per_tensor_fp8_selects_the_fp8_dispatch_format():
    quantization, scale = FlashInferGinPrepareAndFinalize.validate_quant_config(
        _fp8_quant_config((0.25,))
    )

    assert quantization == "fp8_per_tensor"
    # a1_scale is forwarded verbatim; the kernel takes its reciprocal, so
    # inverting it here would double-invert the divisor.
    assert scale.dtype == torch.float32
    assert scale.numel() == 1
    assert float(scale.item()) == 0.25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_selects_the_nvfp4_dispatch_format():
    quant_config = SimpleNamespace(
        use_nvfp4_w4a4=True,
        per_act_token_quant=False,
        a1_gscale=torch.full((1,), 3.0, dtype=torch.float32, device="cuda"),
        is_scale_swizzled=False,
    )

    quantization, scale = FlashInferGinPrepareAndFinalize.validate_quant_config(
        quant_config
    )

    assert quantization == "nvfp4"
    assert float(scale.item()) == 3.0


def test_same_dimension_handles_differ_by_dispatch_format(monkeypatch):
    """The EAGLE deployment's critical case.

    With EPLB, main and draft agree on 144 physical experts as well as hidden,
    top_k, EP, and token bound; only the wire format differs. Without it in the
    key the draft silently reuses the main model's NVFP4 handle and its
    differently sized receive buffers.
    """
    create_calls = []

    class _FakeGinMoeAlltoAll:
        @classmethod
        def allocate_recv_views(cls, *args):
            return ("recv_views", args)

        @classmethod
        def create(cls, **kwargs):
            create_calls.append(kwargs)
            return _FakeGinHandle(len(create_calls))

    monkeypatch.setattr(
        all2all_module,
        "GinMoeAlltoAll",
        _FakeGinMoeAlltoAll,
        raising=False,
    )
    manager = _make_manager()

    nvfp4 = manager.get_handle(
        _handle_args(num_experts=144, dispatch_quantization="nvfp4")
    )
    fp8 = manager.get_handle(
        _handle_args(num_experts=144, dispatch_quantization="fp8_per_tensor")
    )
    fp8_again = manager.get_handle(
        _handle_args(num_experts=144, dispatch_quantization="fp8_per_tensor")
    )

    assert nvfp4 is not fp8
    assert fp8_again is fp8
    assert len(create_calls) == 2
    assert [call["dispatch_quantization"] for call in create_calls] == [
        "nvfp4",
        "fp8_per_tensor",
    ]
    assert [call["num_experts"] for call in create_calls] == [144, 144]
    assert [call["num_experts"] // call["ep_size"] for call in create_calls] == [
        9,
        9,
    ]
    # Each context allocates its own receive views for its own format.
    assert create_calls[0]["recv_views"] == ("recv_views", (16, 128, 4, 7168, "nvfp4"))
    assert create_calls[1]["recv_views"] == (
        "recv_views",
        (16, 128, 4, 7168, "fp8_per_tensor"),
    )
    assert len(manager._handles) == 2


class _FakeZeroCopyA2A:
    ep_size = 2
    top_k = 4
    hidden = 64

    def __init__(self, combine_quant):
        self.combine_quant = combine_quant
        self.registered = torch.empty(
            (self.ep_size, 4, self.hidden),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.registered_calls = []
        self.combine_calls = []

    def dispatch_async(
        self,
        topk_ids,
        activations,
        gscale,
        topk_weights,
        runtime_max_tokens,
        invalid_token_expert_id,
    ):
        assert runtime_max_tokens == 4
        return (
            (
                torch.empty(
                    (self.ep_size, runtime_max_tokens, self.hidden // 2),
                    dtype=torch.uint8,
                    device="cuda",
                ),
                torch.empty(
                    (self.ep_size, runtime_max_tokens, self.hidden // 16),
                    dtype=torch.uint8,
                    device="cuda",
                ),
                torch.full(
                    (self.ep_size, runtime_max_tokens, self.top_k),
                    -1,
                    dtype=torch.int32,
                    device="cuda",
                ),
                torch.zeros(
                    (self.ep_size, runtime_max_tokens, self.top_k),
                    dtype=torch.float32,
                    device="cuda",
                ),
            ),
            torch.zeros(self.ep_size, dtype=torch.int32, device="cuda"),
        )

    def get_registered_combine_input(self, runtime_max_tokens):
        self.registered_calls.append(runtime_max_tokens)
        return self.registered[:, :runtime_max_tokens]

    def combine_async(self, *args):
        self.combine_calls.append(("staged", args))

    def combine_zero_copy_async(self, *args):
        self.combine_calls.append(("zero_copy", args))


def _prepare_zero_copy_adapter(monkeypatch, *, combine_quant, zero_copy):
    a2a = _FakeZeroCopyA2A(combine_quant)
    quant_config = SimpleNamespace()
    prepare_finalize = FlashInferGinPrepareAndFinalize(
        a2a,
        quant_config,
        num_experts=16,
        gscale=torch.ones(1, dtype=torch.float32, device="cuda"),
        dispatch_quantization="nvfp4",
        zero_copy_combine=zero_copy,
    )
    monkeypatch.setattr(
        pf_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [3, 4])
        ),
    )
    activations = torch.randn((3, 64), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.zeros((3, 4), dtype=torch.int32, device="cuda")
    topk_weights = torch.rand((3, 4), dtype=torch.float32, device="cuda")
    prepare_finalize.prepare(
        activations,
        topk_weights,
        topk_ids,
        16,
        None,
        False,
        quant_config,
    )
    return prepare_finalize, a2a, activations, topk_weights, topk_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_raw_zero_copy_uses_registered_expert_output(monkeypatch):
    prepare_finalize, a2a, activations, topk_weights, topk_ids = (
        _prepare_zero_copy_adapter(
            monkeypatch,
            combine_quant=0,
            zero_copy=True,
        )
    )

    fused_output = prepare_finalize.fused_expert_output_buffer()

    assert fused_output is not None
    assert fused_output.shape == (8, 64)
    assert fused_output.is_contiguous()
    assert fused_output.data_ptr() == a2a.registered.data_ptr()
    assert a2a.registered_calls == [4]

    output = torch.empty_like(activations)
    prepare_finalize.finalize(
        output,
        fused_output,
        topk_weights,
        topk_ids,
        False,
        TopKWeightAndReduceNoOP(),
    )

    assert len(a2a.combine_calls) == 1
    mode, combine_args = a2a.combine_calls[0]
    payload, combined_output, local_tokens, runtime_max, _ = combine_args
    assert mode == "zero_copy"
    assert payload.shape == (2, 4, 64)
    assert payload.data_ptr() == a2a.registered.data_ptr()
    assert combined_output is output
    assert (local_tokens, runtime_max) == (3, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_raw_zero_copy_rejects_unregistered_expert_output(monkeypatch):
    prepare_finalize, _, activations, topk_weights, topk_ids = (
        _prepare_zero_copy_adapter(
            monkeypatch,
            combine_quant=0,
            zero_copy=True,
        )
    )
    registered = prepare_finalize.fused_expert_output_buffer()
    assert registered is not None

    with pytest.raises(RuntimeError, match="exact registered"):
        prepare_finalize.finalize(
            torch.empty_like(activations),
            registered.clone(),
            topk_weights,
            topk_ids,
            False,
            TopKWeightAndReduceNoOP(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantized_zero_copy_borrows_ordinary_expert_output(monkeypatch):
    prepare_finalize, a2a, activations, topk_weights, topk_ids = (
        _prepare_zero_copy_adapter(
            monkeypatch,
            combine_quant=1,
            zero_copy=True,
        )
    )
    fused_output = torch.full(
        (8, 64),
        2.0,
        dtype=torch.bfloat16,
        device="cuda",
    )

    assert prepare_finalize.fused_expert_output_buffer() is None
    prepare_finalize.finalize(
        torch.empty_like(activations),
        fused_output,
        topk_weights,
        topk_ids,
        False,
        TopKWeightAndReduceNoOP(),
    )

    assert a2a.registered_calls == []
    assert len(a2a.combine_calls) == 1
    mode, combine_args = a2a.combine_calls[0]
    payload, _, _, _, cscale = combine_args
    assert mode == "zero_copy"
    assert payload.data_ptr() == fused_output.data_ptr()
    assert float(cscale.item()) == pytest.approx((448.0 * 6.0) / 2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("combine_quant", [0, 1])
def test_zero_copy_disabled_keeps_staged_combine(monkeypatch, combine_quant):
    prepare_finalize, a2a, activations, topk_weights, topk_ids = (
        _prepare_zero_copy_adapter(
            monkeypatch,
            combine_quant=combine_quant,
            zero_copy=False,
        )
    )
    fused_output = torch.ones(
        (8, 64),
        dtype=torch.bfloat16,
        device="cuda",
    )

    assert prepare_finalize.fused_expert_output_buffer() is None
    prepare_finalize.finalize(
        torch.empty_like(activations),
        fused_output,
        topk_weights,
        topk_ids,
        False,
        TopKWeightAndReduceNoOP(),
    )

    assert a2a.registered_calls == []
    assert [mode for mode, _ in a2a.combine_calls] == ["staged"]


class _FakeRequiredOutputPrepareFinalize:
    def __init__(self, required_output):
        self.required_output = required_output

    def fused_expert_output_buffer(self):
        return self.required_output


class _FakeRequiredOutputExperts:
    moe_config = SimpleNamespace(moe_parallel_config=None)
    a2_scale = None

    def __init__(self):
        self.outputs = []

    def moe_problem_size(self, a1q, w1, w2, topk_ids):
        return 2, 8, 128, 64, 4

    def apply(self, *, output, **kwargs):
        self.outputs.append(output)


def _run_fused_experts_with_required_output(monkeypatch, required_output):
    prepare_finalize = _FakeRequiredOutputPrepareFinalize(required_output)
    experts = _FakeRequiredOutputExperts()
    kernel = mk_module.FusedMoEKernelModularImpl(prepare_finalize, experts)
    fallback = torch.empty((8, 64), dtype=torch.bfloat16)
    monkeypatch.setattr(
        kernel,
        "_allocate_buffers",
        lambda *args, **kwargs: (torch.empty(0), torch.empty(0), fallback),
    )
    result = kernel._fused_experts(
        in_dtype=torch.bfloat16,
        a1q=torch.empty((8, 64), dtype=torch.uint8),
        a1q_scale=None,
        w1=torch.empty((2, 128, 64)),
        w2=torch.empty((2, 64, 128)),
        topk_weights=torch.empty((8, 4)),
        topk_ids=torch.empty((8, 4), dtype=torch.int32),
        activation=MoEActivation.SILU,
        global_num_experts=16,
        local_num_experts=2,
        expert_map=None,
        apply_router_weight_on_input=False,
        expert_tokens_meta=None,
    )
    return result, experts


def test_modular_kernel_honors_required_expert_output_buffer(monkeypatch):
    required_output = torch.empty((8, 64), dtype=torch.bfloat16)

    result, experts = _run_fused_experts_with_required_output(
        monkeypatch,
        required_output,
    )

    assert result is required_output
    assert len(experts.outputs) == 1
    assert experts.outputs[0] is required_output


def test_modular_kernel_rejects_invalid_required_expert_output_buffer(monkeypatch):
    required_output = torch.empty((7, 64), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="invalid fused-expert output buffer"):
        _run_fused_experts_with_required_output(monkeypatch, required_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepare_finalize_dispatches_eplb_physical_expert_ids(monkeypatch):
    class _FakeA2A:
        ep_size = 16
        top_k = 4
        hidden = 64
        combine_quant = 0

        def __init__(self):
            self.dispatch_calls = []
            self.combine_calls = []

        def dispatch_async(
            self,
            topk_ids,
            activations,
            gscale,
            topk_weights,
            runtime_max_tokens,
            invalid_token_expert_id,
        ):
            self.dispatch_calls.append(
                (
                    topk_ids,
                    activations,
                    gscale,
                    topk_weights,
                    runtime_max_tokens,
                    invalid_token_expert_id,
                )
            )
            return (
                (
                    torch.empty(
                        (self.ep_size, runtime_max_tokens, 32),
                        dtype=torch.uint8,
                        device="cuda",
                    ),
                    torch.empty(
                        (self.ep_size, runtime_max_tokens, 4),
                        dtype=torch.uint8,
                        device="cuda",
                    ),
                    torch.full(
                        (self.ep_size, runtime_max_tokens, self.top_k),
                        -1,
                        dtype=torch.int32,
                        device="cuda",
                    ),
                    torch.zeros(
                        (self.ep_size, runtime_max_tokens, self.top_k),
                        dtype=torch.float32,
                        device="cuda",
                    ),
                ),
                torch.zeros(self.ep_size, dtype=torch.int32, device="cuda"),
            )

        def combine_async(
            self,
            payload,
            output,
            local_num_tokens,
            runtime_max_tokens,
            cscale,
        ):
            self.combine_calls.append(
                (
                    payload,
                    output,
                    local_num_tokens,
                    runtime_max_tokens,
                    cscale,
                )
            )

    a2a = _FakeA2A()
    quant_config = SimpleNamespace(
        use_nvfp4_w4a4=True,
        a1_gscale=torch.ones(1, dtype=torch.float32, device="cuda"),
        is_scale_swizzled=False,
    )
    prepare_finalize = FlashInferGinPrepareAndFinalize(
        a2a,
        quant_config,
        num_experts=144,
    )
    dp_metadata = SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [3] + [4] * 15)
    monkeypatch.setattr(
        pf_module,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=dp_metadata),
    )

    activations = torch.randn(
        (3, 64),
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_ids = torch.tensor(
        [[0, 127, 128, 143], [8, 9, -1, -1], [126, 135, 136, 137]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.rand((3, 4), dtype=torch.float32, device="cuda")
    expert_map = torch.full((144,), -1, dtype=torch.int32, device="cuda")
    expert_map[135:144] = torch.arange(9, dtype=torch.int32, device="cuda")

    mantissa, scales, metadata, recv_ids, recv_weights = prepare_finalize.prepare(
        activations,
        topk_weights,
        topk_ids,
        144,
        expert_map,
        False,
        quant_config,
    )

    assert mantissa.shape == (64, 32)
    assert scales is not None and scales.shape == (64, 4)
    assert metadata is None
    assert recv_ids is not None and recv_ids.shape == (64, 4)
    assert recv_weights is not None and recv_weights.shape == (64, 4)
    assert a2a.dispatch_calls[0][0] is topk_ids
    assert a2a.dispatch_calls[0][4:] == (4, -1)

    output = torch.empty_like(activations)
    fused_output = torch.empty((64, 64), dtype=torch.bfloat16, device="cuda")
    prepare_finalize.finalize(
        output,
        fused_output,
        topk_weights,
        topk_ids,
        False,
        TopKWeightAndReduceNoOP(),
    )

    assert len(a2a.combine_calls) == 1
    payload, combined_output, local_tokens, runtime_max, _ = a2a.combine_calls[0]
    assert payload.shape == (16, 4, 64)
    assert combined_output is output
    assert (local_tokens, runtime_max) == (3, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_prepare_forwards_a1_scale_and_returns_no_scale_payload(monkeypatch):
    class _FakeFp8A2A:
        ep_size = 2
        top_k = 4
        hidden = 64
        combine_quant = 0

        def __init__(self):
            self.dispatch_calls = []

        def dispatch_async(
            self,
            topk_ids,
            activations,
            gscale,
            topk_weights,
            runtime_max_tokens,
            invalid_token_expert_id,
        ):
            self.dispatch_calls.append(gscale)
            return (
                (
                    torch.empty(
                        (self.ep_size, runtime_max_tokens, self.hidden),
                        dtype=torch.float8_e4m3fn,
                        device="cuda",
                    ),
                    None,  # per-tensor FP8 ships no per-token scale
                    torch.full(
                        (self.ep_size, runtime_max_tokens, self.top_k),
                        -1,
                        dtype=torch.int32,
                        device="cuda",
                    ),
                    torch.zeros(
                        (self.ep_size, runtime_max_tokens, self.top_k),
                        dtype=torch.float32,
                        device="cuda",
                    ),
                ),
                torch.zeros(self.ep_size, dtype=torch.int32, device="cuda"),
            )

    a2a = _FakeFp8A2A()
    quant_config = _fp8_quant_config((0.5,))
    prepare_finalize = FlashInferGinPrepareAndFinalize(
        a2a,
        quant_config,
        num_experts=16,
    )
    monkeypatch.setattr(
        pf_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [3, 4])
        ),
    )

    activation, scales, metadata, recv_ids, recv_weights = prepare_finalize.prepare(
        torch.randn((3, 64), dtype=torch.bfloat16, device="cuda"),
        torch.rand((3, 4), dtype=torch.float32, device="cuda"),
        torch.zeros((3, 4), dtype=torch.int32, device="cuda"),
        16,
        None,
        False,
        quant_config,
    )

    assert activation.shape == (8, 64)
    assert activation.dtype == torch.float8_e4m3fn
    assert scales is None
    assert metadata is None
    assert recv_ids.shape == (8, 4)
    assert recv_weights.shape == (8, 4)
    # The DIVISOR reaches the dispatch unmodified.
    assert float(a2a.dispatch_calls[0].item()) == 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("dispatch_quantization", "expert_module", "expert_class", "match"),
    [
        (
            "fp8_per_tensor",
            "vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe",
            "TrtLlmNvFp4ExpertsModular",
            "FP8 dispatch requires",
        ),
        (
            "nvfp4",
            "vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe",
            "TrtLlmFp8ExpertsModular",
            "TrtLlmNvFp4ExpertsModular",
        ),
    ],
)
def test_post_init_setup_rejects_the_other_formats_expert(
    dispatch_quantization, expert_module, expert_class, match
):
    import importlib
    from unittest.mock import MagicMock

    # spec= makes isinstance() see the wrong expert class without constructing
    # one (they need a full FusedMoEConfig and loaded weights).
    experts = MagicMock(
        spec=getattr(importlib.import_module(expert_module), expert_class)
    )
    prepare_finalize = FlashInferGinPrepareAndFinalize.__new__(
        FlashInferGinPrepareAndFinalize
    )
    prepare_finalize._dispatch_quantization = dispatch_quantization
    prepare_finalize._quant_config = SimpleNamespace(is_per_tensor=True)

    with pytest.raises(ValueError, match=match):
        prepare_finalize.post_init_setup(experts)


def test_post_init_setup_accepts_cutedsl_for_nvfp4():
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (  # noqa: E501
        FlashInferCuteDSLExperts,
    )

    experts = FlashInferCuteDSLExperts.__new__(FlashInferCuteDSLExperts)
    prepare_finalize = FlashInferGinPrepareAndFinalize.__new__(
        FlashInferGinPrepareAndFinalize
    )
    prepare_finalize._dispatch_quantization = "nvfp4"
    prepare_finalize.post_init_setup(experts)
