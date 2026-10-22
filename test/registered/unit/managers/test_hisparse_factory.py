"""HiSparse device dispatch and unchanged native constructor arguments."""

import builtins
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).parents[4]
FACTORY = ROOT / "python/sglang/srt/managers/hisparse_factory.py"


@pytest.fixture
def factory(monkeypatch):
    context = types.ModuleType("sglang.srt.runtime_context")
    memory = SimpleNamespace(enable_hisparse=True)
    platform = SimpleNamespace(is_npu=False)
    parallel = SimpleNamespace(enable_dp_attention=False)
    context.get_memory = lambda: memory
    context.get_platform = lambda: platform
    context.get_parallel = lambda: parallel
    context.get_spec = lambda: SimpleNamespace(speculative_num_draft_tokens=6)
    monkeypatch.setitem(sys.modules, context.__name__, context)
    qualified = "sglang.srt.managers.hisparse_factory"
    spec = importlib.util.spec_from_file_location(qualified, FACTORY)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, qualified, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        module=module, memory=memory, platform=platform, parallel=parallel
    )


@pytest.mark.parametrize(
    "enabled,is_npu,expected",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
def test_native_capability_is_enabled_only_off_npu(factory, enabled, is_npu, expected):
    factory.memory.enable_hisparse = enabled
    factory.platform.is_npu = is_npu
    assert factory.module.uses_native_hisparse() is expected


def test_staging_defaults_to_native_but_excludes_missing_and_npu(factory):
    assert factory.module.uses_hisparse_staging(object())
    assert not factory.module.uses_hisparse_staging(None)
    assert not factory.module.uses_hisparse_staging(SimpleNamespace(uses_staging=False))


@pytest.mark.parametrize(
    "queue", ["disagg_prefill_bootstrap_queue", "disagg_decode_prealloc_queue"]
)
def test_pending_pd_layer_writer_prevents_npu_idle(factory, queue):
    pending = Mock(return_value=True)
    scheduler = SimpleNamespace(
        hisparse_coordinator=SimpleNamespace(uses_staging=False),
        **{
            queue: SimpleNamespace(
                kv_manager=SimpleNamespace(has_pending_layer_transfers=pending)
            )
        },
    )
    assert factory.module.has_pending_hisparse_transfers(scheduler)
    pending.return_value = False
    assert not factory.module.has_pending_hisparse_transfers(scheduler)
    scheduler.hisparse_coordinator = object()
    pending.reset_mock()
    assert not factory.module.has_pending_hisparse_transfers(scheduler)
    pending.assert_not_called()


def test_disabled_factory_does_not_construct_or_import_backends(factory):
    factory.memory.enable_hisparse = False
    assert factory.module.create_hisparse_coordinator(object()) is None


@pytest.mark.parametrize("native", [False, True])
def test_native_and_non_offload_shutdown_keep_existing_cache_cleanup(factory, native):
    coordinator = SimpleNamespace(destroy=Mock()) if native else None
    cache = SimpleNamespace(release_host_resources=Mock())
    scheduler = SimpleNamespace(hisparse_coordinator=coordinator, tree_cache=cache)
    factory.module.release_hisparse_resources(scheduler)
    cache.release_host_resources.assert_called_once_with()
    if coordinator is not None:
        coordinator.destroy.assert_called_once_with()


def test_npu_dispatch_does_not_import_native_kernels_or_config(factory, monkeypatch):
    factory.platform.is_npu = True
    runtime = types.ModuleType("sglang.srt.hardware_backend.npu.hisparse.runtime")
    runtime.NpuHiSparseCoordinator = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(
            (
                "sglang.srt.managers.hisparse_coordinator",
                "sglang.srt.mem_cache.sparsity",
                "sglang.kernels",
                "sglang.srt.hardware_backend.npu.hisparse.config",
            )
        ):
            raise AssertionError(f"NPU factory imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runner = object()
    result = factory.module.create_hisparse_coordinator(runner)
    assert result is runtime.NpuHiSparseCoordinator.return_value
    runtime.NpuHiSparseCoordinator.assert_called_once_with(runner)


@pytest.mark.parametrize("dp_attention", [False, True])
@pytest.mark.parametrize("model_top_k", [None, 2048])
def test_native_arguments_match_existing_model_runner(
    factory, monkeypatch, dp_attention, model_top_k
):
    native = types.ModuleType("sglang.srt.managers.hisparse_coordinator")
    native.HiSparseCoordinator = Mock(return_value=object())
    native.resolve_shared_index_layers = Mock(return_value=[False, True])
    monkeypatch.setitem(sys.modules, native.__name__, native)
    sparsity = types.ModuleType("sglang.srt.mem_cache.sparsity")
    config = SimpleNamespace(
        top_k=1024,
        device_buffer_size=4096,
        host_to_device_ratio=3,
        swap_in_block_size=960,
    )
    sparsity.parse_hisparse_config = Mock(return_value=config)
    group = SimpleNamespace(cpu_group=object())
    distributed = types.ModuleType("sglang.srt.distributed.parallel_state")
    distributed.get_dp_tp_group = lambda: group
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    monkeypatch.setitem(sys.modules, sparsity.__name__, sparsity)
    text_config = SimpleNamespace()
    if model_top_k is not None:
        text_config.index_topk = model_top_k
    runner = SimpleNamespace(
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        model_config=SimpleNamespace(hf_text_config=text_config),
        device="cuda",
        tp_group=SimpleNamespace(cpu_group=object()),
        attention_tp_group=SimpleNamespace(cpu_group=object()),
        ps=SimpleNamespace(pp_size=2),
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
    )
    factory.parallel.enable_dp_attention = dp_attention
    result = factory.module.create_hisparse_coordinator(runner)
    assert result is native.HiSparseCoordinator.return_value
    native.HiSparseCoordinator.assert_called_once_with(
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        top_k=model_top_k or config.top_k,
        device_buffer_size=config.device_buffer_size,
        device="cuda",
        tp_group=group.cpu_group,
        host_to_device_ratio=config.host_to_device_ratio,
        swap_in_block_size=config.swap_in_block_size,
        shared_index_layers=[False, True],
        num_draft_tokens=6,
    )
    native.resolve_shared_index_layers.assert_called_once_with(
        hf_text_config=text_config
    )


if __name__ == "__main__":
    run_pytest(__file__)
