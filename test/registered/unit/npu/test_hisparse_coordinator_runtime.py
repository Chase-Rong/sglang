"""NPU coordinator composition and request ownership with CPU device boundaries."""

import contextlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import test_hicache_memfabric_pool as fabric_tests
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

runtime = fabric_tests.runtime

ROOT = Path(__file__).parents[4]
RUNTIME = ROOT / "python/sglang/srt/hardware_backend/npu/hisparse/runtime.py"


@pytest.fixture
def runtime_module(monkeypatch):
    context = types.ModuleType("sglang.srt.runtime_context")
    context.get_disagg = lambda: SimpleNamespace(disaggregation_mode="null")
    monkeypatch.setitem(sys.modules, context.__name__, context)
    qualified = "sglang.srt.hardware_backend.npu.hisparse.runtime"
    spec = importlib.util.spec_from_file_location(qualified, RUNTIME)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, qualified, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    host_pool = SimpleNamespace(
        usable_size=384,
        available_size=Mock(return_value=128),
        free=Mock(side_effect=AssertionError("coordinator freed HiCache pages")),
        destroy=Mock(),
    )
    binding = SimpleNamespace(
        ready_event=None,
        release_device_pages=Mock(
            side_effect=AssertionError("coordinator released request references")
        ),
    )
    pool = SimpleNamespace(
        decode_host_backing=SimpleNamespace(
            hicache_pool=host_pool,
            page_binding=binding,
            pool=SimpleNamespace(close=Mock(), shared_group=None),
        )
    )
    allocator = SimpleNamespace(
        size=256,
        available_size=Mock(return_value=128),
        free=Mock(side_effect=AssertionError("coordinator freed logical pages")),
    )
    return SimpleNamespace(
        device="cpu",
        token_to_kv_pool=pool,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=SimpleNamespace(
            req_generation=torch.tensor([0, 7, 4], dtype=torch.int64),
            req_generation_epoch=0,
        ),
        prefill_offload_runtime=None,
    )


class Request:
    def __init__(self, slot=1):
        self.rid = "request"
        self.kv = SimpleNamespace(req_pool_idx=slot)
        self.retraction_count = 0
        self.hisparse_staging = False


class PlannerState:
    def __init__(self, pool):
        self.pool = pool
        self.generations = pool.req_generation.clone()
        self.pending = {0: object()}
        self.plan_versions = {0: object()}
        self.resident = torch.arange(6)
        self.refreshes = 0

    def refresh_generations(self):
        self.generations.copy_(self.pool.req_generation)
        self.refreshes += 1


def integrated_runtime():
    return SimpleNamespace(
        active=False,
        host_visible=None,
        drain=Mock(),
        release_request=Mock(
            side_effect=AssertionError("request release belongs to the existing owner")
        ),
    )


def test_constructor_only_adopts_resources(runtime_module, runner, monkeypatch):
    allocations = []
    original_zeros = torch.zeros

    def zeros(*args, **kwargs):
        allocations.append((args, kwargs))
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros)
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    assert coordinator.mem_pool_device is runner.token_to_kv_pool
    assert coordinator.req_to_token_pool is runner.req_to_token_pool
    assert coordinator.token_to_kv_pool_allocator is runner.token_to_kv_pool_allocator
    assert coordinator.prefill_offload_runtime is None
    assert coordinator.ack_staging_queue == []
    assert not coordinator.uses_staging
    assert len(allocations) == 1
    assert allocations[0][0] == (1,)
    assert coordinator.num_real_reqs.dtype == torch.int32
    address = coordinator.num_real_reqs.data_ptr()
    coordinator.num_real_reqs.fill_(2)
    assert coordinator.num_real_reqs.tolist() == [2]
    assert coordinator.num_real_reqs.data_ptr() == address


def test_initialize_prefill_delegates_once_and_adopts_runtime(
    runtime_module, runner, monkeypatch
):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    runtime = integrated_runtime()
    module = types.ModuleType("sglang.srt.hardware_backend.npu.hisparse.prefill")

    def initialize(model_runner):
        assert model_runner is runner
        model_runner.prefill_offload_runtime = runtime

    module.initialize_prefill_runtime = Mock(side_effect=initialize)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert coordinator.initialize_prefill() is runtime
    assert coordinator.initialize_prefill() is runtime
    module.initialize_prefill_runtime.assert_called_once_with(runner)


def test_initialize_prefill_reuses_runtime_already_supplied_by_runner(
    runtime_module, runner
):
    runtime = integrated_runtime()
    runner.prefill_offload_runtime = runtime
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    assert coordinator.initialize_prefill() is runtime


def test_bind_and_forward_preserve_existing_runtime_context(runtime_module, runner):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    entries = []

    @contextlib.contextmanager
    def forward(batch):
        entries.append(batch)
        yield
        entries.append("finished")

    runtime = integrated_runtime()
    runtime.forward = forward
    coordinator.bind_prefill_runtime(runtime)
    coordinator.bind_prefill_runtime(runtime)
    assert runner.token_to_kv_pool.prefill_offload_runtime is runtime
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
    with coordinator.prefill_offload_forward(batch):
        assert entries == [batch]
    assert entries == [batch, "finished"]
    batch.forward_mode.is_idle = lambda: True
    with coordinator.prefill_offload_forward(batch):
        pass
    assert entries == [batch, "finished"]
    with pytest.raises(RuntimeError, match="already has"):
        coordinator.bind_prefill_runtime(integrated_runtime())


def test_publication_waits_use_existing_events(runtime_module, runner):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    runtime = integrated_runtime()
    runtime.host_visible = object()
    coordinator.bind_prefill_runtime(runtime)
    binding = runner.token_to_kv_pool.decode_host_backing.page_binding
    binding.ready_event = runtime.host_visible
    stream = SimpleNamespace(wait_event=Mock())
    coordinator.set_decode_producer_stream(stream)
    coordinator.wait_for_pending_backup()
    stream.wait_event.assert_not_called()
    runner.prefill_offload_runtime = None
    coordinator.wait_for_pending_backup()
    coordinator.wait_for_pending_backup()
    stream.wait_event.assert_called_once_with(runtime.host_visible)
    binding.ready_event = object()
    stream.wait_event.reset_mock()
    coordinator.wait_for_pending_backup()
    stream.wait_event.assert_called_once_with(binding.ready_event)


@pytest.mark.parametrize(
    "method", ["request_finished", "retract_req", "abort_staging_request"]
)
def test_retirement_invalidates_identity_without_releasing_pages(
    runtime_module, runner, method
):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    runtime = integrated_runtime()
    coordinator.bind_prefill_runtime(runtime)
    state = PlannerState(runner.req_to_token_pool)
    runner.token_to_kv_pool.fsa_runtime_state = state
    resident_address = state.resident.data_ptr()
    generation_address = state.generations.data_ptr()
    request = Request()
    getattr(coordinator, method)(request)
    coordinator.request_finished(request)
    coordinator.retract_req(request)
    assert runner.req_to_token_pool.req_generation.tolist() == [0, 8, 4]
    assert state.generations.tolist() == [0, 7, 4]
    assert state.pending and state.plan_versions
    assert state.refreshes == 0
    assert state.resident.data_ptr() == resident_address
    assert state.generations.data_ptr() == generation_address
    runtime.drain.assert_not_called()
    runtime.release_request.assert_not_called()
    runner.token_to_kv_pool_allocator.free.assert_not_called()
    backing = runner.token_to_kv_pool.decode_host_backing
    backing.page_binding.release_device_pages.assert_not_called()
    backing.hicache_pool.free.assert_not_called()


def test_old_callbacks_do_not_retire_reused_slot(runtime_module, runner):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    old_request = Request()
    coordinator.request_finished(old_request)
    runner.req_to_token_pool.req_generation[1] += 1
    coordinator.request_finished(old_request)
    assert runner.req_to_token_pool.req_generation[1] == 9
    new_request = Request()
    coordinator.request_finished(new_request)
    assert runner.req_to_token_pool.req_generation[1] == 10


def test_retracted_request_can_retire_new_allocation(runtime_module, runner):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    request = Request()
    coordinator.retract_req(request)
    request.retraction_count += 1
    runner.req_to_token_pool.req_generation[1] += 1
    coordinator.request_finished(request)
    assert runner.req_to_token_pool.req_generation[1] == 10


@pytest.mark.parametrize("role", ["null", "prefill", "decode"])
@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("already_skipped", [False, True])
def test_finished_cache_policy_matches_pd_admission(
    runtime_module, runner, monkeypatch, role, cache_enabled, already_skipped
):
    context = sys.modules["sglang.srt.runtime_context"]
    monkeypatch.setattr(
        context,
        "get_disagg",
        lambda: SimpleNamespace(
            disaggregation_mode=role,
            disaggregation_decode_enable_radix_cache=cache_enabled,
        ),
    )
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    request = Request()
    request.skip_radix_cache_insert = already_skipped
    coordinator.request_finished(request)
    assert request.skip_radix_cache_insert == (
        already_skipped or role == "decode" and not cache_enabled
    )
    runner.token_to_kv_pool_allocator.free.assert_not_called()
    runner.token_to_kv_pool.decode_host_backing.hicache_pool.free.assert_not_called()


def test_old_callback_after_pool_reset_is_ignored(runtime_module, runner):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    request = Request()
    coordinator.request_finished(request)
    runner.req_to_token_pool.req_generation.zero_()
    runner.req_to_token_pool.req_generation_epoch += 1
    runner.req_to_token_pool.req_generation[1] += 1
    coordinator.request_finished(request)
    assert runner.req_to_token_pool.req_generation.tolist() == [0, 1, 0]


def test_inflight_prefill_blocks_retirement_until_drained(runtime_module, runner):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    runtime = SimpleNamespace(can_release_request=Mock(return_value=False))
    coordinator.bind_prefill_runtime(runtime)
    request = Request()
    with pytest.raises(RuntimeError, match="in-flight"):
        coordinator.request_finished(request)
    assert runner.req_to_token_pool.req_generation[1] == 7
    runtime.can_release_request.return_value = True
    coordinator.request_finished(request)
    assert runner.req_to_token_pool.req_generation[1] == 8


def test_other_integrated_prefill_does_not_force_retirement_synchronization(
    runtime_module, runner
):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    runtime = integrated_runtime()
    runtime.active = True
    coordinator.bind_prefill_runtime(runtime)
    coordinator.request_finished(Request())
    runtime.drain.assert_not_called()
    assert runner.req_to_token_pool.req_generation[1] == 8


@pytest.mark.parametrize("slot", [None, 0, -1, 3])
def test_retirement_guards_missing_and_padding_slots(runtime_module, runner, slot):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    if slot is None:
        coordinator.request_finished(Request(slot))
    else:
        with pytest.raises(ValueError, match="exclude padding"):
            coordinator.request_finished(Request(slot))
    assert runner.req_to_token_pool.req_generation.tolist() == [0, 7, 4]


def test_current_kv_location_and_staging_are_unchanged(runtime_module, runner):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    locations = torch.tensor([129, 257])
    coordinator.map_last_loc_to_buffer(None, locations, None, None, None)
    assert locations.tolist() == [129, 257]
    request = Request()
    coordinator.admit_request_into_staging(request)
    coordinator.admit_request_direct(request)
    assert not request.hisparse_staging
    assert coordinator.collect_ready_reqs() == []
    assert not coordinator.has_ongoing_staging()
    assert coordinator.ack_staging_queue == []


def test_stats_preserve_existing_pool_observer_accounting(runtime_module, runner):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    assert coordinator.get_token_stats() is None
    runner.token_to_kv_pool_allocator.available_size.assert_not_called()


def test_real_hicache_backing_survives_finish_restore_and_last_cache_release(
    runtime_module, runtime, monkeypatch
):
    fabric_runtime = runtime
    backing, pool = fabric_runtime.make(rope_dim=0)
    host_pool = backing.bind_hicache(pool)
    binding = backing.page_binding
    assert binding.ensure_pages(torch.tensor([1]))
    cached = host_pool.acquire_device_pages(torch.arange(4, 8))
    payload = backing.regions[0][0].tensor
    payload[4:8].fill_(19)
    initial_events = list(fabric_runtime.events)
    request_pool = SimpleNamespace(
        req_generation=torch.tensor([0, 7]), req_generation_epoch=0
    )
    model_runner = SimpleNamespace(
        device="cpu",
        req_to_token_pool=request_pool,
        token_to_kv_pool=pool,
        token_to_kv_pool_allocator=object(),
        prefill_offload_runtime=None,
    )
    context = types.ModuleType("sglang.srt.runtime_context")
    context.get_memory = lambda: SimpleNamespace(enable_hisparse=True)
    context.get_spec = lambda: SimpleNamespace()
    context.get_disagg = lambda: SimpleNamespace(disaggregation_mode="null")
    context.get_platform = lambda: SimpleNamespace(is_npu=True)
    context.get_parallel = lambda: SimpleNamespace(enable_dp_attention=False)
    monkeypatch.setitem(sys.modules, context.__name__, context)
    factory_path = ROOT / "python/sglang/srt/managers/hisparse_factory.py"
    factory_spec = importlib.util.spec_from_file_location(
        "hisparse_factory_integration", factory_path
    )
    factory = importlib.util.module_from_spec(factory_spec)
    factory_spec.loader.exec_module(factory)
    coordinator = factory.create_hisparse_coordinator(model_runner)
    assert isinstance(coordinator, runtime_module.NpuHiSparseCoordinator)
    request = Request()
    coordinator.request_finished(request)
    coordinator.request_finished(request)
    assert binding.references[1] == 2
    assert request_pool.req_generation[1] == 8
    binding.release_device_pages(torch.tensor([1]))
    assert binding.references[1] == 1
    destination = torch.arange(12, 16)
    host_pool.load_to_device_per_layer(pool, cached, destination, 0, "kernel_ascend")
    assert binding.device_to_host_cpu[3] == 1
    assert binding.references[1] == 2
    assert torch.all(payload[4:8] == 19)
    assert fabric_runtime.events == initial_events
    binding.release_device_pages(torch.tensor([3]))
    host_pool.free(cached)
    for event in fabric_runtime.device_events:
        event.complete = True
    binding.reap()
    assert host_pool.available_size() == host_pool.usable_size


def test_destroy_drains_and_closes_physical_owner_once(runtime_module, runner):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    runtime = integrated_runtime()
    coordinator.bind_prefill_runtime(runtime)
    stream = SimpleNamespace(synchronize=Mock())
    coordinator.set_decode_producer_stream(stream)
    coordinator.destroy()
    coordinator.destroy()
    runtime.drain.assert_called_once_with()
    stream.synchronize.assert_called_once_with()
    backing = runner.token_to_kv_pool.decode_host_backing
    backing.hicache_pool.destroy.assert_called_once_with()
    backing.pool.close.assert_called_once_with(drained=True)


def test_prepare_shutdown_preserves_host_owner_until_all_runners_drain(
    runtime_module, runner
):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    coordinator.prepare_shutdown()
    backing = runner.token_to_kv_pool.decode_host_backing
    backing.hicache_pool.destroy.assert_not_called()
    backing.pool.close.assert_not_called()
    coordinator.destroy()
    backing.pool.close.assert_called_once_with(drained=True)


def test_destroy_failed_prefill_preserves_retry_after_failed_drain(
    runtime_module, runner
):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    runtime = SimpleNamespace(
        failed=True,
        drain_failed_batch=Mock(side_effect=[RuntimeError("busy"), None]),
    )
    coordinator.bind_prefill_runtime(runtime)
    with pytest.raises(RuntimeError, match="busy"):
        coordinator.destroy()
    coordinator.destroy()
    coordinator.destroy()
    assert runtime.drain_failed_batch.call_count == 2


if __name__ == "__main__":
    run_pytest(__file__)
