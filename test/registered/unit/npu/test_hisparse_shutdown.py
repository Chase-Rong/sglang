"""Service shutdown over real coordinator and Host allocation owners."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import test_hicache_memfabric_pool as fabric_tests
import test_hisparse_coordinator_runtime as coordinator_tests
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

fabric_runtime = fabric_tests.runtime
runner = coordinator_tests.runner
runtime_module = coordinator_tests.runtime_module

SOURCE = (
    Path(__file__).parents[4]
    / "python/sglang/srt/hardware_backend/npu/hisparse/shutdown.py"
)


@pytest.fixture
def shutdown(monkeypatch):
    name = "sglang.srt.hardware_backend.npu.hisparse.shutdown"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def scheduler(coordinator, **changes):
    values = dict(
        hisparse_coordinator=coordinator,
        draft_worker=None,
        tree_cache=SimpleNamespace(release_host_resources=Mock()),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_real_target_draft_backings_close_only_after_both_runtimes_and_hicache(
    shutdown, runtime_module, fabric_runtime
):
    order, coordinators, backings = [], [], []
    for name in ("target", "draft"):
        backing, pool = fabric_runtime.make(rope_dim=0)
        backing.bind_hicache(pool)
        backings.append(backing)
        runtime = SimpleNamespace(
            active=False, drain=lambda name=name: order.append(name)
        )
        model_runner = SimpleNamespace(
            device="cpu",
            token_to_kv_pool=pool,
            req_to_token_pool=SimpleNamespace(req_generation=torch.zeros(2)),
            token_to_kv_pool_allocator=object(),
            prefill_offload_runtime=runtime,
        )
        coordinators.append(runtime_module.NpuHiSparseCoordinator(model_runner))

    def detach_cache():
        assert order == ["target", "draft"]
        assert all(not backing.regions[0][0].closed for backing in backings)
        order.append("cache")
        for backing in backings:
            backing.hicache_pool.destroy()

    service = scheduler(
        coordinators[0],
        draft_worker=SimpleNamespace(
            _draft_model_runners=lambda: (
                SimpleNamespace(hisparse_coordinator=coordinators[1]),
            )
        ),
        tree_cache=SimpleNamespace(release_host_resources=detach_cache),
    )
    shutdown.release_npu_hisparse_resources(service)
    assert order == ["target", "draft", "cache"]
    assert all(backing.regions[0][0].closed for backing in backings)
    assert fabric_runtime.events.count(("uninitialize",)) == 1
    events = list(fabric_runtime.events)
    shutdown.release_npu_hisparse_resources(service)
    assert fabric_runtime.events == events


def test_drain_failure_prevents_cache_and_host_close_and_can_retry(
    shutdown, runtime_module, runner
):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    runtime = SimpleNamespace(
        drain_for_shutdown=Mock(side_effect=[RuntimeError("busy"), None])
    )
    coordinator.bind_prefill_runtime(runtime)
    service = scheduler(coordinator)
    with pytest.raises(RuntimeError, match="busy"):
        shutdown.release_npu_hisparse_resources(service)
    service.tree_cache.release_host_resources.assert_not_called()
    runner.token_to_kv_pool.decode_host_backing.pool.close.assert_not_called()
    shutdown.release_npu_hisparse_resources(service)
    service.tree_cache.release_host_resources.assert_called_once_with()
    runner.token_to_kv_pool.decode_host_backing.pool.close.assert_called_once_with(
        drained=True
    )


def test_peer_drain_failure_keeps_local_backing(shutdown, runtime_module, runner):
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    group = SimpleNamespace(all_gather_object=Mock(return_value=[None, "peer busy"]))
    runner.token_to_kv_pool.decode_host_backing.pool.shared_group = SimpleNamespace(
        group=group
    )
    service = scheduler(coordinator)
    with pytest.raises(RuntimeError, match="all TP ranks"):
        shutdown.release_npu_hisparse_resources(service)
    runner.token_to_kv_pool.decode_host_backing.pool.close.assert_not_called()
    service.tree_cache.release_host_resources.assert_not_called()


def test_prefill_owned_registration_closes_once_after_drain(
    shutdown, runtime_module, runner
):
    runner.token_to_kv_pool.decode_host_backing = None
    events = []
    host = SimpleNamespace(
        close=Mock(side_effect=lambda **kwargs: events.append("close"))
    )
    runtime = SimpleNamespace(
        adapter=SimpleNamespace(host_pool=host),
        drain_for_shutdown=Mock(side_effect=lambda: events.append("drain")),
    )
    coordinator = runtime_module.NpuHiSparseCoordinator(runner)
    coordinator.bind_prefill_runtime(runtime)
    service = scheduler(
        coordinator,
        tree_cache=SimpleNamespace(
            release_host_resources=lambda: events.append("cache")
        ),
    )
    shutdown.release_npu_hisparse_resources(service)
    shutdown.release_npu_hisparse_resources(service)
    assert events == ["drain", "cache", "close"]
    host.close.assert_called_once_with(drained=True)


@pytest.fixture
def receive_manager(monkeypatch):
    modules = {
        "sglang.srt.disaggregation.base.conn": dict(
            KVPoll=SimpleNamespace(Success=3, Failed=4)
        ),
        "sglang.srt.hardware_backend.npu.hisparse.layer_protocol": dict(
            encode=lambda value: json.dumps(value).encode()
        ),
        "sglang.srt.utils.network": dict(
            NetworkAddress=lambda address, port: SimpleNamespace(
                to_tcp=lambda: f"{address}:{port}", is_ipv6=False
            )
        ),
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    lease = SimpleNamespace(
        session_id="session",
        lease_id="lease",
        failed=False,
        drained=False,
        abort_generation=0,
        ready=lambda: False,
        writer_endpoints=lambda: {("a", 1), ("b", 2)},
        close=Mock(),
    )
    manager = SimpleNamespace(
        _layer_leases={7: lease},
        enable_deferred_decode_kv_release=False,
        local_ip="local",
        rank_port=3,
        _layer_staging=SimpleNamespace(forget=Mock(), drain=Mock()),
        update_status=Mock(),
        check_status=lambda room: 1,
        _send_multipart_locked=Mock(),
    )
    return manager, lease


def test_receive_close_waits_for_all_writer_acknowledgements(shutdown, receive_manager):
    manager, lease = receive_manager
    sent = []

    def send(address, frames, **kwargs):
        assert not lease.close.called
        assert frames[0] == b"FSA_LAYER_ABORT"
        assert json.loads(frames[1])["lease"] == lease.lease_id
        assert json.loads(frames[1])["abort_generation"] == lease.abort_generation == 1
        sent.append(address)
        if len(sent) == 2:
            lease.drained = True

    manager._send_multipart_locked.side_effect = send
    shutdown.drain_receive_leases(manager, 0.1)
    assert len(sent) == 2
    assert not manager._layer_leases
    lease.close.assert_called_once_with()
    manager._layer_staging.forget.assert_called_once_with(lease)
    manager._layer_staging.drain.assert_called_once_with()


def test_receive_timeout_keeps_host_pin_and_staging_credit(shutdown, receive_manager):
    manager, lease = receive_manager
    with pytest.raises(TimeoutError, match="writers have not drained"):
        shutdown.drain_receive_leases(manager, 0)
    assert manager._layer_leases[7] is lease
    lease.close.assert_not_called()
    manager._layer_staging.forget.assert_not_called()
    manager._layer_staging.drain.assert_not_called()


def test_shutdown_abort_reuses_registered_generation_after_timeout(
    shutdown, receive_manager
):
    manager, lease = receive_manager
    manager.enable_deferred_decode_kv_release = True
    manager.register_deferred_abort_room = Mock(return_value=9)
    for _ in range(2):
        with pytest.raises(TimeoutError):
            shutdown.drain_receive_leases(manager, 0)
    manager.register_deferred_abort_room.assert_called_once_with(7)
    assert lease.abort_generation == 9
    for call in manager._send_multipart_locked.call_args_list:
        assert json.loads(call.args[1][1])["abort_generation"] == 9
    lease.close.assert_not_called()


def test_ready_kv_still_aborts_unfinished_state_transfer(shutdown, receive_manager):
    manager, lease = receive_manager
    lease.ready = lambda: True
    with pytest.raises(TimeoutError):
        shutdown.drain_receive_leases(manager, 0)
    assert manager._send_multipart_locked.call_count == 2
    lease.close.assert_not_called()


if __name__ == "__main__":
    run_pytest(__file__)
