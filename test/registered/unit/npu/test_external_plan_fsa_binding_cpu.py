"""Actual Host binding release/reap with CPU-simulated NPU event readiness."""

from types import SimpleNamespace
import pytest
import torch

from test_hicache_offload_binding import runtime, device, bound, rows
from test_external_plan_fsa_lifecycle_cpu import load_graph_backend
from sglang.srt.hardware_backend.npu.hisparse.fsa_graph_lifecycle import FsaGraphReaders
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_replay_release_waits_for_new_reader_after_capture_event_reaped(
    bound, monkeypatch
):
    pool, binding, device = bound.pool, bound.binding, bound.device
    assert binding.ensure_pages(torch.tensor([1]))
    capture_event = device.Event()
    capture_event.record(device.stream)
    binding.record_reader(capture_event)
    capture_event.complete = True
    binding.reap()
    assert not binding.reader_events
    readers = FsaGraphReaders()
    readers.add(binding, torch.tensor(binding.mapping_epoch))
    backend = object.__new__(load_graph_backend(monkeypatch))
    backend._graphs = {1: SimpleNamespace(replay=lambda: None)}
    backend._outputs = {1: None}
    backend._fsa_readers = {1: readers}
    backend._device_module = device
    backend.replay(1, None)
    assert len(binding.reader_events) == 1
    replay_event = binding.reader_events[0]
    assert replay_event is not capture_event
    binding.release_device_pages(torch.tensor([1]))
    retirement = binding.retired[0][0]
    assert replay_event in retirement.dependencies
    retirement.complete = True
    binding.reap()
    assert pool.slot_used[rows(1)].all()
    replay_event.complete = True
    binding.reap()
    assert pool.available_size() == pool.usable_size


def test_failed_reader_drain_blocks_release_reap_and_new_replay(bound, monkeypatch):
    binding = bound.binding
    assert binding.ensure_pages(torch.tensor([1]))
    old_mapping = binding.device_to_host_cpu.clone()
    old_refs = binding.references.clone()
    readers = FsaGraphReaders()
    readers.add(binding, torch.tensor(binding.mapping_epoch))

    def fail():
        raise RuntimeError("injected stream drain failure")

    bound.device.stream.synchronize = fail
    with pytest.raises(RuntimeError, match="injected stream"):
        readers.drain_failure(bound.device)
    for action in (
        lambda: binding.release_device_pages(torch.tensor([1])),
        binding.reap,
        binding.clear,
        readers.before_replay,
    ):
        with pytest.raises(RuntimeError, match="quarantined"):
            action()
    assert torch.equal(binding.device_to_host_cpu, old_mapping)
    assert torch.equal(binding.references, old_refs)
