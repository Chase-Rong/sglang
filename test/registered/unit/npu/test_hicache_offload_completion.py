"""CPU contract gates using extracted production APIs and mock streams/pools.

These tests do not initialize NPU resources or qualify full Unified HiCache E2E.
Sidecar payloads use real CPU tensor copies behind simulated stream dependencies.
"""

import ast
import logging
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache
from inspect import getattr_static
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, NamedTuple, Optional
from unittest.mock import Mock

import pytest
import torch

SRT = Path(__file__).parents[4] / "python/sglang/srt"


def extract(relative_path, namespace, definitions):
    path = SRT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if getattr(node, "name", None) not in definitions:
            continue
        methods = definitions[node.name]
        if methods is not None:
            node.body = [
                method
                for method in node.body
                if getattr(method, "name", None) in methods
            ]
        selected.append(node)
    assert len(selected) == len(definitions)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.Module(body=[future, *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class Gate:
    def __init__(self):
        self.open = False

    def query(self):
        return self.open


class Stream:
    def __init__(self, device):
        self.device = device
        self.work = []
        self.completed = 0

    def enqueue(self, action=None, dependency=None):
        self.work.append((action, dependency))

    def wait_event(self, event):
        self.enqueue(dependency=event)

    def wait_stream(self, stream):
        event = self.device.Event()
        event.record(stream)
        self.wait_event(event)

    def step(self):
        if self.completed == len(self.work):
            return False
        action, dependency = self.work[self.completed]
        if dependency is not None and not dependency.query():
            return False
        if action is not None:
            action()
        self.completed += 1
        return True

    def synchronize(self):
        self.device.synchronized.append(self)
        self.device.progress()
        if self.completed != len(self.work):
            raise RuntimeError("mock stream still blocked")


class Event:
    def __init__(self, device, enable_timing=False):
        self.device = device
        self.enable_timing = enable_timing
        self.stream = None
        self.position = 0

    def record(self, stream=None):
        self.stream = stream or self.device.current_stream()
        self.stream.enqueue()
        self.position = len(self.stream.work)

    def wait(self, stream):
        stream.wait_event(self)

    def query(self):
        return self.stream is None or self.stream.completed >= self.position

    def synchronize(self):
        if self.stream is not None:
            self.stream.synchronize()


class Device:
    __name__ = "mock_device"

    def __init__(self, timing=True):
        self.timing = timing
        self.streams = []
        self.synchronized = []
        self.current = self.Stream()

    def Stream(self):
        stream = Stream(self)
        self.streams.append(stream)
        return stream

    def Event(self, enable_timing=False):
        if enable_timing and not self.timing:
            raise NotImplementedError("timing unavailable")
        return Event(self, enable_timing)

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous

    def progress(self):
        while True:
            progressed = [stream.step() for stream in self.streams]
            if not any(progressed):
                return


@pytest.fixture
def api():
    device = Device()
    namespace = dict(
        globals(),
        device_module=device,
        logger=logging.getLogger(__name__),
        dataclass=dataclass,
        replace=replace,
        Enum=Enum,
        cache=cache,
        getattr_static=getattr_static,
        Any=Any,
        Callable=Callable,
        List=List,
        NamedTuple=NamedTuple,
        Optional=Optional,
    )
    extract(
        "mem_cache/hicache_storage.py",
        namespace,
        dict.fromkeys(("PoolName", "PoolHitPolicy", "PoolTransfer")),
    )
    extract(
        "mem_cache/pool_host/group.py",
        namespace,
        dict.fromkeys(("PoolEntry", "HostPoolGroup")),
    )
    extract(
        "mem_cache/l2_transfer.py",
        namespace,
        dict.fromkeys(
            (
                "_timing_events_supported",
                "make_timing_event_pair",
                "L2Transfer",
                "TransferCompletion",
                "L2TransferEngine",
            )
        ),
    )
    extract(
        "managers/cache_controller.py",
        namespace,
        {
            "CacheOperation": None,
            "HiCacheAck": None,
            "LayerLoadingEvent": None,
            "LayerDoneCounter": None,
            "HiCacheController": {
                "start_writing",
                "start_loading",
                "move_indices",
            },
        },
    )
    namespace["BaseHiCacheController"] = namespace["HiCacheController"]
    extract(
        "mem_cache/hybrid_cache/hybrid_cache_controller.py",
        namespace,
        {
            "HybridCacheController": {
                "write",
                "_rollback_write",
                "load",
                "_rollback_load",
                "_resolve_device_transfers",
                "_move_write_operation",
                "_move_op_indices",
                "move_hybrid_indices",
                "_l2_transfers",
                "_l2_load_transfers",
                "_num_tokens_by_pool",
                "_transfer_num_bytes",
            }
        },
    )
    return SimpleNamespace(**namespace)


class CopyPool:
    layout = "layer_first"
    page_size = 2
    device = "cpu"
    size = 16
    logical_size = 16
    size_per_token = 8
    can_use_write_back_jit = False

    def __init__(self, device, layers=2):
        self.device_module = device
        self.layer_num = layers
        self.data = torch.full((layers, 16), -1.0)
        self.size_per_token = layers * self.data.element_size()
        self.indices = torch.tensor([0, 1])
        self.alloc = Mock(side_effect=lambda count: self.indices.clone())
        self.free = Mock(return_value=2)
        self.backups = 0
        self.loads = []
        self.copy_ready = None
        self.fail_backup = False

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        self.backups += 1
        if self.fail_backup:
            raise RuntimeError("sidecar backup failed")
        stream = self.device_module.current_stream()
        if self.copy_ready is not None:
            stream.wait_event(self.copy_ready)

        def copy():
            self.data[:, host_indices] = device_pool.data[:, device_indices]

        stream.enqueue(copy)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft,
    ):
        self.loads.append((layer_id, is_draft))

        def copy():
            device_pool.data[layer_id, device_indices] = self.data[
                layer_id, host_indices
            ]

        self.device_module.current_stream().enqueue(copy)


class AdoptedPool(CopyPool):
    def __init__(self, device):
        super().__init__(device)
        self.indices = torch.tensor([6, 7])
        self.references = 1
        self.acquire_device_pages = Mock(side_effect=self.acquire)
        self.free = Mock(side_effect=self.release)
        self.binding = SimpleNamespace(record_reader=Mock())
        self.runtime_ready = Gate()
        self.ready_event = device.Event()
        self.runtime_stream = device.Stream()
        self.runtime_stream.wait_event(self.runtime_ready)
        self.runtime_stream.enqueue(
            lambda: self.data.index_fill_(1, self.indices, 123.0)
        )
        self.ready_event.record(self.runtime_stream)

    def acquire(self, device_indices):
        self.references += 1
        return self.indices.clone()

    def release(self, host_indices):
        assert torch.equal(host_indices, self.indices)
        self.references -= 1
        return len(host_indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        self.device_module.current_stream().wait_event(self.ready_event)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft,
    ):
        self.loads.append((layer_id, is_draft))
        device_pool.mapping[layer_id] = (host_indices.clone(), device_indices.clone())


def controller_with_pools(api, *, adopted=True):
    device = api.device_module
    anchor = AdoptedPool(device) if adopted else CopyPool(device)
    kda = CopyPool(device, layers=1)
    indexer = CopyPool(device)
    names = api.PoolName
    entries = [
        api.PoolEntry(
            name=name,
            host_pool=pool,
            device_pool=SimpleNamespace(
                data=torch.arange(pool.layer_num * 16, dtype=torch.float32).reshape(
                    pool.layer_num, 16
                ),
                mapping={},
            ),
            layer_mapper=mapper,
            is_primary_index_anchor=name == names.KV,
        )
        for name, pool, mapper in (
            (names.KV, anchor, {0: 0, 2: 1}.get),
            (names.MAMBA, kda, {1: 0}.get),
            (names.INDEXER, indexer, {0: 0, 2: 1}.get),
        )
    ]
    controller = api.HybridCacheController.__new__(api.HybridCacheController)
    controller.mem_pool_host = api.HostPoolGroup(entries)
    controller.io_backend = "kernel_ascend"
    controller.device = "cpu"
    controller.layer_num = 3
    controller.layer_done_counter = api.LayerDoneCounter(3)
    controller.load_fence_stream = None
    controller.l2_transfer_engine = api.L2TransferEngine(controller.io_backend)
    controller.write_queue = []
    controller.ack_write_queue = []
    controller.load_queue = []
    controller.ack_load_queue = []
    transfers = [
        api.PoolTransfer(name=names.MAMBA, device_indices=torch.tensor([4, 5])),
        api.PoolTransfer(name=names.INDEXER, indices_from_pool=names.KV),
    ]
    return controller, anchor, kda, indexer, transfers


@pytest.mark.parametrize("timing", [True, False])
def test_adoption_ack_joins_runtime_and_real_sidecar_copies(api, timing):
    api.device_module.timing = timing
    controller, anchor, kda, indexer, transfers = controller_with_pools(api)
    device_indices = torch.tensor([2, 3])
    backing_pointer = anchor.data.data_ptr()
    ready_position = anchor.ready_event.position
    sidecar_ready = Gate()
    indexer.copy_ready = sidecar_ready

    host_indices = controller.write(device_indices, node_id=41, extra_pools=transfers)

    anchor.acquire_device_pages.assert_called_once_with(device_indices)
    anchor.alloc.assert_not_called()
    indexer.alloc.assert_not_called()
    kda.alloc.assert_called_once_with(2)
    assert anchor.references == 2
    assert torch.equal(host_indices, anchor.indices)
    assert controller.write_queue == []
    ack = controller.ack_write_queue[0]
    anchor.binding.record_reader.assert_called_once_with(ack.finish_event)
    assert ack.node_ids == [41]
    assert ack.timing_enabled is timing
    assert ack.num_tokens_by_pool == {"kv": 2, "mamba": 2}
    assert ack.num_bytes == 24
    assert (
        ack.finish_event.stream is controller.l2_transfer_engine.device_to_host_stream
    )
    api.device_module.progress()
    assert not ack.finish_event.query()
    assert torch.all(kda.data == -1)

    anchor.runtime_ready.open = True
    api.device_module.progress()
    assert anchor.ready_event.query()
    assert anchor.ready_event.position == ready_position
    assert torch.all(anchor.data[:, host_indices] == 123)
    assert not ack.finish_event.query()
    assert torch.all(indexer.data == -1)

    sidecar_ready.open = True
    api.device_module.progress()
    assert ack.finish_event.query()
    assert anchor.data.data_ptr() == backing_pointer
    assert anchor.backups == 0
    assert api.device_module.synchronized == []
    for transfer in transfers:
        entry = controller.mem_pool_host.entry_map[transfer.name]
        assert entry.host_pool.backups == 1
        torch.testing.assert_close(
            entry.host_pool.data[:, transfer.host_indices],
            entry.device_pool.data[:, transfer.device_indices],
        )


def test_ordinary_pools_keep_allocation_and_payload_copy(api):
    controller, anchor, kda, indexer, transfers = controller_with_pools(
        api, adopted=False
    )
    kda.acquire_device_pages = Mock()
    host_indices = controller.write(torch.tensor([2, 3]), extra_pools=transfers)
    anchor.alloc.assert_called_once_with(2)
    kda.acquire_device_pages.assert_not_called()
    api.device_module.progress()
    assert controller.ack_write_queue[0].finish_event.query()
    assert controller.ack_write_queue[0].num_bytes == 40
    assert anchor.backups == kda.backups == indexer.backups == 1
    torch.testing.assert_close(
        anchor.data[:, host_indices],
        controller.mem_pool_host.anchor_entry.device_pool.data[:, [2, 3]],
    )


def test_acquire_failure_does_not_allocate_or_publish_ack(api):
    controller, anchor, kda, _, transfers = controller_with_pools(api)
    anchor.acquire_device_pages.side_effect = lambda indices: None
    assert controller.write(torch.tensor([2, 3]), extra_pools=transfers) is None
    anchor.alloc.assert_not_called()
    kda.alloc.assert_not_called()
    assert anchor.references == 1
    assert controller.write_queue == controller.ack_write_queue == []


@pytest.mark.parametrize("failure", ["allocation", "derived_resolution"])
def test_resolution_failure_rolls_back_acquired_reference_and_sidecars(api, failure):
    controller, anchor, kda, _, transfers = controller_with_pools(api)
    if failure == "allocation":
        kda.alloc.side_effect = lambda count: None
    else:
        transfers.append(
            api.PoolTransfer(
                name=api.PoolName.DRAFT_INDEXER,
                indices_from_pool=api.PoolName.DRAFT,
            )
        )
    assert controller.write(torch.tensor([2, 3]), extra_pools=transfers) is None
    assert anchor.references == 1
    anchor.free.assert_called_once()
    if failure == "derived_resolution":
        kda.free.assert_called_once()
    assert transfers[0].host_indices is None
    assert controller.ack_write_queue == []


@pytest.mark.parametrize("adopted", [True, False])
def test_write_capability_lookup_also_accepts_ungrouped_pool(api, adopted):
    controller, anchor, _, _, _ = controller_with_pools(api, adopted=adopted)
    anchor.resolve_host_transfers = Mock(return_value=None)
    controller.mem_pool_host = anchor
    controller.start_writing = Mock()
    host_indices = controller.write(torch.tensor([2, 3]), node_id=7)
    assert torch.equal(host_indices, anchor.indices)
    if adopted:
        anchor.alloc.assert_not_called()
    else:
        anchor.alloc.assert_called_once_with(2)
    controller.start_writing.assert_called_once()
    assert controller.write_queue[0].node_ids == [7]


def test_load_binds_mapping_inline_and_retains_layer_and_ack_fences(api):
    controller, anchor, kda, indexer, transfers = controller_with_pools(api)
    host_indices = controller.write(torch.tensor([2, 3]), extra_pools=transfers)
    anchor.runtime_ready.open = True
    api.device_module.progress()
    for entry in controller.mem_pool_host.entries:
        entry.device_pool.data.fill_(-99)
    forward_ready = Gate()
    controller.load_fence_stream = api.device_module.Stream()
    controller.load_fence_stream.wait_event(forward_ready)
    transfers = [
        replace(transfer, device_indices=torch.tensor(indices))
        for transfer, indices in zip(transfers, ([10, 11], [8, 9]))
    ]
    controller.load_queue.append(
        api.CacheOperation(
            host_indices, torch.tensor([8, 9]), 53, pool_transfers=transfers
        )
    )

    producer_id = controller.start_loading()

    device_anchor = controller.mem_pool_host.anchor_entry.device_pool
    assert set(device_anchor.mapping) == {0, 1}
    for mapped_host, mapped_device in device_anchor.mapping.values():
        assert torch.equal(mapped_host, host_indices)
        assert mapped_device.tolist() == [8, 9]
    assert anchor.loads == [(0, False), (1, False)]
    assert kda.loads == [(0, False)]
    assert indexer.loads == [(0, False), (1, False)]
    producer = controller.layer_done_counter.events[producer_id]
    ack = controller.ack_load_queue[0]
    assert anchor.binding.record_reader.call_args.args[0] is ack.finish_event
    assert ack.node_ids == [53]
    assert ack.num_bytes == 24
    assert (
        ack.finish_event.stream is controller.l2_transfer_engine.host_to_device_stream
    )
    api.device_module.progress()
    assert not ack.finish_event.query()
    assert all(not event.query() for event in producer.load_events)
    assert torch.all(device_anchor.data == -99)

    forward_ready.open = True
    while controller.load_fence_stream.step():
        pass
    load_stream = controller.l2_transfer_engine.host_to_device_stream
    while not producer.load_events[0].query():
        assert load_stream.step()
    indexer_entry = controller.mem_pool_host.entry_map[api.PoolName.INDEXER]
    torch.testing.assert_close(
        indexer_entry.device_pool.data[0, [8, 9]], indexer.data[0, host_indices]
    )
    assert torch.all(indexer_entry.device_pool.data[1] == -99)
    kda_entry = controller.mem_pool_host.entry_map[api.PoolName.MAMBA]
    assert torch.all(kda_entry.device_pool.data == -99)
    assert not producer.load_events[1].query()
    assert not ack.finish_event.query()
    api.device_module.progress()
    assert ack.finish_event.query()
    assert all(event.query() for event in producer.load_events)
    assert torch.all(device_anchor.data == -99)
    for transfer in transfers:
        entry = controller.mem_pool_host.entry_map[transfer.name]
        torch.testing.assert_close(
            entry.device_pool.data[:, transfer.device_indices],
            entry.host_pool.data[:, transfer.host_indices],
        )


def test_sidecar_submission_failure_drains_before_rollback_without_ack(api):
    controller, anchor, kda, indexer, transfers = controller_with_pools(api)
    anchor.runtime_ready.open = True
    indexer.fail_backup = True
    original_release = anchor.free.side_effect

    def release(host_indices):
        assert api.device_module.synchronized == [
            api.device_module.current_stream(),
            controller.l2_transfer_engine.device_to_host_stream,
        ]
        torch.testing.assert_close(kda.data[:, [0, 1]], torch.tensor([[4.0, 5.0]]))
        return original_release(host_indices)

    anchor.free.side_effect = release
    with pytest.raises(RuntimeError, match="sidecar backup failed"):
        controller.write(torch.tensor([2, 3]), extra_pools=transfers)
    assert controller.write_queue == []
    assert controller.ack_write_queue == []
    assert anchor.references == 1
    anchor.free.assert_called_once()
    kda.free.assert_called_once()
    indexer.free.assert_not_called()
    assert transfers[0].host_indices is None
    assert transfers[1].host_indices is None
    assert transfers[1].device_indices is None


@pytest.mark.parametrize(
    "device_type", ["cuda", "npu", "privateuseone", "privateuse1", "cpu"]
)
def test_transfer_engine_keeps_index_tensor_stream_lifetimes(api, device_type):
    device = api.device_module
    engine = api.L2TransferEngine("kernel_ascend")
    host_indices = SimpleNamespace(
        device=SimpleNamespace(type=device_type), record_stream=Mock()
    )
    device_indices = SimpleNamespace(
        device=SimpleNamespace(type=device_type), record_stream=Mock()
    )
    transfer = api.L2Transfer(
        host_pool=Mock(layer_num=1),
        device_pool=object(),
        host_indices=host_indices,
        device_indices=device_indices,
    )
    engine.submit_device_to_host([transfer])
    engine.submit_host_to_device([transfer], layer_num=1)
    for indices in (host_indices, device_indices):
        expected_streams = (
            [engine.device_to_host_stream, engine.host_to_device_stream]
            if device_type != "cpu"
            else []
        )
        assert [
            call.args[0] for call in indices.record_stream.call_args_list
        ] == expected_streams
    assert device.current_stream() is device.streams[0]


def test_resolve_exception_releases_only_new_host_references(api):
    controller, anchor, kda, indexer, transfers = controller_with_pools(api)
    transfers[1].indices_from_pool = None
    transfers[1].device_indices = torch.tensor([2, 3])
    indexer.alloc.side_effect = RuntimeError("sidecar allocation failed")
    previous = api.CacheOperation(torch.tensor([10, 11]), torch.tensor([10, 11]), 91)
    controller.write_queue.append(previous)
    with pytest.raises(RuntimeError, match="sidecar allocation failed"):
        controller.write(torch.tensor([2, 3]), extra_pools=transfers)
    assert anchor.references == 1
    anchor.free.assert_called_once()
    kda.free.assert_called_once()
    indexer.free.assert_not_called()
    assert transfers[0].host_indices is None
    assert controller.write_queue == [previous]
    assert controller.ack_write_queue == []
    assert api.device_module.synchronized == []


def test_failed_merged_write_preserves_older_queued_ops_and_acks(api):
    controller, anchor, kda, indexer, transfers = controller_with_pools(api)
    anchor.runtime_ready.open = True
    anchor.references += 1
    older_transfer = api.PoolTransfer(
        name=api.PoolName.MAMBA,
        host_indices=torch.tensor([12, 13]),
        device_indices=torch.tensor([12, 13]),
    )
    previous = api.CacheOperation(
        torch.tensor([10, 11]),
        torch.tensor([10, 11]),
        91,
        pool_transfers=[older_transfer],
    )
    existing_ack = object()
    controller.write_queue.append(previous)
    controller.ack_write_queue.append(existing_ack)
    indexer.fail_backup = True
    with pytest.raises(RuntimeError, match="sidecar backup failed"):
        controller.write(torch.tensor([2, 3]), extra_pools=transfers)
    assert controller.write_queue == [previous]
    assert controller.ack_write_queue == [existing_ack]
    assert anchor.references == 2
    anchor.free.assert_called_once()
    kda.free.assert_called_once()
    assert kda.free.call_args.args[0].tolist() == [0, 1]
    assert older_transfer.host_indices.tolist() == [12, 13]
    controller.start_writing()
    api.device_module.progress()
    assert controller.ack_write_queue[-1].node_ids == [91]
    assert controller.ack_write_queue[-1].finish_event.query()


def test_submission_failure_does_not_free_preallocated_sidecar(api):
    controller, anchor, kda, indexer, transfers = controller_with_pools(api)
    anchor.runtime_ready.open = True
    host_indices = torch.tensor([12, 13])
    transfers[0].host_indices = host_indices
    indexer.fail_backup = True
    with pytest.raises(RuntimeError, match="sidecar backup failed"):
        controller.write(torch.tensor([2, 3]), extra_pools=transfers)
    kda.free.assert_not_called()
    assert transfers[0].host_indices is host_indices
    assert anchor.references == 1


@pytest.mark.parametrize("failed_stream", ["producer", "backup"])
def test_drain_failure_retains_pages_and_operations_for_domain_recovery(
    api, failed_stream
):
    controller, anchor, kda, indexer, transfers = controller_with_pools(api)
    anchor.runtime_ready.open = True
    indexer.fail_backup = True
    stream = (
        api.device_module.current_stream()
        if failed_stream == "producer"
        else controller.l2_transfer_engine.device_to_host_stream
    )
    stream.synchronize = Mock(side_effect=RuntimeError("drain failed"))
    with pytest.raises(RuntimeError, match="drain failed"):
        controller.write(torch.tensor([2, 3]), node_id=71, extra_pools=transfers)
    assert controller.write_queue[0].node_ids == [71]
    assert controller.ack_write_queue == []
    assert anchor.references == 2
    anchor.free.assert_not_called()
    kda.free.assert_not_called()
    other_stream = (
        controller.l2_transfer_engine.device_to_host_stream
        if failed_stream == "producer"
        else api.device_module.current_stream()
    )
    assert other_stream in api.device_module.synchronized


def test_index_preparation_failure_drains_producer_before_release(api):
    controller, anchor, _, _, _ = controller_with_pools(api)
    prepared = []

    def fail_preparation(operation):
        api.device_module.current_stream().enqueue(lambda: prepared.append(True))
        raise RuntimeError("index preparation failed")

    controller._move_write_operation = fail_preparation
    original_release = anchor.free.side_effect

    def release(indices):
        assert prepared == [True]
        return original_release(indices)

    anchor.free.side_effect = release
    with pytest.raises(RuntimeError, match="index preparation failed"):
        controller.write(torch.tensor([2, 3]))
    assert controller.write_queue == controller.ack_write_queue == []
    assert anchor.references == 1


def restore_allocator(controller, anchor):
    device_pool = controller.mem_pool_host.anchor_entry.device_pool

    def restore(host_indices):
        device_indices = torch.tensor([8, 9])
        device_pool.mapping[0] = (host_indices.clone(), device_indices)
        anchor.references += 1
        return device_indices

    def free(device_indices):
        assert device_indices.tolist() == [8, 9]
        anchor.references -= 1
        device_pool.mapping.clear()

    return SimpleNamespace(
        alloc=Mock(side_effect=AssertionError("unexpected Host allocation")),
        alloc_for_host_restore=Mock(side_effect=restore),
        free=Mock(side_effect=free),
    )


@pytest.mark.parametrize("hybrid_allocator", [True, False])
def test_load_adopts_existing_host_pages_without_allocating_host_backing(
    api, hybrid_allocator
):
    controller, anchor, _, _, _ = controller_with_pools(api)
    allocator = restore_allocator(controller, anchor)
    controller.mem_pool_device_allocator = (
        SimpleNamespace(full_attn_allocator=allocator)
        if hybrid_allocator
        else allocator
    )
    device_indices = controller.load(anchor.indices, node_id=81)
    allocator.alloc_for_host_restore.assert_called_once_with(anchor.indices)
    allocator.alloc.assert_not_called()
    anchor.alloc.assert_not_called()
    assert device_indices.tolist() == [8, 9]
    assert anchor.references == 2
    assert controller.load_queue[0].node_ids == [81]
    mapping = controller.mem_pool_host.anchor_entry.device_pool.mapping
    assert mapping[0][1].tolist() == [8, 9]


def test_load_adoption_rejects_incompatible_allocator(api):
    controller, anchor, _, _, _ = controller_with_pools(api)
    allocator = SimpleNamespace(alloc=Mock())
    controller.mem_pool_device_allocator = allocator
    with pytest.raises(RuntimeError, match="Incompatible.*alloc_for_host_restore"):
        controller.load(anchor.indices)
    allocator.alloc.assert_not_called()
    assert controller.load_queue == []


def test_load_adoption_allocation_failure_does_not_queue_or_fallback(api):
    controller, anchor, _, _, _ = controller_with_pools(api)
    allocator = SimpleNamespace(
        alloc=Mock(), alloc_for_host_restore=Mock(return_value=None)
    )
    controller.mem_pool_device_allocator = allocator
    assert controller.load(anchor.indices) is None
    allocator.alloc.assert_not_called()
    assert controller.load_queue == []


def test_load_sidecar_failure_releases_restored_mapping_with_device_free(api):
    controller, anchor, _, _, _ = controller_with_pools(api)
    allocator = restore_allocator(controller, anchor)
    controller.mem_pool_device_allocator = allocator
    kda_entry = controller.mem_pool_host.entry_map[api.PoolName.MAMBA]
    kda_entry.device_alloc_fn = Mock(return_value=None)
    kda_entry.device_free_fn = Mock()
    transfer = api.PoolTransfer(
        name=api.PoolName.MAMBA, host_indices=torch.tensor([0, 1])
    )
    assert controller.load(anchor.indices, extra_pools=[transfer]) is None
    allocator.free.assert_called_once()
    assert anchor.references == 1
    assert controller.mem_pool_host.anchor_entry.device_pool.mapping == {}
    assert controller.load_queue == []


def test_ordinary_load_keeps_original_device_allocator(api):
    controller, anchor, _, _, _ = controller_with_pools(api, adopted=False)
    allocator = SimpleNamespace(
        alloc=Mock(return_value=torch.tensor([8, 9])), alloc_for_host_restore=Mock()
    )
    controller.mem_pool_device_allocator = allocator
    assert controller.load(anchor.indices).tolist() == [8, 9]
    allocator.alloc.assert_called_once_with(2)
    allocator.alloc_for_host_restore.assert_not_called()


@pytest.mark.parametrize("direction", ["backup", "load"])
def test_adopted_pool_requires_reader_lifetime_registration(api, direction):
    controller, anchor, _, _, _ = controller_with_pools(api)
    anchor.binding = SimpleNamespace()
    engine = controller.l2_transfer_engine
    transfer = api.L2Transfer(
        host_pool=anchor,
        device_pool=controller.mem_pool_host.anchor_entry.device_pool,
        host_indices=anchor.indices,
        device_indices=torch.tensor([2, 3]),
    )
    with pytest.raises(RuntimeError, match="binding.record_reader"):
        if direction == "backup":
            engine.submit_device_to_host([transfer])
        else:
            engine.submit_host_to_device([transfer], layer_num=2)
    assert engine.device_to_host_stream.work == []
    assert engine.host_to_device_stream.work == []


@pytest.mark.parametrize("failure", ["allocation", "eviction"])
@pytest.mark.parametrize("preallocated", [True, False])
def test_load_resolve_exception_releases_only_new_allocations(
    api, failure, preallocated
):
    controller, anchor, _, _, _ = controller_with_pools(api)
    allocator = restore_allocator(controller, anchor)
    controller.mem_pool_device_allocator = allocator
    kda_entry = controller.mem_pool_host.entry_map[api.PoolName.MAMBA]
    kda_indices = torch.tensor([10, 11])
    kda_entry.device_alloc_fn = Mock(return_value=kda_indices)
    kda_entry.device_free_fn = Mock()
    indexer_entry = controller.mem_pool_host.entry_map[api.PoolName.INDEXER]
    indexer_entry.device_free_fn = Mock()
    if failure == "allocation":
        indexer_entry.device_alloc_fn = Mock(
            side_effect=RuntimeError("sidecar allocation failed")
        )
    else:
        indexer_entry.device_alloc_fn = Mock(return_value=None)
        indexer_entry.device_evict_fn = Mock(
            side_effect=RuntimeError("sidecar eviction failed")
        )
    sidecar = api.PoolTransfer(
        name=api.PoolName.MAMBA,
        host_indices=torch.tensor([0, 1]),
        device_indices=kda_indices if preallocated else None,
    )
    failing = api.PoolTransfer(
        name=api.PoolName.INDEXER, host_indices=torch.tensor([6, 7])
    )
    previous = api.CacheOperation(torch.tensor([12, 13]), torch.tensor([12, 13]), 91)
    controller.load_queue.append(previous)

    with pytest.raises(RuntimeError, match=f"sidecar {failure} failed"):
        controller.load(anchor.indices, extra_pools=[sidecar, failing])

    allocator.free.assert_called_once()
    assert anchor.references == 1
    assert controller.mem_pool_host.anchor_entry.device_pool.mapping == {}
    assert controller.load_queue == [previous]
    assert controller.ack_load_queue == []
    anchor.free.assert_not_called()
    assert failing.device_indices is None
    indexer_entry.device_free_fn.assert_not_called()
    if preallocated:
        kda_entry.device_alloc_fn.assert_not_called()
        kda_entry.device_free_fn.assert_not_called()
        assert sidecar.device_indices is kda_indices
    else:
        kda_entry.device_free_fn.assert_called_once_with(kda_indices)
        assert sidecar.device_indices is None


def test_failed_load_restores_derived_descriptors_without_double_free(api):
    controller, anchor, _, _, _ = controller_with_pools(api)
    allocator = restore_allocator(controller, anchor)
    controller.mem_pool_device_allocator = allocator
    kda_entry = controller.mem_pool_host.entry_map[api.PoolName.MAMBA]
    kda_indices = torch.tensor([10, 11])
    kda_entry.device_alloc_fn = Mock(return_value=kda_indices)
    kda_entry.device_free_fn = Mock()
    sidecar = api.PoolTransfer(
        name=api.PoolName.MAMBA, host_indices=torch.tensor([0, 1])
    )
    original_host = torch.tensor([12, 13])
    original_device = torch.tensor([12, 13])
    derived = api.PoolTransfer(
        name=api.PoolName.INDEXER,
        indices_from_pool=api.PoolName.KV,
        host_indices=original_host,
        device_indices=original_device,
    )
    missing_source = api.PoolTransfer(
        name=api.PoolName.DRAFT_INDEXER, indices_from_pool=api.PoolName.DRAFT
    )

    assert (
        controller.load(anchor.indices, extra_pools=[sidecar, derived, missing_source])
        is None
    )

    allocator.free.assert_called_once()
    kda_entry.device_free_fn.assert_called_once_with(kda_indices)
    anchor.free.assert_not_called()
    assert anchor.references == 1
    assert sidecar.device_indices is None
    assert derived.host_indices is original_host
    assert derived.device_indices is original_device
    assert controller.load_queue == controller.ack_load_queue == []


@pytest.mark.parametrize("is_draft", [True, False])
def test_prepared_load_runs_once_before_start_on_caller_stream(api, is_draft):
    device = api.device_module
    engine = api.L2TransferEngine("kernel_ascend")
    caller_stream = device.current_stream()
    pool = CopyPool(device)
    pool.data[:, :2] = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    target = SimpleNamespace(data=torch.full((2, 16), -1.0))
    host_indices = torch.tensor([0, 1])
    device_indices = torch.tensor([2, 3])
    prepared = []
    executed = []

    def prepare(device_pool, host_rows, device_rows, io_backend, *, is_draft):
        assert device.current_stream() is caller_stream
        assert io_backend == "kernel_ascend"
        source_rows, target_rows = host_rows.clone(), device_rows.clone()
        prepared.append(is_draft)

        def load_layer(layer_id):
            assert device.current_stream() is engine.host_to_device_stream
            executed.append(layer_id)

            def copy():
                device_pool.data[layer_id, target_rows] = pool.data[
                    layer_id, source_rows
                ]

            device.current_stream().enqueue(copy)

        return load_layer

    pool.prepare_load_to_device = prepare
    pool.load_to_device_per_layer = Mock(
        side_effect=AssertionError("unexpected fallback")
    )
    original_start = engine._start_event

    def start(event):
        assert prepared == [is_draft]
        assert device.current_stream() is caller_stream
        return original_start(event)

    engine._start_event = start
    transfer = api.L2Transfer(
        pool,
        target,
        host_indices,
        device_indices,
        layer_mapper={1: 0, 3: 1}.get,
        is_draft=is_draft,
    )
    completion = engine.submit_host_to_device([transfer], layer_num=4)
    assert executed == [0, 1]
    assert not completion.finish_event.query()
    pool.load_to_device_per_layer.assert_not_called()
    host_indices.fill_(7)
    device_indices.fill_(8)
    device.progress()
    assert completion.finish_event.query()
    torch.testing.assert_close(target.data[:, [2, 3]], pool.data[:, :2])
    assert torch.all(target.data[:, 8] == -1)


def test_prepared_load_is_rebuilt_for_each_submission_not_tensor_identity(api):
    device = api.device_module
    engine = api.L2TransferEngine("kernel_ascend")
    pool = CopyPool(device)
    prepared_rows = []

    def prepare(device_pool, host_rows, device_rows, io_backend, *, is_draft):
        prepared_rows.append(host_rows.clone())
        return lambda layer_id: None

    pool.prepare_load_to_device = prepare
    host_indices = torch.tensor([0, 1])
    transfer = api.L2Transfer(pool, object(), host_indices, torch.tensor([2, 3]))
    engine.submit_host_to_device([transfer], layer_num=2)
    host_indices.add_(4)
    engine.submit_host_to_device([transfer], layer_num=2)
    assert [rows.tolist() for rows in prepared_rows] == [[0, 1], [4, 5]]


def test_prepared_load_invalid_callback_fails_before_start_or_submission(api):
    engine = api.L2TransferEngine("kernel_ascend")
    pool = CopyPool(api.device_module)
    pool.prepare_load_to_device = Mock(return_value=None)
    engine._start_event = Mock()
    transfer = api.L2Transfer(
        pool, object(), torch.tensor([0, 1]), torch.tensor([2, 3])
    )
    with pytest.raises(TypeError, match="must return a layer callable"):
        engine.submit_host_to_device([transfer], layer_num=2)
    engine._start_event.assert_not_called()
    assert engine.host_to_device_stream.work == []
