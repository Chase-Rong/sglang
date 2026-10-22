"""Real controller/allocator constructors with simulated L2 stream dependencies."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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

    shared_allocation_domain = None

    def prepare_transfer_indices(self, host, device, backend):
        return host, device

    def backup_from_device_all_layer_physical(self, *args):
        return self.backup_from_device_all_layer(*args)

    def load_to_device_per_layer_physical(self, *args, **kwargs):
        return self.load_to_device_per_layer(*args, **kwargs)

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


@pytest.fixture
def api(monkeypatch):
    from sglang.srt.managers import cache_controller
    from sglang.srt.mem_cache import l2_transfer
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry

    device = Device(timing=False)
    monkeypatch.setattr(torch, "npu", device, raising=False)
    monkeypatch.setattr(cache_controller, "device_module", device)
    monkeypatch.setattr(l2_transfer, "device_module", device)
    l2_transfer._timing_events_supported.cache_clear()
    yield SimpleNamespace(
        device_module=device,
        HybridCacheController=HybridCacheController,
        L2TransferEngine=l2_transfer.L2TransferEngine,
        TokenToKVPoolAllocator=TokenToKVPoolAllocator,
        PoolName=PoolName,
        PoolTransfer=PoolTransfer,
        HostPoolGroup=HostPoolGroup,
        PoolEntry=PoolEntry,
    )
    l2_transfer._timing_events_supported.cache_clear()


def controller_with_pools(api, *, adopted=True):
    device = api.device_module
    anchor = AdoptedPool(device) if adopted else CopyPool(device)
    state = CopyPool(device, layers=1)
    indexer = CopyPool(device)
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
            is_primary_index_anchor=name == api.PoolName.KV,
        )
        for name, pool, mapper in (
            (api.PoolName.KV, anchor, {0: 0, 2: 1}.get),
            (api.PoolName.MAMBA, state, {1: 0}.get),
            (api.PoolName.INDEXER, indexer, {0: 0, 2: 1}.get),
        )
    ]
    kv_pool = SimpleNamespace(
        device="cpu", layer_num=3, register_layer_transfer_counter=Mock()
    )
    allocator = api.TokenToKVPoolAllocator(
        size=16, dtype=torch.float32, device="cpu", kvcache=kv_pool, need_sort=False
    )
    allocator.alloc_for_host_restore = Mock(
        side_effect=lambda host: allocator.alloc(len(host))
    )
    controller = api.HybridCacheController(
        allocator,
        api.HostPoolGroup(entries),
        2,
        None,
        load_cache_event=None,
        io_backend="kernel_ascend",
        transfer_layer_id_max=3,
    )
    transfers = [
        api.PoolTransfer(name=api.PoolName.MAMBA, device_indices=torch.tensor([4, 5])),
        api.PoolTransfer(name=api.PoolName.INDEXER, indices_from_pool=api.PoolName.KV),
    ]
    return controller, anchor, state, indexer, transfers


def test_real_controller_adopts_existing_pages_and_joins_sidecar_events(api):
    controller, anchor, state, indexer, transfers = controller_with_pools(api)
    assert type(controller) is api.HybridCacheController
    assert type(controller.mem_pool_device_allocator) is api.TokenToKVPoolAllocator
    host = controller.write(torch.tensor([2, 3]), extra_pools=transfers)
    assert torch.equal(host, anchor.indices)
    anchor.alloc.assert_not_called()
    anchor.acquire_device_pages.assert_called_once()
    indexer.alloc.assert_not_called()
    state.alloc.assert_called_once_with(2)
    assert anchor.references == 2
    completion = controller.ack_write_queue[0]
    anchor.binding.record_reader.assert_called_once_with(completion.finish_event)
    api.device_module.progress()
    assert not completion.finish_event.query()
    anchor.runtime_ready.open = True
    api.device_module.progress()
    assert completion.finish_event.query()
    assert torch.all(anchor.data[:, host] == 123)
    assert completion.num_bytes == state.size_per_token * 2 + indexer.size_per_token * 2


def test_real_controller_regular_backup_allocates_host_pages(api):
    controller, anchor, _, _, transfers = controller_with_pools(api, adopted=False)
    assert controller.write(torch.tensor([2, 3]), extra_pools=transfers) is not None
    anchor.alloc.assert_called_once_with(2)


@pytest.mark.parametrize("failure", ["capacity", "exception", "derived"])
def test_adoption_rolls_back_cache_reference_and_sidecars(api, failure):
    controller, anchor, state, _, transfers = controller_with_pools(api)
    if failure == "capacity":
        state.alloc.side_effect = lambda count: None
    elif failure == "exception":
        state.alloc.side_effect = RuntimeError("allocation failed")
    else:
        transfers.append(
            api.PoolTransfer(
                name=api.PoolName.DRAFT_INDEXER,
                indices_from_pool=api.PoolName.DRAFT,
            )
        )
    if failure == "exception":
        with pytest.raises(RuntimeError, match="allocation failed"):
            controller.write(torch.tensor([2, 3]), extra_pools=transfers)
    else:
        assert controller.write(torch.tensor([2, 3]), extra_pools=transfers) is None
    assert anchor.references == 1
    assert not controller.write_queue and not controller.ack_write_queue
    assert all(transfer.host_indices is None for transfer in transfers)


def test_failed_backup_submission_drains_before_releasing_adopted_reference(api):
    controller, anchor, state, indexer, transfers = controller_with_pools(api)
    anchor.runtime_ready.open = True
    indexer.fail_backup = True
    with pytest.raises(RuntimeError, match="sidecar backup failed"):
        controller.write(torch.tensor([2, 3]), extra_pools=transfers)
    assert (
        controller.l2_transfer_engine.device_to_host_stream
        in api.device_module.synchronized
    )
    assert anchor.references == 1
    state.free.assert_called_once()
    assert not controller.write_queue and not controller.ack_write_queue


@pytest.mark.parametrize("failure", ["prepare", "sidecar", "exception"])
def test_restore_failure_returns_real_allocator_capacity_and_descriptors(api, failure):
    controller, _, _, _, _ = controller_with_pools(api)
    allocator = controller.mem_pool_device_allocator
    capacity = allocator.available_size()
    original_host = torch.tensor([6, 7])
    side = api.PoolTransfer(name=api.PoolName.MAMBA, host_indices=torch.tensor([0, 1]))
    entry = controller.mem_pool_host.entry_map[api.PoolName.MAMBA]
    entry.device_alloc_fn = Mock(return_value=None)
    entry.device_free_fn = Mock()
    prepare = None
    if failure == "prepare":

        def prepare(rows):
            raise RuntimeError("mapping rejected")

    if failure == "exception":
        entry.device_alloc_fn.side_effect = RuntimeError("sidecar rejected")
    if failure == "sidecar":
        assert controller.load(original_host, extra_pools=[side]) is None
    else:
        with pytest.raises(RuntimeError, match="rejected"):
            controller.load(
                original_host, extra_pools=[side], prepare_device_indices=prepare
            )
    allocator.alloc_for_host_restore.assert_called_once_with(original_host)
    assert allocator.available_size() == capacity
    assert side.device_indices is None
    assert not controller.load_queue


def test_caller_owned_restore_rows_survive_preparation_failure(api):
    controller, _, _, _, _ = controller_with_pools(api)
    allocator = controller.mem_pool_device_allocator
    rows = allocator.alloc(2)
    capacity = allocator.available_size()

    def reject(unused):
        raise RuntimeError("caller mapping rejected")

    with pytest.raises(RuntimeError, match="caller mapping"):
        controller.load(
            torch.tensor([6, 7]), device_indices=rows, prepare_device_indices=reject
        )
    assert allocator.available_size() == capacity
    allocator.alloc_for_host_restore.assert_not_called()


if __name__ == "__main__":
    run_pytest(__file__)
