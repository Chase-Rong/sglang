"""CPU tests of real HiCache page management and MemFabric borrowing.

Load the entire production HostKVCache module, without AST extraction or
replacing alloc/free/clear. Import-only framework dependencies are isolated;
only MemFabric allocation/registration is simulated with CPU storage. These
tests do not qualify NPU DMA, TP visibility or controller completion.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import test_memfabric_kv_pool as fabric_tests
import torch

SRT = Path(__file__).parents[4] / "python/sglang/srt"


def load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    allocator = fabric_tests.allocator.__wrapped__(monkeypatch)
    for name in (
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.pool_host",
        "sglang.srt.distributed",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "Borrowing must not invoke framework allocation or collectives"
        )

    dependencies = {
        "sglang.srt.distributed.parallel_state": {"get_world_group": unexpected},
        "sglang.srt.mem_cache.memory_pool": {"KVCache": object},
        "sglang.srt.mem_cache.pool_host.common": {
            "_cuda_host_unregister": unexpected,
            "get_allocator_from_storage": unexpected,
        },
        "sglang.srt.runtime_context": {"get_parallel": unexpected},
        "sglang.srt.utils": {"is_cuda": lambda: False, "is_hip": lambda: False},
    }
    for name, attributes in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    base = load_module(
        monkeypatch,
        "sglang.srt.mem_cache.pool_host.base",
        SRT / "mem_cache/pool_host/base.py",
    )
    host = load_module(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.hicache_host_pool",
        SRT / "hardware_backend/npu/hicache_host_pool.py",
    )
    binding = load_module(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.hicache_offload",
        SRT / "hardware_backend/npu/hicache_offload.py",
    )
    allocations = {}
    device_events = []
    stream = SimpleNamespace(waits=[], synchronizations=0)
    stream.wait_event = stream.waits.append

    class Event:
        def __init__(self):
            self.complete = False
            self.stream = None
            device_events.append(self)

        def record(self, current_stream):
            self.stream = current_stream

        def query(self):
            return self.complete

    def synchronize():
        stream.synchronizations += 1
        for event in device_events:
            event.complete = True

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            Event=Event, current_stream=lambda: stream, synchronize=synchronize
        ),
        raising=False,
    )

    def malloc(size):
        storage = torch.empty(size, dtype=torch.uint8)
        pointer = storage.data_ptr()
        allocations[pointer] = storage
        allocator.events.append(("malloc", size))
        return pointer

    def register(pointer, shape, template, device_id, shared):
        allocator.events.append(("register", shared))
        return allocations[pointer].view(template.dtype).view(shape), pointer + 4096

    def free(pointer):
        allocator.events.append(("free", pointer))
        del allocations[pointer]

    monkeypatch.setattr(sys.modules["memfabric_hybrid"].offload, "malloc", malloc)
    monkeypatch.setattr(sys.modules["memfabric_hybrid"].offload, "free", free)
    operations = SimpleNamespace(
        memfabric_region=register,
        prefill_dma_copy=unexpected,
        unregister_memfabric_region=lambda *args: allocator.events.append(
            ("unregister",)
        ),
    )
    monkeypatch.setattr(allocator.module, "native_ops", lambda: operations)
    backings = []

    def make(*, rope_dim=2, page_size=4, pages=5):
        backing = allocator.backing.DecodeHostKvBacking(
            layers=2,
            pages=pages,
            page_size=page_size,
            latent_dim=8,
            rope_dim=rope_dim,
            dtype=torch.bfloat16,
            device_id=0,
            host_budget=1 << 20,
            integrated=True,
        )
        backings.append(backing)
        device_pool = SimpleNamespace(
            store_dtype=torch.bfloat16,
            size=page_size * (pages - 1),
            page_size=page_size,
            start_layer=3,
            end_layer=5,
            layer_num=2,
            kv_lora_rank=8,
            qk_rope_head_dim=rope_dim,
            device="cpu",
            layer_shard_enabled=False,
        )
        return backing, device_pool

    yield SimpleNamespace(
        make=make,
        pool_type=host.MemFabricHiCachePool,
        base=base.HostKVCache,
        events=allocator.events,
        allocations=allocations,
        binding_type=binding.OffloadPageBinding,
        stream=stream,
        device_events=device_events,
    )
    for backing in reversed(backings):
        backing.pool.close(drained=True)


def make_pool(runtime, **kwargs):
    backing, device_pool = runtime.make(**kwargs)
    return runtime.pool_type(device_pool=device_pool, backing=backing)


@pytest.mark.parametrize("end_layer", [4, 5])
def test_host_geometry_uses_layer_count_with_either_endpoint_convention(
    runtime, end_layer
):
    backing, device_pool = runtime.make()
    device_pool.end_layer = end_layer
    pool = runtime.pool_type(backing, device_pool)
    assert pool.layer_num == device_pool.layer_num == len(backing.regions) == 2
    device_pool.layer_num = 1
    with pytest.raises(ValueError, match="geometry differ"):
        runtime.pool_type(backing, device_pool)


@pytest.mark.parametrize("rope_dim", [0, 2])
def test_borrows_real_memfabric_regions_without_second_allocation(runtime, rope_dim):
    backing, device_pool = runtime.make(rope_dim=rope_dim)
    before = list(runtime.events)
    pool = runtime.pool_type(backing, device_pool)
    assert isinstance(pool, runtime.base)
    assert runtime.events == before
    assert pool.binding is None
    assert pool.get_size_per_token() == 2 * (8 + rope_dim) * 2
    assert pool.size * pool.size_per_token == backing.nbytes
    assert len(runtime.allocations) == 2 * (1 + bool(rope_dim))
    for layer, regions in enumerate(backing.regions):
        for view, region, width in zip(pool.layer_views(layer), regions, (8, rope_dim)):
            assert view.shape == (5, 4, 1, width)
            assert view.is_contiguous()
            if region is not None:
                assert view.data_ptr() == region.host_ptr == region.tensor.data_ptr()
                assert region.device_ptr != region.host_ptr
            else:
                assert view.numel() == 0


@pytest.mark.parametrize("page_size", [1, 4, 128])
def test_real_page_allocation_exhaustion_and_fragmented_reuse(runtime, page_size):
    pool = make_pool(runtime, page_size=page_size)
    assert pool.available_size() == pool.usable_size == 4 * page_size
    assert pool.slot_used[:page_size].all()
    allocated = [pool.alloc(page_size) for _ in range(4)]
    assert torch.equal(torch.cat(allocated), torch.arange(page_size, pool.size))
    assert pool.alloc(page_size) is None
    assert pool.alloc(0).numel() == 0
    assert pool.free(allocated[2]) == page_size
    assert pool.free(allocated[0]) == page_size
    reused = pool.alloc(2 * page_size)
    assert torch.equal(reused, torch.cat((allocated[2], allocated[0])))
    assert pool.available_size() == 0
    for slots in (allocated[1], allocated[3], reused):
        pool.free(slots)
    assert pool.available_size() == pool.usable_size
    assert torch.equal(
        pool.alloc(pool.usable_size).sort().values, torch.arange(page_size, pool.size)
    )


@pytest.mark.parametrize("method", ["clear", "reset"])
def test_page0_survives_reset_and_payload_is_not_cleared(runtime, method):
    pool = make_pool(runtime)
    pool.alloc(pool.usable_size)
    region = pool.regions[0][0]
    region.tensor.fill_(7)
    before = list(runtime.events)
    getattr(pool, method)()
    assert runtime.events == before
    assert torch.all(region.tensor == 7)
    assert pool.available_size() == pool.usable_size
    assert pool.slot_used[: pool.page_size].all()
    assert pool.alloc(pool.usable_size).tolist() == list(
        range(pool.page_size, pool.size)
    )
    with pytest.raises(ValueError, match="page0"):
        pool.free(torch.arange(pool.page_size))


@pytest.mark.parametrize("size", [-4, -1, 1, 4.0, True])
def test_invalid_alloc_does_not_change_capacity(runtime, size):
    pool = make_pool(runtime)
    with pytest.raises(ValueError):
        pool.alloc(size)
    assert pool.available_size() == pool.usable_size
    assert not pool.slot_used[pool.page_size :].any()


@pytest.mark.parametrize(
    "slots",
    [
        [-4, -3, -2, -1],
        [0, 1, 2, 3],
        [20, 21, 22, 23],
        [4, 5],
        [4, 5, 6, 8],
        [5, 6, 7, 8],
        [4, 5, 6, 7, 4, 5, 6, 7],
    ],
)
def test_invalid_free_cannot_inflate_capacity(runtime, slots):
    pool = make_pool(runtime)
    pool.alloc(pool.usable_size)
    before = pool.slot_used.clone()
    with pytest.raises(ValueError):
        pool.free(torch.tensor(slots))
    assert pool.available_size() == 0
    assert torch.equal(pool.slot_used, before)


def test_double_free_and_caller_mutation_cannot_corrupt_free_list(runtime):
    pool = make_pool(runtime)
    slots = pool.alloc(pool.usable_size)
    released = slots[:4].clone()
    pool.free_unreferenced(released)
    with pytest.raises(ValueError, match="Double-free"):
        pool.free(released)
    released.zero_()
    assert pool.alloc(4).tolist() == [4, 5, 6, 7]
    assert pool.free(torch.empty(0, dtype=torch.int64)) == 0
    for invalid in (slots.float(), slots.view(4, 4)):
        with pytest.raises(ValueError):
            pool.free(invalid)


@pytest.mark.parametrize("rope_dim", [0, 2])
def test_page_serialization_keeps_split_fsa_layout(runtime, rope_dim):
    pool = make_pool(runtime, rope_dim=rope_dim)
    slots = pool.alloc(pool.usable_size)
    for regions in pool.regions:
        for region in regions:
            if region is not None:
                region.tensor.fill_(-99)
    data = torch.arange(pool.layer_num * pool.page_size * pool.kv_cache_dim).to(
        pool.dtype
    )
    for index in (slots[0].item(), slots[-pool.page_size].item()):
        pool.set_from_flat_data_page(index, data)
        assert torch.equal(pool.get_data_page(index), data)
        assert pool.get_data_page(index, flat=False).shape == (2, 4, 1, 8 + rope_dim)
        expected = data.view(2, 4, 8 + rope_dim)
        for layer, regions in enumerate(pool.regions):
            assert torch.equal(
                regions[0].tensor[index : index + 4], expected[layer, :, :8]
            )
            if rope_dim:
                assert torch.equal(
                    regions[1].tensor[index : index + 4], expected[layer, :, 8:]
                )
    for regions in pool.regions:
        for region in regions:
            if region is not None:
                assert torch.all(region.tensor[:4] == -99)
                assert torch.all(region.tensor[8:16] == -99)
    dummy = pool.get_dummy_flat_data_page()
    assert dummy.dtype == pool.dtype and dummy.shape == data.shape
    assert torch.count_nonzero(dummy) == 0


def test_page_access_rejects_unallocated_reserved_and_malformed_pages(runtime):
    pool = make_pool(runtime)
    for index in (-4, 0, 1, 4, pool.size):
        with pytest.raises(ValueError):
            pool.get_data_page(index)
        with pytest.raises(ValueError):
            pool.set_from_flat_data_page(index, pool.get_dummy_flat_data_page())
    pool.alloc(4)
    for data in (
        pool.get_dummy_flat_data_page().float(),
        torch.empty(0, dtype=pool.dtype),
    ):
        with pytest.raises(ValueError):
            pool.set_from_flat_data_page(4, data)


def test_tp_peer_rejects_cpu_payload_access_before_any_tensor_read(
    runtime, monkeypatch
):
    pool = make_pool(runtime)
    pool.alloc(4)
    monkeypatch.setattr(pool.backing.pool, "_writer", False)
    for method, args in (
        (pool.get_data_page, (4,)),
        (pool.set_from_flat_data_page, (4, pool.get_dummy_flat_data_page())),
    ):
        with pytest.raises(RuntimeError, match="TP peer"):
            method(*args)
    assert pool.layer_views(0)[0].data_ptr() == pool.regions[0][0].host_ptr
    pool.destroy()
    monkeypatch.setattr(pool.backing.pool, "_writer", True)


def test_borrowed_close_never_frees_or_unregisters_external_owner(runtime):
    pool = make_pool(runtime)
    backing = pool.backing
    region = backing.regions[0][0]
    before = list(runtime.events)
    pool.destroy()
    pool.close()
    assert runtime.events == before
    assert not backing.pool._closed and not region.closed
    region.tensor.fill_(3)
    for operation in (
        lambda: pool.alloc(4),
        pool.clear,
        pool.reset,
        pool.available_size,
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()
    backing.pool.close(drained=True)
    assert sum(event[0] == "free" for event in runtime.events) == 4
    before = list(runtime.events)
    pool.close()
    backing.pool.close(drained=True)
    assert runtime.events == before


def test_external_owner_close_invalidates_adapter(runtime):
    pool = make_pool(runtime)
    pool.backing.pool.close(drained=True)
    for operation in (lambda: pool.layer_views(0), lambda: pool.alloc(4), pool.clear):
        with pytest.raises(RuntimeError, match="closed"):
            operation()
    pool.destroy()


@pytest.mark.parametrize(
    "attribute,value", [("page_size", 8), ("kv_lora_rank", 4), ("layer_num", 1)]
)
def test_failed_borrow_does_not_close_or_allocate_owner(runtime, attribute, value):
    backing, device_pool = runtime.make()
    setattr(device_pool, attribute, value)
    before = list(runtime.events)
    with pytest.raises(ValueError):
        runtime.pool_type(backing, device_pool)
    assert runtime.events == before
    assert not backing.pool._closed
    assert all(
        not region.closed for regions in backing.regions for region in regions if region
    )


def test_unbound_transfer_and_acquire_fail_explicitly(runtime):
    pool = make_pool(runtime)
    slots = pool.alloc(4)
    with pytest.raises(RuntimeError, match="Bind"):
        pool.acquire_device_pages(slots)
    with pytest.raises(RuntimeError, match="Bind"):
        pool.load_to_device_per_layer(pool.device_pool, slots, slots, 0, "kernel")
    with pytest.raises(RuntimeError, match="Bind"):
        pool.backup_from_device_all_layer(pool.device_pool, slots, slots, "kernel")


def test_backing_binding_is_shared_and_recycling_waits_for_last_reader(runtime):
    backing, device_pool = runtime.make()
    before = list(runtime.events)
    pool = backing.bind_hicache(device_pool)
    assert backing.bind_hicache(device_pool) is pool
    assert backing.page_binding is pool.binding
    assert runtime.events == before
    binding = pool.binding
    assert isinstance(binding, runtime.binding_type)
    assert binding.ensure_pages(torch.tensor([1, 2]))
    cached = pool.acquire_device_pages(torch.arange(4, 12))
    assert cached.tolist() == list(range(4, 12))
    assert binding.references[1:3].tolist() == [2, 2]
    assert pool.free(cached) == 8
    assert binding.references[1:3].tolist() == [1, 1]
    assert pool.available_size() == pool.usable_size - 8
    binding.release_device_pages(torch.tensor([1, 2]))
    assert pool.slot_used[cached].all()
    binding.reap()
    assert pool.available_size() == pool.usable_size - 8
    runtime.device_events[-1].complete = True
    binding.reap()
    assert pool.available_size() == pool.usable_size
    assert not pool.slot_used[cached].any()
    assert pool.slot_used[: pool.page_size].all()


def test_load_only_restores_mapping_and_backup_waits_for_ready_event(runtime):
    backing, device_pool = runtime.make()
    pool = backing.bind_hicache(device_pool)
    binding = pool.binding
    assert binding.ensure_pages(torch.tensor([1]))
    cached = pool.acquire_device_pages(torch.arange(4, 8))
    binding.release_device_pages(torch.tensor([1]))
    before = list(runtime.events)
    region = pool.regions[0][0]
    region.tensor.fill_(9)
    destination = torch.arange(12, 16)
    pool.load_to_device_per_layer(device_pool, cached, destination, 0, "kernel")
    pool.load_to_device_per_layer(device_pool, cached, destination, 1, "kernel")
    pool.load_to_device_per_layer(device_pool, cached, destination, 0, "kernel")
    assert binding.device_to_host_cpu[3] == 1
    assert binding.references[1] == 2
    assert runtime.events == before
    assert torch.all(region.tensor == 9)
    with pytest.raises(RuntimeError, match="ready event"):
        pool.backup_from_device_all_layer(device_pool, cached, destination, "kernel")
    ready = torch.npu.Event()
    ready.record(runtime.stream)
    binding.ready_event = ready
    pool.backup_from_device_all_layer(device_pool, cached, destination, "kernel")
    assert runtime.stream.waits == [ready]
    assert not ready.query()
    assert runtime.events == before
    with pytest.raises(NotImplementedError, match="main KV"):
        pool.load_to_device_per_layer(
            device_pool, cached, destination, 0, "kernel", is_draft=True
        )
    with pytest.raises(NotImplementedError, match="main KV"):
        pool.backup_from_device_all_layer(object(), cached, destination, "kernel")


def test_bound_reset_drains_and_clears_mapping_before_page_reuse(runtime):
    backing, device_pool = runtime.make()
    pool = backing.bind_hicache(device_pool)
    binding = pool.binding
    assert binding.ensure_pages(torch.tensor([1, 2]))
    binding.release_device_pages(torch.tensor([1]))
    assert binding.retired
    pool.reset()
    assert runtime.stream.synchronizations == 1
    assert not binding.references.any()
    assert not binding.device_to_host_cpu.any()
    assert not binding.device_to_host.any()
    assert not binding.retired
    assert pool.available_size() == pool.usable_size
    assert pool.alloc(pool.usable_size).tolist() == list(range(4, 20))
