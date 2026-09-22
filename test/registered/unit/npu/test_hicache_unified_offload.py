"""Real UnifiedRadix/HiCache constructors with CPU storage and simulated IO.

Exercises the default FULL+MAMBA strategy with KDA states and a registered
Indexer sidecar. No AST extraction, accelerator execution, or model E2E.
"""

import atexit
from array import array
from types import SimpleNamespace

import pytest
import torch
from test_hicache_offload_completion import Device, Gate

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu import hicache_mamba_pool
from sglang.srt.hardware_backend.npu.hicache_offload import OffloadPageBinding
from sglang.srt.managers import cache_controller as controller_module
from sglang.srt.mem_cache import l2_transfer, memory_pool
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer, SidecarPoolSpec
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
)
from sglang.srt.mem_cache.pool_host import PoolEntry
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    FullComponent,
    MambaComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import restore_context, snapshot_context
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


class CpuHostSlots:
    page_size = 1
    device = "cpu"
    layout = "layer_first"
    can_use_write_back_jit = False

    def __init__(self, capacity):
        self.size = capacity + 1
        self.logical_size = capacity
        self.page_num = self.size
        self.slot_used = torch.zeros(self.size, dtype=torch.bool)
        self.slot_used[0] = True
        self.alloc_calls = 0

    def alloc(self, count):
        self.alloc_calls += 1
        available = torch.where(~self.slot_used)[0]
        if len(available) < count:
            return None
        indices = available[:count].clone()
        self.slot_used[indices] = True
        return indices

    def free_unreferenced(self, indices):
        assert self.slot_used[indices].all()
        self.slot_used[indices] = False
        return len(indices)

    def free(self, indices):
        return self.free_unreferenced(indices)

    def available_size(self):
        return int((~self.slot_used).sum())

    def destroy(self):
        pass


class CpuAdoptedHost(CpuHostSlots):
    def __init__(self, device_pool, device, capacity=4):
        super().__init__(capacity)
        self.device_pool = device_pool
        self.device_module = device
        self.layer_num = device_pool.layer_num
        self.payload = torch.zeros(self.layer_num, self.size, 2)
        self.size_per_token = self.layer_num * 2 * self.payload.element_size()
        self.backing = SimpleNamespace(pool=SimpleNamespace(shared_group=None))
        self.binding = OffloadPageBinding(self, "cpu", device_pool.size + 1)
        self.binding.ready_event = device.Event()
        self.binding.ready_event.record()

    def acquire_device_pages(self, device_indices):
        return self.binding.retain_device_rows(device_indices)

    def free(self, indices):
        return self.binding.release_cached_rows(indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        assert device_pool is self.device_pool
        self.device_module.current_stream().wait_event(self.binding.ready_event)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        assert device_pool is self.device_pool and not is_draft
        if layer_id == 0:
            self.binding.restore_mapping(host_indices, device_indices)


class CpuStateHost(CpuHostSlots):
    def __init__(self, buffers, device, capacity):
        super().__init__(capacity)
        self.device_buffers = buffers
        self.device_module = device
        self.layer_num = buffers[0].shape[0]
        self.buffers = [
            torch.full(
                (buffer.shape[0], self.size, *buffer.shape[2:]),
                -1,
                dtype=buffer.dtype,
            )
            for buffer in buffers
        ]
        self.size_per_token = sum(
            buffer[:, 0].numel() * buffer.element_size() for buffer in buffers
        )
        self.copy_ready = Gate()
        self.copy_ready.open = True
        self.submissions = []

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host_rows, device_rows = host_indices.clone(), device_indices.clone()
        self.submissions.append(("backup", host_rows, device_rows))
        stream = self.device_module.current_stream()
        stream.wait_event(self.copy_ready)

        def copy():
            for target, source in zip(self.buffers, self.device_buffers):
                target[:, host_rows] = source[:, device_rows]

        stream.enqueue(copy)

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        assert not is_draft
        host_rows, device_rows = host_indices.clone(), device_indices.clone()
        self.submissions.append(("load", host_rows, device_rows))
        stream = self.device_module.current_stream()
        stream.wait_event(self.copy_ready)

        def copy():
            for source, target in zip(self.buffers, self.device_buffers):
                target[layer_id, device_rows] = source[layer_id, host_rows]

        stream.enqueue(copy)


class CpuRestoreAllocator(TokenToKVPoolAllocator):
    def __init__(self, kv_pool, binding):
        self.binding = binding
        self.restore_calls = 0
        super().__init__(
            size=kv_pool.size,
            dtype=torch.float32,
            device="cpu",
            kvcache=kv_pool,
            need_sort=True,
        )

    def alloc(self, count):
        indices = super().alloc(count)
        if indices is not None and not self.binding.ensure_pages(indices):
            super().free(indices)
            return None
        return indices

    def alloc_for_host_restore(self, host_indices):
        self.restore_calls += 1
        indices = super().alloc(len(host_indices))
        if indices is not None:
            self.binding.restore_mapping(host_indices, indices)
        return indices

    def free(self, indices):
        if self.free_group is None and indices.numel():
            self.binding.release_device_pages(indices)
        super().free(indices)


@pytest.fixture
def unified_offload(monkeypatch):
    context = snapshot_context()
    device = Device(timing=False)
    monkeypatch.setattr(torch, "npu", device, raising=False)
    monkeypatch.setattr(controller_module, "device_module", device)
    monkeypatch.setattr(l2_transfer, "device_module", device)
    monkeypatch.setattr(memory_pool, "_cpu_has_amx_support", False)
    l2_transfer._timing_events_supported.cache_clear()
    cache = None
    try:
        args = ServerArgs(
            model_path="dummy",
            device="cpu",
            page_size=1,
            enable_hierarchical_cache=True,
            hicache_io_backend="kernel_ascend",
            hicache_mem_layout="layer_first",
            hicache_write_policy="write_through",
        )
        args._mamba_cache_chunk_size = 2
        set_global_server_args_for_scheduler(args)
        shape = KimiLinearStateShape.create(
            tp_world_size=1, num_heads=1, head_dim=2, conv_kernel_size=2
        )
        state_params = KimiLinearCacheParams(
            shape=shape,
            layers=[1],
            dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
        )
        requests = HybridReqToTokenPool(
            size=2,
            mamba_size=4,
            mamba_spec_state_size=2,
            max_context_len=16,
            device="cpu",
            enable_memory_saver=False,
            cache_params=state_params,
            mamba_layer_ids=[1],
            enable_mamba_extra_buffer=False,
        )
        full_pool = MHATokenToKVPool(
            size=8,
            page_size=1,
            dtype=torch.float32,
            head_num=1,
            head_dim=2,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        hybrid_pool = HybridLinearKVPool(
            size=8,
            page_size=1,
            dtype=torch.float32,
            head_num=1,
            head_dim=2,
            full_attention_layer_ids=[0, 2],
            device="cpu",
            mamba_pool=requests.mamba_pool,
            full_kv_pool=full_pool,
        )
        anchor = CpuAdoptedHost(full_pool, device)
        full_pool.integrated_offload = True
        full_pool.decode_host_backing = SimpleNamespace(hicache_pool=anchor)
        allocator = CpuRestoreAllocator(hybrid_pool, anchor.binding)

        def mamba_host_factory(device_pool, *unused_args, **unused_kwargs):
            state = device_pool.mamba_cache
            return CpuStateHost([state.temporal, *state.conv], device, device_pool.size)

        monkeypatch.setattr(hicache_mamba_pool, "NpuMambaPoolHost", mamba_host_factory)
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=requests,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
            cache = UnifiedRadixCache(params=params)
            cache.init_hicache(args, params)
        cache.write_through_threshold = 1 << 30
        cache.load_back_threshold = 0
        index_data = torch.arange(2 * 9 * 2).reshape(2, 9, 2).float()
        indexer = CpuStateHost([index_data], device, anchor.logical_size)
        indexer.binding = anchor.binding
        cache.register_sidecar_pool(
            SidecarPoolSpec(PoolName.INDEXER, PoolName.KV),
            PoolEntry(
                name=PoolName.INDEXER,
                host_pool=indexer,
                device_pool=full_pool,
                layer_mapper={0: 0, 2: 1}.get,
            ),
        )
        assert type(cache) is UnifiedRadixCache
        assert type(cache.cache_controller) is HybridCacheController
        assert type(cache.components[ComponentType.FULL]) is FullComponent
        assert type(cache.components[ComponentType.MAMBA]) is MambaComponent
        assert cache.cache_controller.mem_pool_host.anchor_entry.host_pool is anchor
        assert cache.cache_controller.layer_num == 3
        yield SimpleNamespace(
            cache=cache,
            allocator=allocator,
            requests=requests,
            anchor=anchor,
            mamba=cache.host_pool_group.get_pool(PoolName.MAMBA),
            indexer=indexer,
            device=device,
            index_data=index_data,
        )
    finally:
        if cache is not None:
            atexit.unregister(cache.shutdown)
            cache.release_host_resources()
        l2_transfer._timing_events_supported.cache_clear()
        restore_context(context)


def test_full_mamba_default_constructor_backup_evict_hit_load_lifecycle(
    unified_offload,
):
    fixture = unified_offload
    cache, allocator = fixture.cache, fixture.allocator
    indices = allocator.alloc(4)
    assert indices is not None
    host_indices = fixture.anchor.binding.host_rows(indices).clone()
    fixture.anchor.payload[:, host_indices] = 123
    original_payload = fixture.anchor.payload.clone()
    assert fixture.anchor.available_size() == 0
    state_indices = fixture.requests.mamba_allocator.alloc(1)
    state = fixture.requests.mamba_pool.mamba_cache
    state_buffers = [state.temporal, *state.conv]
    for marker, buffer in enumerate(state_buffers, start=20):
        buffer[:, state_indices] = marker
    expected_state = [buffer[:, state_indices].clone() for buffer in state_buffers]
    expected_index = fixture.index_data[:, indices].clone()
    key = RadixKey(array("q", [101, 102, 103, 104]))
    cache.insert(InsertParams(key=key, value=indices, mamba_value=state_indices))
    match = cache.match_prefix(MatchPrefixParams(key=key))
    node = match.last_device_node
    assert match.device_indices.tolist() == indices.tolist()
    fixture.indexer.copy_ready.open = False
    assert cache.tree_core.node_by_id(node).parent is cache.tree_core.root_node
    action = BackupKV(node_ids=[node])
    assert cache._execute_and_commit_kv_backup(action, False) == 4
    controller = cache.cache_controller
    write_ack = controller.ack_write_queue[0]
    assert fixture.anchor.binding.reader_events[-1] is write_ack.finish_event
    assert list(fixture.anchor.binding.reader_events).count(write_ack.finish_event) == 1
    fixture.device.progress()
    cache.writing_check()
    assert not write_ack.finish_event.query()
    assert node in cache.ongoing_write_through
    assert fixture.anchor.alloc_calls == 1
    assert torch.all(fixture.anchor.binding.references[host_indices] == 2)

    fixture.indexer.copy_ready.open = True
    fixture.device.progress()
    cache.writing_check()
    assert not cache.ongoing_write_through
    assert cache.tree_core.is_backuped(node)
    assert (
        write_ack.num_bytes
        == fixture.mamba.size_per_token + 4 * fixture.indexer.size_per_token
    )
    torch.testing.assert_close(fixture.anchor.payload, original_payload)
    torch.testing.assert_close(
        fixture.indexer.buffers[0][:, host_indices], expected_index
    )
    cache.evict(EvictParams(num_tokens=4, mamba_num=1))
    assert cache.tree_core.is_full_device_evicted(node)
    assert cache.tree_core.get_component_device_value(node, ComponentType.MAMBA) is None
    assert torch.all(fixture.anchor.binding.references[host_indices] == 1)
    assert fixture.anchor.available_size() == 0
    for buffer in state_buffers:
        buffer.fill_(-99)
    fixture.index_data.fill_(-99)
    host_match = cache.match_prefix(MatchPrefixParams(key=key))
    assert host_match.host_hit_length == 4
    assert host_match.mamba_host_hit_length >= 1

    assert cache.load_back(host_match.last_host_node)
    assert allocator.restore_calls == 1
    loaded_indices = cache.tree_core.get_component_device_value(
        node, ComponentType.FULL
    )
    torch.testing.assert_close(
        fixture.anchor.binding.host_rows(loaded_indices), host_indices
    )
    assert fixture.anchor.alloc_calls == 1
    fixture.indexer.copy_ready.open = False
    assert cache.ready_to_load_host_cache() >= 0
    load_ack = controller.ack_load_queue[0]
    assert fixture.anchor.binding.reader_events[-1] is load_ack.finish_event
    fixture.device.progress()
    cache.loading_check()
    assert node in cache.ongoing_load_back
    assert not load_ack.finish_event.query()
    fixture.indexer.copy_ready.open = True
    fixture.device.progress()
    cache.loading_check()
    assert not cache.ongoing_load_back
    assert load_ack.num_bytes == write_ack.num_bytes
    loaded_state = cache.tree_core.get_component_device_value(node, ComponentType.MAMBA)
    for buffer, expected in zip(state_buffers, expected_state):
        torch.testing.assert_close(buffer[:, loaded_state], expected)
    torch.testing.assert_close(fixture.index_data[:, loaded_indices], expected_index)
    torch.testing.assert_close(fixture.anchor.payload, original_payload)
    cache.sanity_check()
    cache.evict(EvictParams(num_tokens=4, mamba_num=1))
    cache.evict_host(4)
    cache.evict_host(1, ComponentType.MAMBA)
    fixture.device.progress()
    fixture.anchor.binding.reap()
    assert fixture.anchor.available_size() == 4
    assert not fixture.anchor.binding.references.any()


@pytest.mark.parametrize("direction", ["backup", "load"])
def test_cached_host_reference_still_fences_indexer_device_slot_reuse(
    unified_offload, direction
):
    fixture = unified_offload
    controller = fixture.cache.cache_controller
    indices = fixture.allocator.alloc(4)
    expected = fixture.index_data[:, indices].clone()
    transfer = PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)
    fixture.indexer.copy_ready.open = direction == "load"
    host_indices = controller.write(indices, extra_pools=[transfer])
    if direction == "load":
        fixture.device.progress()
        fixture.allocator.free(indices)
        fixture.allocator.merge_and_sort_free()
        fixture.index_data.fill_(-99)
        fixture.indexer.copy_ready.open = False
        transfer = PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)
        indices = controller.load(host_indices, extra_pools=[transfer])
        assert controller.start_loading() >= 0
        completion = controller.ack_load_queue[-1].finish_event
    else:
        completion = controller.ack_write_queue[-1].finish_event
    fixture.device.progress()
    assert not completion.query()
    assert fixture.indexer.submissions[-1][0] == direction
    assert fixture.indexer.submissions[-1][2].tolist() == indices.tolist()
    fixture.allocator.free(indices)
    assert torch.all(fixture.anchor.binding.references[host_indices] == 1)
    assert fixture.anchor.available_size() == 0
    fixture.allocator.merge_and_sort_free()
    reused = fixture.allocator.alloc_for_host_restore(host_indices)
    assert torch.equal(reused, indices)
    observed = []

    def overwrite_reused_indexer():
        observed.append(fixture.index_data[:, indices].clone())
        fixture.index_data[:, reused] = -777

    fixture.device.current_stream().enqueue(overwrite_reused_indexer)
    fixture.device.progress()
    assert observed == []
    assert not completion.query()
    assert fixture.anchor.available_size() == 0
    fixture.indexer.copy_ready.open = True
    fixture.device.progress()
    assert completion.query()
    assert len(observed) == 1
    torch.testing.assert_close(observed[0], expected)
    torch.testing.assert_close(fixture.indexer.buffers[0][:, host_indices], expected)
    fixture.allocator.free(reused)
    fixture.anchor.free(host_indices)
    fixture.device.progress()
    fixture.anchor.binding.reap()
    assert fixture.anchor.available_size() == 4
