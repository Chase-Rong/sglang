"""Resident KV and compressed-indexer round trips through different physical pages."""

from types import SimpleNamespace

import pytest
import torch
from test_prefill_offload_runtime import production_pool_type

from sglang.srt.hardware_backend.npu.hicache_resident_pool import NpuResidentHiCachePool
from sglang.srt.hardware_backend.npu.hicache_indexer_pool import NPUIndexerPoolHost
from sglang.srt.hardware_backend.npu.hicache_indexer_mapping import (
    IndexerPageMapping,
    get_indexer_mapping,
    maybe_update_indexer_mapping,
)


def rows(pages):
    return (torch.tensor(pages)[:, None] * 4 + torch.arange(4)).flatten()


def make_pool(layers, rope_dim=0):
    pool = production_pool_type()(
        size=64, page_size=4, dtype=torch.bfloat16, kv_lora_rank=8,
        qk_rope_head_dim=rope_dim, layer_num=layers, device="cpu",
        enable_memory_saver=False, index_head_dim=4, index_kpool=2,
        index_kpool_compress=True, max_running_requests=2,
        start_layer=0, end_layer=layers,
    )
    pool.layer_shard_enabled = False
    return pool


def test_resident_rejects_nonzero_rope_before_allocating_host():
    target = make_pool(2, rope_dim=2)
    with pytest.raises(NotImplementedError, match="qk_rope_head_dim=0"):
        NpuResidentHiCachePool(target, 2, 0, 4, pin_memory=False)


def test_resident_requires_pinned_memory_for_npu_before_allocating_host():
    target = make_pool(2)
    target.device = "npu"
    with pytest.raises(ValueError, match="pinned CPU"):
        NpuResidentHiCachePool(target, 2, 0, 4, pin_memory=False)


def test_resident_target_draft_and_indexer_restore_to_different_page_groups():
    target, draft = make_pool(2), make_pool(1)
    anchor = NpuResidentHiCachePool(
        target, 2, 0, 4, pin_memory=False, mtp_draft_device_pools=(draft,),
    )
    indexer = NPUIndexerPoolHost(target, anchor, pin_memory=False)
    mapping = target.hicache_indexer_mapping
    mapping.update_request_pages(torch.tensor([3, 5]))
    host_rows, source_rows, destination_rows = rows([4, 2]), rows([3, 5]), rows([7, 1])
    for layer, pool in enumerate((target, draft), start=1):
        for component in (pool.k_buffer, pool.v_buffer, pool.index_k_buffer):
            component.copy_(torch.arange(component.numel()).reshape_as(component) % 53 + layer * 100)
    expected = []
    for pool in (target, draft):
        for layer in range(pool.layer_num):
            expected.append((
                pool.k_buffer[layer].view(-1, 8)[source_rows].clone(),
                pool.index_k_buffer[layer].view(-1, 4)[12:16].clone(),
            ))
    anchor.backup_from_device_all_layer(target, host_rows, source_rows, "kernel_ascend")
    indexer.backup_from_device_all_layer(target, host_rows, source_rows, "kernel_ascend")
    for pool in (target, draft):
        for component in (pool.k_buffer, pool.v_buffer, pool.index_k_buffer):
            component.fill_(-1)
    mapping.release_device_pages(torch.tensor([3, 5]))
    mapping.prepare_restore(torch.empty(0, dtype=torch.int64), destination_rows)
    for pool, is_draft, first in ((target, False, 0), (draft, True, 2)):
        for local in range(pool.layer_num):
            layer = first + local
            anchor.load_to_device_per_layer(pool, host_rows, destination_rows, layer, "kernel_ascend", is_draft=is_draft)
            indexer.load_to_device_per_layer(pool, host_rows, destination_rows, layer, "kernel_ascend", is_draft=is_draft)
            k, compressed = expected[layer]
            torch.testing.assert_close(pool.k_buffer[local].view(-1, 8)[destination_rows], k)
            torch.testing.assert_close(pool.index_k_buffer[local].view(-1, 4)[28:32], compressed)
            assert torch.all(pool.k_buffer[local, 0] == -1)


def test_resident_mapping_updates_after_scheduler_writes_request_table():
    target = make_pool(2)
    NpuResidentHiCachePool(target, 2, 0, 4, pin_memory=False)
    allocator = SimpleNamespace(offload_backing=None, get_kvcache=lambda: target)
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
        req_to_token_pool=SimpleNamespace(req_to_token=rows([3, 5]).reshape(1, -1)),
    )
    maybe_update_indexer_mapping(batch, torch.tensor([0]), torch.tensor([8]))
    assert get_indexer_mapping(allocator) is target.hicache_indexer_mapping
    assert get_indexer_mapping(allocator).indexer_page_locations(torch.tensor([3, 5])).tolist() == [12, 14]


def test_resident_rejects_partial_page_transfer():
    target = make_pool(2)
    anchor = NpuResidentHiCachePool(target, 2, 0, 4, pin_memory=False)
    with pytest.raises(ValueError, match="whole pages"):
        anchor.backup_from_device_all_layer(target, torch.tensor([4]), torch.tensor([8]), "kernel_ascend")


def test_resident_allocator_releases_mapping_and_clears_it():
    from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator

    target = make_pool(2)
    allocator = NPUPagedTokenToKVPoolAllocator(64, 4, torch.bfloat16, "cpu", target, True)
    anchor = NpuResidentHiCachePool(target, 2, 0, 4, pin_memory=False)
    allocated = allocator.alloc(8)
    pages = allocated[::4] // 4
    anchor.binding.update_request_pages(pages)
    allocator.free(allocated)
    with pytest.raises(RuntimeError, match="No logical"):
        anchor.binding.indexer_page_locations(pages)
    allocated = allocator.alloc(8)
    pages = allocated[::4] // 4
    anchor.binding.update_request_pages(pages)
    allocator.clear()
    with pytest.raises(RuntimeError, match="No logical"):
        anchor.binding.indexer_page_locations(pages)


def test_resident_radix_loadback_installs_new_mapping_before_transfer():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    target = make_pool(2)
    anchor = NpuResidentHiCachePool(target, 2, 0, 4, pin_memory=False)
    allocator = SimpleNamespace(offload_backing=None, get_kvcache=lambda: target)
    cache = SimpleNamespace(
        token_to_kv_pool_allocator=allocator, page_size=8,
        root_node=SimpleNamespace(id=0),
        tree_core=SimpleNamespace(
            node_by_id=lambda _: SimpleNamespace(parent=SimpleNamespace(id=0)),
            collect_full_device_indices=lambda *_: torch.empty(0, dtype=torch.int64),
        ),
    )
    plan = UnifiedRadixCache._build_loadback_plan(
        cache, SimpleNamespace(host_indices=rows([4, 2]), nodes_to_load=[10]),
    )
    assert callable(plan)
    plan(rows([7, 1]))
    assert anchor.binding.indexer_page_locations(torch.tensor([7, 1])).tolist() == [28, 30]


def test_real_hybrid_assembler_selects_resident_kv_indexer_and_npu_mamba(monkeypatch):
    from sglang.srt.hardware_backend.npu import hicache_resident_pool, hicache_indexer_pool, hicache_mamba_pool
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.pool_host import mla

    target, draft = make_pool(2), make_pool(1)
    # Select the NPU branch but allocate tiny CPU tensors for this assembly test.
    monkeypatch.setattr(assembler, "_has_npu_compressed_indexer", lambda _: True)
    monkeypatch.setattr(assembler, "get_parallel", lambda: SimpleNamespace(attn_cp_size=1, attn_dcp_size=1, dcp_enabled=False))
    monkeypatch.setattr(assembler, "get_memory", lambda: SimpleNamespace(
        hicache_ratio=2, hicache_size=0, hicache_mem_layout="page_first_kv_split",
        hicache_write_policy="write_through", hicache_io_backend="kernel_ascend", hicache_host_memory_mode="cache",
    ))
    monkeypatch.setattr(assembler, "_get_allocator_type", lambda: "default")
    monkeypatch.setitem(mla.ALLOC_MEMORY_FUNCS, "cpu", lambda dims, **kw: torch.empty(dims, dtype=kw['dtype']))
    for module, name in (
        (hicache_resident_pool, 'NpuResidentHiCachePool'),
        (hicache_indexer_pool, 'NPUIndexerPoolHost'),
        (hicache_mamba_pool, 'NpuMambaPoolHost'),
    ):
        cls = getattr(module, name)
        monkeypatch.setattr(module, name, lambda *args, _cls=cls, **kw: _cls(*args, **(kw | {'pin_memory': False})))
    monkeypatch.setattr(assembler, "HybridCacheController", lambda *a, **kw: SimpleNamespace(**kw))
    mamba = SimpleNamespace(
        device="cpu", size=3, num_mamba_layers=1,
        mamba_cache=SimpleNamespace(
            temporal=torch.zeros(1, 4, 2, 2, 2), conv=[torch.zeros(1, 4, 2, 4)],
        ),
    )
    params = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(mamba_allocator=SimpleNamespace(alloc=lambda *_: None, free=lambda *_: None)),
        mtp_draft_device_pools=(draft,), page_size=8,
        token_to_kv_pool_allocator=SimpleNamespace(), tp_cache_group=None,
        attn_cp_cache_group=None, attn_tp_cache_group=None, pp_cache_group=None,
    )
    group, controller = assembler.build_hybrid_mamba_stack(
        params=params, kv_pool=target, mamba_pool=mamba,
        full_layer_mapping={0: 0, 2: 1}, mamba_layer_mapping={1: 0},
        load_cache_event=None, storage_backend=None, use_mla=True,
    )
    assert isinstance(group.get_pool(PoolName.KV), NpuResidentHiCachePool)
    assert isinstance(group.get_pool(PoolName.INDEXER), NPUIndexerPoolHost)
    mamba_host = group.get_pool(PoolName.MAMBA)
    assert type(mamba_host).__name__ == "NpuMambaPoolHost"
    assert mamba_host.layout == "page_first"
    mamba.mamba_cache.temporal[:, 2].fill_(73)
    mamba_host.backup_from_device_all_layer(mamba, torch.tensor([1]), torch.tensor([2]), 'kernel_ascend')
    mamba.mamba_cache.temporal.fill_(-1)
    mamba_host.load_to_device_per_layer(mamba, torch.tensor([1]), torch.tensor([3]), 0, 'kernel_ascend')
    assert torch.all(mamba.mamba_cache.temporal[:, 3] == 73)


def test_private_resident_pool_submits_through_real_l2_engine(monkeypatch):
    from test_hicache_offload_completion import Device
    from sglang.srt.mem_cache import l2_transfer

    device = Device()
    monkeypatch.setattr(l2_transfer, "device_module", device)
    l2_transfer._timing_events_supported.cache_clear()
    try:
        target, draft = make_pool(2), make_pool(1)
        anchor = NpuResidentHiCachePool(target, 2, 0, 4, pin_memory=False, mtp_draft_device_pools=(draft,))
        indexer = NPUIndexerPoolHost(target, anchor, pin_memory=False)
        target.hicache_indexer_mapping.update_request_pages(torch.tensor([3, 5]))
        target.k_buffer.fill_(11)
        draft.k_buffer.fill_(73)
        engine = l2_transfer.L2TransferEngine("kernel_ascend")
        completion = engine.submit_device_to_host([
            l2_transfer.L2Transfer(anchor, target, rows([4, 2]), rows([3, 5])),
            l2_transfer.L2Transfer(indexer, target, rows([4, 2]), rows([3, 5])),
        ])
        assert not completion.finish_event.query()
        device.progress()
        assert completion.finish_event.query()
        target.k_buffer.fill_(-1)
        draft.k_buffer.fill_(-1)
        completion = engine.submit_host_to_device([
            l2_transfer.L2Transfer(anchor, target, rows([4, 2]), rows([3, 5]), {0: 0, 1: 1}.get),
            l2_transfer.L2Transfer(anchor, draft, rows([4, 2]), rows([3, 5]), {0: 2}.get, True),
        ], layer_num=2)
        device.progress()
        assert completion.finish_event.query()
        assert torch.all(draft.k_buffer[0].view(-1, 8)[rows([3, 5])] == 73)
        assert torch.all(target.k_buffer[0].view(-1, 8)[rows([3, 5])] == 11)
    finally:
        l2_transfer._timing_events_supported.cache_clear()


def test_resident_binding_waits_before_releasing_pages_and_clear():
    from sglang.srt.hardware_backend.npu.hicache_resident_pool import ResidentIndexerBinding

    binding = ResidentIndexerBinding(4, 2, 17)
    binding.update_request_pages(torch.tensor([3, 5]))
    waited = []

    class PendingEvent:
        done = False

        def query(self):
            return self.done

        def synchronize(self):
            assert binding.indexer_page_locations(torch.tensor([3, 5])).tolist() == [12, 14]
            waited.append(True)
            self.done = True

    binding.record_reader(PendingEvent())
    binding.release_device_pages(torch.tensor([3, 5]))
    assert waited == [True]
    binding.update_request_pages(torch.tensor([3, 5]))
    binding.record_reader(PendingEvent())
    binding.clear()
    assert waited == [True, True]
    with pytest.raises(RuntimeError, match="No logical"):
        binding.indexer_page_locations(torch.tensor([3]))
