"""Real UnifiedRadix/KDA constructors on CPU; no HiCache IO or NPU execution.

CPU MHA storage supplies physical page geometry, with compressed-indexer
metadata attached for the admission policy. The mapping helper, allocator,
radix tree and KDA checkpoint component are real implementations.
"""

from array import array
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("resource", reason="SGLang runtime requires POSIX resource")

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2StateDType,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.hicache_indexer_mapping import IndexerPageMapping
from sglang.srt.mem_cache import memory_pool
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType, MambaComponent
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import (
    mamba_checkpoint_grid,
    mamba_track_grid,
    restore_context,
    snapshot_context,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


class CpuIndexerAllocator(PagedTokenToKVPoolAllocator):
    def _release_page_ids(self, *page_ids):
        if page_ids:
            self.offload_backing.page_binding.indexer_mapping.release_device_pages(
                torch.cat(page_ids)
            )
        super()._release_page_ids(*page_ids)


@pytest.fixture(params=["offload", "resident"])
def grouped_cache(monkeypatch, request):
    context = snapshot_context()
    monkeypatch.setattr(memory_pool, "_cpu_has_amx_support", False)
    physical_page = 128
    compression = 4
    group_size = physical_page * compression
    capacity = 4 * group_size
    try:
        args = ServerArgs(model_path="dummy", device="cpu", page_size=physical_page)
        args._mamba_cache_chunk_size = physical_page
        set_global_server_args_for_scheduler(args)
        state_params = KimiLinearCacheParams(
            shape=KimiLinearStateShape.create(
                tp_world_size=1, num_heads=1, head_dim=2, conv_kernel_size=2
            ),
            layers=[1],
            dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
        )
        requests = HybridReqToTokenPool(
            size=2,
            mamba_size=8,
            mamba_spec_state_size=2,
            max_context_len=capacity,
            device="cpu",
            enable_memory_saver=False,
            cache_params=state_params,
            mamba_layer_ids=[1],
            enable_mamba_extra_buffer=True,
        )
        full_pool = MHATokenToKVPool(
            size=capacity,
            page_size=physical_page,
            dtype=torch.float32,
            head_num=1,
            head_dim=2,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        full_pool._kpool_use_compress = True
        full_pool.index_kpool = compression
        hybrid_pool = HybridLinearKVPool(
            size=capacity,
            page_size=physical_page,
            dtype=torch.float32,
            head_num=1,
            head_dim=2,
            full_attention_layer_ids=[0, 2],
            device="cpu",
            mamba_pool=requests.mamba_pool,
            full_kv_pool=full_pool,
        )
        offloaded = request.param == "offload"
        allocator_cls = (
            CpuIndexerAllocator if offloaded else PagedTokenToKVPoolAllocator
        )
        allocator = allocator_cls(
            size=capacity,
            page_size=physical_page,
            dtype=torch.float32,
            device="cpu",
            kvcache=hybrid_pool,
            need_sort=True,
        )
        mapping = None
        if offloaded:
            mapping = IndexerPageMapping(
                physical_page, compression, capacity // physical_page + 1
            )
            allocator.offload_backing = SimpleNamespace(
                hicache_pool=SimpleNamespace(
                    device_pool=full_pool, page_size=physical_page
                ),
                page_binding=SimpleNamespace(indexer_mapping=mapping),
            )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=requests,
            token_to_kv_pool_allocator=allocator,
            page_size=physical_page,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
            cache = UnifiedRadixCache(params)
        yield SimpleNamespace(
            cache=cache,
            allocator=allocator,
            requests=requests,
            mapping=mapping,
            params=params,
            full_pool=full_pool,
        )
    finally:
        restore_context(context)


def test_real_unified_kda_uses_group_grid_without_resizing_physical_pool(grouped_cache):
    fixture = grouped_cache
    cache = fixture.cache
    assert type(cache) is UnifiedRadixCache
    assert cache.page_size == 512
    assert fixture.params.page_size == fixture.allocator.page_size == 128
    assert fixture.full_pool.page_size == 128
    component = cache.components[ComponentType.MAMBA]
    assert type(component) is MambaComponent
    assert (
        component.mamba_checkpoint_grid == mamba_checkpoint_grid(cache.page_size) == 512
    )
    assert mamba_track_grid(cache.page_size) % 512 == 0


def test_real_unified_partial_group_branch_match_split_and_evict(grouped_cache):
    fixture = grouped_cache
    cache, allocator, mapping = fixture.cache, fixture.allocator, fixture.mapping
    initial_capacity = allocator.available_size()
    original_rows = allocator.alloc(1024)
    if mapping is not None:
        mapping.update_request_pages(original_rows[::128] // 128)
    states = fixture.requests.mamba_allocator.alloc(3)
    original_tokens = array("q", range(1024))
    cache.insert(
        InsertParams(
            key=RadixKey(original_tokens[:512]),
            value=original_rows[:512],
            mamba_value=states[:1],
        )
    )
    cache.insert(
        InsertParams(
            key=RadixKey(original_tokens),
            value=original_rows,
            mamba_value=states[1:2],
            prev_prefix_len=512,
        )
    )
    branch_tokens = array("q", original_tokens)
    branch_tokens[700] = -1
    match = cache.match_prefix(MatchPrefixParams(key=RadixKey(branch_tokens)))
    assert match.full_kv_hit_length == match.device_indices.numel() == 512
    assert torch.equal(match.device_indices, original_rows[:512])
    assert (
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(original_tokens[:700]))
        ).device_indices.numel()
        == 512
    )
    branch_tail = allocator.alloc(512)
    branch_rows = torch.cat((match.device_indices, branch_tail))
    if mapping is not None:
        mapping.update_request_pages(branch_rows[::128] // 128)
    cache.insert(
        InsertParams(
            key=RadixKey(branch_tokens),
            value=branch_rows,
            mamba_value=states[2:3],
            prev_prefix_len=512,
        )
    )
    assert torch.equal(
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(original_tokens))
        ).device_indices,
        original_rows,
    )
    assert torch.equal(
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(branch_tokens))
        ).device_indices,
        branch_rows,
    )
    cache.evict(EvictParams(num_tokens=1536, mamba_num=3))
    assert allocator.available_size() == initial_capacity
    if mapping is not None:
        with pytest.raises(RuntimeError, match="No logical"):
            mapping.indexer_page_locations(original_rows[::128] // 128)
    cache.sanity_check()


def test_grouped_kda_rejects_missing_checkpoint_buffers(grouped_cache):
    params = replace(grouped_cache.params, enable_mamba_extra_buffer=False)
    with pytest.raises(ValueError, match="requires Mamba checkpoint buffers"):
        UnifiedRadixCache(params)


@pytest.mark.parametrize("grouped_cache", ["resident"], indirect=True)
@pytest.mark.parametrize("cuda_compressed", [False, True])
def test_resident_non_npu_indexer_keeps_physical_prefix_boundaries(
    grouped_cache, monkeypatch, cuda_compressed
):
    fixture = grouped_cache
    # CUDA exposes kpool_use_compress (without the NPU-specific leading '_').
    # Both an ordinary pool and a CUDA compressed pool retain their existing grid.
    monkeypatch.delattr(fixture.full_pool, "_kpool_use_compress")
    monkeypatch.setattr(
        fixture.full_pool, "kpool_use_compress", cuda_compressed, raising=False
    )
    params = replace(fixture.params, tree_components=(ComponentType.FULL,))
    with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
        cache = UnifiedRadixCache(params)
    original_rows = fixture.allocator.alloc(512)
    original_tokens = array("q", range(512))
    cache.insert(InsertParams(key=RadixKey(original_tokens), value=original_rows))
    branch_tokens = array("q", original_tokens)
    branch_tokens[200] = -1
    match = cache.match_prefix(MatchPrefixParams(key=RadixKey(branch_tokens)))
    assert match.device_indices.numel() == 128
    assert torch.equal(match.device_indices, original_rows[:128])
    cache.evict(EvictParams(num_tokens=512))
    cache.sanity_check()
