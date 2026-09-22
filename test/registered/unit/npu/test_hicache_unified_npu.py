"""Real Unified FULL/KDA/Indexer lifecycle over MemFabric on an idle NPU."""

import atexit
import os
import traceback
from array import array

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_HICACHE_TEST_NPU") != "1",
    reason="requires an explicitly selected idle NPU and the matching FSA extension",
)


def exercise_unified_memfabric_lifecycle(dtype, *, shared=None, device_id=None):
    import torch_npu
    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams,
        KimiLinearStateShape,
        Mamba2StateDType,
    )
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        HybridReqToTokenPool,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.runtime_context import restore_context, snapshot_context
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    if device_id is None:
        device_id = int(os.environ.get("SGLANG_FSA_TEST_DEVICE_ID", "0"))
    torch_npu.npu.set_device(device_id)
    device = "npu"
    context = snapshot_context()
    cache = pool = None
    try:
        args = ServerArgs(
            model_path="dummy",
            device="npu",
            page_size=128,
            enable_hierarchical_cache=True,
            hicache_io_backend="kernel_ascend",
            hicache_mem_layout="page_first_kv_split",
            hicache_write_policy="write_through",
            mamba_radix_cache_strategy="extra_buffer",
            tp_size=1 if shared is None else shared.group.world_size,
        )
        args._mamba_cache_chunk_size = 128
        set_global_server_args_for_scheduler(args)
        state_params = KimiLinearCacheParams(
            shape=KimiLinearStateShape.create(
                tp_world_size=1, num_heads=1, head_dim=16, conv_kernel_size=4
            ),
            layers=[1],
            dtype=Mamba2StateDType(conv=dtype, temporal=torch.float32),
        )
        requests = HybridReqToTokenPool(
            size=2,
            mamba_size=8,
            mamba_spec_state_size=2,
            max_context_len=2048,
            device=device,
            enable_memory_saver=False,
            cache_params=state_params,
            mamba_layer_ids=[1],
            enable_mamba_extra_buffer=True,
        )
        pool = NPUMLATokenToKVPool(
            size=2048,
            page_size=128,
            dtype=dtype,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            index_head_dim=128,
            layer_num=2,
            device=device,
            enable_memory_saver=False,
            index_kpool=4,
            index_kpool_compress=True,
            max_running_requests=2,
            prefill_layer_slots=2,
            decode_host_budget=1 << 24,
            integrated_offload=True,
            shared_host_group=shared,
        )
        hybrid = HybridLinearKVPool(
            size=2048,
            page_size=128,
            dtype=dtype,
            head_num=1,
            head_dim=512,
            full_attention_layer_ids=[0, 2],
            device=device,
            mamba_pool=requests.mamba_pool,
            full_kv_pool=pool,
            use_mla=True,
            use_dsa=True,
        )
        allocator = NPUPagedTokenToKVPoolAllocator(
            size=2048,
            page_size=128,
            dtype=dtype,
            device=device,
            kvcache=hybrid,
            need_sort=True,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=requests,
            token_to_kv_pool_allocator=allocator,
            page_size=128,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
            tp_cache_group=None if shared is None else shared.group.cpu_group,
            attn_tp_cache_group=None if shared is None else shared.group.cpu_group,
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
            cache = UnifiedRadixCache(params)
            cache.init_hicache(args, params)
        assert cache.page_size == 512
        cache.write_through_threshold = 1 << 30
        cache.load_back_threshold = 0
        backing = pool.decode_host_backing
        binding = backing.page_binding
        pointers = [regions[0].tensor.data_ptr() for regions in backing.regions]
        held = allocator.alloc(512)
        indices = allocator.alloc(512)
        binding.indexer_mapping.update_request_pages(indices[::128].cpu() // 128)
        host_rows = binding.host_rows(indices).cpu()
        if shared is not None:
            mappings = shared.group.all_gather_object(host_rows.tolist())
            assert all(mapping == mappings[0] for mapping in mappings)
        expected_kv = []
        locations = binding.indexer_page_locations(indices[::128].cpu() // 128)
        index_rows = (locations[:, None] + torch.arange(32)).flatten().to(device)
        expected_index = []
        for layer in range(2):
            values = torch.full((512, 512), layer + 3, dtype=dtype, device=device)
            backing.write_current(layer, indices, values, values[:, :0])
            expected_kv.append(values.cpu())
            index_buffer = pool.index_k_buffer[layer].view(-1, 128)
            index_buffer[index_rows] = layer + 7
            expected_index.append(index_buffer[index_rows].clone())
        state_slot = requests.mamba_allocator.alloc(1)
        state_buffers = [
            requests.mamba_pool.mamba_cache.temporal,
            *requests.mamba_pool.mamba_cache.conv,
        ]
        for marker, buffer in enumerate(state_buffers, 11):
            buffer[:, state_slot] = marker
        expected_state = [buffer[:, state_slot].clone() for buffer in state_buffers]
        binding.ready_event = torch.npu.Event()
        binding.ready_event.record()
        key = RadixKey(array("q", range(512)))
        cache.insert(InsertParams(key=key, value=indices, mamba_value=state_slot))
        node = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        assert (
            cache._execute_and_commit_kv_backup(BackupKV(node_ids=[node]), False) == 512
        )
        torch.npu.synchronize()
        cache.writing_check()
        assert cache.tree_core.is_backuped(node)
        assert backing.hicache_pool is cache.host_pool_group.get_pool(PoolName.KV)
        cache.evict(EvictParams(num_tokens=512, mamba_num=1))
        hit = cache.match_prefix(MatchPrefixParams(key=key))
        assert hit.host_hit_length == 512
        allocator.free(held)
        allocator.merge_and_sort_free()
        pool.index_k_buffer.fill_(-1)
        for buffer in state_buffers:
            buffer.fill_(-2)
        assert cache.load_back(node)
        assert cache.ready_to_load_host_cache() >= 0
        torch.npu.synchronize()
        cache.loading_check()
        restored = cache.tree_core.get_component_device_value(node, ComponentType.FULL)
        assert not torch.equal(restored.cpu(), indices.cpu())
        assert torch.equal(binding.host_rows(restored).cpu(), host_rows)
        locations = binding.indexer_page_locations(restored[::128].cpu() // 128)
        restored_index_rows = (
            (locations[:, None] + torch.arange(32)).flatten().to(device)
        )
        for layer in range(2):
            torch.testing.assert_close(
                pool.index_k_buffer[layer].view(-1, 128)[restored_index_rows],
                expected_index[layer],
            )
            if shared is None:
                actual_kv = backing.regions[layer][0].tensor[host_rows]
            else:
                actual_kv = torch.empty((512, 512), dtype=dtype, device=device)
                backing.pool.copy_from_host(
                    f"decode-{layer}-latent",
                    actual_kv,
                    host_rows.to(device),
                    torch.arange(512, dtype=torch.int64, device=device),
                    torch.ones(512, dtype=torch.bool, device=device),
                )
                actual_kv = actual_kv.cpu()
            torch.testing.assert_close(actual_kv, expected_kv[layer])
        restored_state = cache.tree_core.get_component_device_value(
            node, ComponentType.MAMBA
        )
        for buffer, expected in zip(state_buffers, expected_state):
            torch.testing.assert_close(buffer[:, restored_state], expected)
        assert pointers == [regions[0].tensor.data_ptr() for regions in backing.regions]
        cache.sanity_check()
        cache.evict(EvictParams(num_tokens=512, mamba_num=1))
        cache.evict_host(512)
        cache.evict_host(1, ComponentType.MAMBA)
        torch.npu.synchronize()
        binding.reap()
        binding.reclaim_for_capacity(2048)
        assert not binding.references.any()
        assert backing.hicache_pool.available_size() == 2048
    except Exception:
        traceback.print_exc()
        raise
    finally:
        torch.npu.synchronize()
        if cache is not None:
            atexit.unregister(cache.shutdown)
            cache.release_host_resources()
        if pool is not None:
            pool.decode_host_backing.pool.close(drained=True)
        restore_context(context)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unified_memfabric_backup_prefix_hit_and_fragmented_restore(dtype):
    exercise_unified_memfabric_lifecycle(dtype)


@pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_SHARED_TP") != "1",
    reason="requires explicitly selected idle TP devices",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_shared_unified_memfabric_lifecycle(dtype, request):
    from test_fsa_shared_tp_npu import shared_group

    parallel = request.getfixturevalue("tp")
    exercise_unified_memfabric_lifecycle(
        dtype, shared=shared_group(parallel), device_id=parallel.device_id
    )
