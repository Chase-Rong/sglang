"""Real integrated Host KV reuse across completed requests and shared prefixes."""
import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_ATTENTION") != "1",
    reason="requires an explicitly selected idle NPU",
)


def test_shared_prefix_release_and_reallocated_rows():
    import torch_npu
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.hardware_backend.npu.prefill_overlap import PipelinedIntegratedOffloadRuntime

    device = f"npu:{int(os.environ['SGLANG_FSA_TEST_DEVICE_ID'])}"
    torch.npu.set_device(device)
    pool = NPUMLATokenToKVPool(
        size=1024, page_size=128, dtype=torch.bfloat16, kv_lora_rank=512,
        qk_rope_head_dim=0, index_head_dim=128, layer_num=4, device=device,
        enable_memory_saver=False, start_layer=0, end_layer=4,
        index_kpool=4, index_kpool_compress=True, max_running_requests=2,
        prefill_layer_slots=2, decode_host_budget=1 << 24, integrated_offload=True,
    )
    shared = torch.arange(128, 256, device=device)
    mappings = torch.stack((torch.cat((shared, torch.arange(256, 384, device=device))),
                            torch.cat((shared, torch.arange(384, 512, device=device)))))
    runtime = PipelinedIntegratedOffloadRuntime(pool, SimpleNamespace(req_to_token=mappings))
    pool.prefill_offload_runtime = runtime
    reference = torch.zeros(4, 1152, 512, dtype=torch.bfloat16)
    initialized = set()
    checks = []
    try:
        # Producer; two simultaneous consumers; retained prefix; then recycled
        # rows written with different values and restored by another consumer.
        for slots, prefix, length, value in [([0], 0, 128, 1), ([0, 1], 128, 256, 3),
                                             ([1], 128, 256, 5), ([0], 0, 128, 7),
                                             ([0, 1], 128, 256, 9)]:
            new = torch.cat([mappings[s, prefix:length] for s in slots])
            batch = SimpleNamespace(
                extend_prefix_lens_cpu=[prefix] * len(slots),
                req_pool_indices=torch.tensor(slots, device=device), out_cache_loc=new,
                forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
            )
            with runtime.forward(batch):
                if prefix:
                    assert int(runtime.prefix_plan[:, 1].sum()) == 128
                for layer in range(4):
                    runtime.prepare_layer(layer)
                    slot = pool.get_kv_buffer(layer)[0].view(-1, 512)
                    if prefix:
                        checks.append((slot[shared].clone(), reference[layer, 128:256].clone()))
                    values = torch.cat([
                        torch.full((length-prefix, 1, 512), value + layer + s,
                                   dtype=torch.bfloat16, device=device) for s in slots
                    ])
                    pool.set_kv_buffer(SimpleNamespace(layer_id=layer), new, values,
                                       values.new_empty(len(new), 1, 0))
                    runtime.on_compute_start(layer)
                    runtime.finish_layer(layer)
                    for s in slots:
                        reference[layer, mappings[s, prefix:length].cpu()] = value + layer + s
            for s in slots:
                runtime.release_request(f"request-{value}-{s}")
                initialized.update(mappings[s, prefix:length].cpu().tolist())
            for layer in range(4):
                region = pool.decode_host_backing.regions[layer][0].tensor
                indices = sorted(initialized)
                torch.testing.assert_close(region[indices], reference[layer, indices])
        runtime.drain()
        torch.npu.synchronize()
        for actual, expected in checks:
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    finally:
        runtime.drain()
        torch.npu.synchronize()
        pool.decode_host_backing.pool.close(drained=True)


def test_unified_radix_lru_recycles_target_and_draft_host_rows():
    """Real tree locks and paged allocator drive Host row reuse in both pools."""
    import torch_npu
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.hardware_backend.npu.prefill_overlap import PipelinedIntegratedOffloadRuntime
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    device = f"npu:{int(os.environ['SGLANG_FSA_TEST_DEVICE_ID'])}"
    torch.npu.set_device(device)
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=128))
    requests = ReqToTokenPool(2, 512, device, False)
    pools = [NPUMLATokenToKVPool(
        size=256, page_size=128, dtype=torch.bfloat16, kv_lora_rank=512,
        qk_rope_head_dim=0, index_head_dim=128, layer_num=2, device=device,
        enable_memory_saver=False, start_layer=0, end_layer=2,
        index_kpool=4, index_kpool_compress=True, max_running_requests=2,
        prefill_layer_slots=2, decode_host_budget=1 << 24, integrated_offload=True,
    ) for _ in range(2)]
    runtimes = [PipelinedIntegratedOffloadRuntime(pool, requests) for pool in pools]
    for pool, runtime in zip(pools, runtimes):
        pool.prefill_offload_runtime = runtime
    allocator = PagedTokenToKVPoolAllocator(256, 128, torch.bfloat16, device, pools[0], False)
    cache = UnifiedRadixCache(CacheInitParams(
        disable=False, req_to_token_pool=requests,
        token_to_kv_pool_allocator=allocator, page_size=128,
        tree_components=(ComponentType.FULL,),
    ))
    keys = [RadixKey(list(range(start, start + 128))) for start in (1000, 2000, 3000, 4000)]

    def write(rows, value):
        requests.req_to_token[0, :128] = rows
        batch = SimpleNamespace(
            extend_prefix_lens_cpu=[0], req_pool_indices=torch.tensor([0], device=device),
            out_cache_loc=rows,
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        )
        for role, (pool, runtime) in enumerate(zip(pools, runtimes)):
            with runtime.forward(batch):
                for layer in range(2):
                    runtime.prepare_layer(layer)
                    data = torch.full((128, 1, 512), value + role * 20 + layer,
                                      device=device, dtype=torch.bfloat16)
                    pool.set_kv_buffer(SimpleNamespace(layer_id=layer), rows, data, data.new_empty(128, 1, 0))
                    runtime.on_compute_start(layer)
                    runtime.finish_layer(layer)
            runtime.release_request(f"producer-{value}")

    def read(key, value):
        hit = cache.match_prefix(MatchPrefixParams(key))
        assert hit.device_indices.numel() == 128
        requests.req_to_token[0, :128] = hit.device_indices
        batch = SimpleNamespace(
            extend_prefix_lens_cpu=[128], req_pool_indices=torch.tensor([0], device=device),
            out_cache_loc=torch.empty(0, dtype=torch.int64, device=device),
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        )
        actual = []
        for role, (pool, runtime) in enumerate(zip(pools, runtimes)):
            with runtime.forward(batch):
                for layer in range(2):
                    runtime.prepare_layer(layer)
                    actual.append((pool.get_kv_buffer(layer)[0].view(-1, 512)[hit.device_indices].clone(),
                                   value + role * 20 + layer))
                    runtime.on_compute_start(layer)
                    runtime.finish_layer(layer)
        torch.npu.synchronize()
        for data, expected in actual:
            torch.testing.assert_close(data.cpu(), torch.full_like(data.cpu(), expected), rtol=0, atol=0)
        return hit

    try:
        rows_a = allocator.alloc(128)
        write(rows_a, 1)
        cache.insert(InsertParams(key=keys[0], value=rows_a))
        a = read(keys[0], 1)
        locks = [cache.inc_lock_ref(a.last_device_node) for _ in range(2)]
        rows_b = allocator.alloc(128)
        write(rows_b, 3)
        cache.insert(InsertParams(key=keys[1], value=rows_b))
        assert allocator.available_size() == 0
        cache.dec_lock_ref(a.last_device_node, locks[0].to_dec_params())
        assert cache.evict(EvictParams(num_tokens=256)).num_tokens_evicted == 128
        assert cache.match_prefix(MatchPrefixParams(keys[1])).device_indices.numel() == 0
        read(keys[0], 1)
        rows_c = allocator.alloc(128)
        assert torch.equal(rows_c, rows_b)
        write(rows_c, 7)
        cache.insert(InsertParams(key=keys[2], value=rows_c))
        read(keys[2], 7)
        cache.dec_lock_ref(a.last_device_node, locks[1].to_dec_params())
        assert cache.evict(EvictParams(num_tokens=128)).num_tokens_evicted == 128
        assert cache.match_prefix(MatchPrefixParams(keys[0])).device_indices.numel() == 0
        rows_d = allocator.alloc(128)
        assert torch.equal(rows_d, rows_a)
        write(rows_d, 9)
        cache.insert(InsertParams(key=keys[3], value=rows_d))
        read(keys[3], 9)
        read(keys[2], 7)
    finally:
        for runtime in runtimes:
            runtime.drain()
        torch.npu.synchronize()
        for pool in pools:
            pool.decode_host_backing.pool.close(drained=True)
