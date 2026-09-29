"""CPU regression for packed MTP construction with Ascend split KV buffers."""

import pytest
import torch
from test_prefill_offload_runtime import production_pool_type

from sglang.srt.mem_cache.pool_host import mla


@pytest.fixture
def resident_pool(monkeypatch):
    cls = production_pool_type()

    def make(layers):
        pool = cls(
            size=16, page_size=4, dtype=torch.bfloat16, kv_lora_rank=8,
            qk_rope_head_dim=0, layer_num=layers, device="cpu",
            enable_memory_saver=False, index_head_dim=4, index_kpool=2,
            index_kpool_compress=True, max_running_requests=2,
            start_layer=0, end_layer=layers,
        )
        pool.layer_shard_enabled = False
        return pool

    monkeypatch.setitem(
        mla.ALLOC_MEMORY_FUNCS, "cpu",
        lambda dims, **kw: torch.zeros(dims, dtype=kw["dtype"]),
    )
    return make


@pytest.mark.parametrize("with_draft", [False, True])
def test_split_host_constructs_without_cuda_device_pointers(resident_pool, with_draft):
    target = resident_pool(2)
    drafts = (resident_pool(1),) if with_draft else ()
    assert not hasattr(target, "data_ptrs")
    host = mla.MLATokenToKVPoolHost(
        target, 2, 0, 4, "page_first_kv_split",
        pin_memory=False, mtp_draft_device_pools=drafts,
    )
    layers = 2 + len(drafts)
    assert host.layer_num == layers
    assert host.k_buffer.shape == (host.page_num, layers, 4, 1, 8)
    assert host.index_k_buffer.shape == (host.page_num, layers, 4, 1, 4)
    assert host.k_buffer.nbytes == host.size * layers * 8 * 2
    assert host.mtp_draft_device_pools == drafts


def test_generic_packed_host_retains_target_and_draft_pointers(resident_pool):
    target, draft = resident_pool(2), resident_pool(1)
    for pool in (target, draft):
        pool.kv_buffer = [
            torch.zeros(pool.size + pool.page_size, 1, pool.kv_lora_rank,
                        dtype=pool.dtype)
            for _ in range(pool.layer_num)
        ]
        pool.data_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in pool.kv_buffer], dtype=torch.uint64,
        )
    host = mla.MLATokenToKVPoolHost(
        target, 2, 0, 4, "layer_first", pin_memory=False,
        mtp_draft_device_pools=(draft,),
    )
    assert host.packed_device_data_ptrs.tolist() == (
        target.data_ptrs.tolist() + draft.data_ptrs.tolist()
    )
    assert len(host.packed_device_kv_buffers) == 3
    assert host.packed_device_kv_buffers[-1] is draft.kv_buffer[0]
