"""CPU regression for packed MTP construction with Ascend split KV buffers."""

import pytest
import torch
from test_prefill_offload_runtime import production_pool_type

from sglang.srt.mem_cache.pool_host import mla
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture
def resident_pool(monkeypatch):
    cls = production_pool_type()

    def make(layers):
        pool = cls(
            size=16,
            page_size=4,
            dtype=torch.bfloat16,
            kv_lora_rank=8,
            qk_rope_head_dim=0,
            layer_num=layers,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=4,
            index_kpool=2,
            index_kpool_compress=True,
            max_running_requests=2,
            start_layer=0,
            end_layer=layers,
        )
        pool.layer_shard_enabled = False
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        pool.get_device_pool_infos = lambda: MLATokenToKVPool.get_device_pool_infos(
            pool
        )
        return pool

    monkeypatch.setitem(
        mla.ALLOC_MEMORY_FUNCS,
        "cpu",
        lambda dims, **kw: torch.zeros(dims, dtype=kw["dtype"]),
    )
    return make


@pytest.mark.parametrize("with_draft", [False, True])
def test_split_host_constructs_without_cuda_device_pointers(resident_pool, with_draft):
    target = resident_pool(2)
    drafts = (resident_pool(1),) if with_draft else ()
    assert not hasattr(target, "data_ptrs")
    target_info = target.get_device_pool_infos()[0].buffer_info
    draft_infos = tuple(pool.get_device_pool_infos()[0].buffer_info for pool in drafts)
    host = mla.MLATokenToKVPoolHost(
        target_info,
        2,
        0,
        4,
        "page_first_kv_split",
        pin_memory=False,
        draft_buffer_infos=draft_infos,
        device_capacity_tokens=target.size,
        start_layer=target.start_layer,
        end_layer=target.end_layer,
        target_layer_num=target.layer_num,
    )
    layers = 2 + len(drafts)
    assert host.layer_num == layers
    assert host.k_buffer.shape == (host.page_num, layers, 4, 1, 8)
    assert target_info.buffers.index_layer_indices == (0, 1)
    assert host.index_k_buffer.shape == (host.page_num, 2, 4, 1, 4)
    assert host.k_buffer.nbytes == host.size * layers * 8 * 2
    assert host._native_infos[0] is target_info
    assert host._native_infos[1:] == draft_infos
    for draft_info in draft_infos:
        assert draft_info.buffers.index_layer_indices == (0,)
        assert draft_info.buffers.index_k.shape[0] == 1


def test_resident_indexer_restores_declared_target_and_draft_layers(resident_pool):
    from sglang.srt.hardware_backend.npu.hicache_indexer_pool import (
        NPUIndexerPoolHost,
        make_npu_indexer_pool_decl,
    )
    from sglang.srt.hardware_backend.npu.hicache_resident_pool import (
        NpuResidentHiCachePool,
    )

    target, draft = resident_pool(2), resident_pool(1)
    target_decl = make_npu_indexer_pool_decl(target)
    draft_decl = make_npu_indexer_pool_decl(draft)
    assert target_decl.owned_device_layers == (0, 1)
    assert draft_decl.owned_device_layers == (0,)
    assert target_decl.storage_info == draft_decl.storage_info
    anchor = NpuResidentHiCachePool(
        target,
        2,
        0,
        4,
        pin_memory=False,
        mtp_draft_device_pools=(draft,),
    )
    indexer = NPUIndexerPoolHost(target, anchor, pin_memory=False)
    assert indexer.mtp_draft_device_pools == (draft,)
    assert indexer.target_layer_num == 2
    assert indexer.layer_num == 3
    assert indexer.kv_buffer.shape == (3, anchor.page_num, 16)
    expected = []
    for offset, pool in enumerate((target, draft), start=1):
        buffer = pool.get_device_pool_infos()[0].buffer_info.buffers.index_k
        buffer.copy_(
            torch.arange(buffer.numel()).reshape_as(buffer) % 53 + offset * 100
        )
        expected.extend(layer.view(-1, 4)[4:8].clone() for layer in buffer)
    host_rows = torch.arange(4, 12, dtype=torch.int64)
    source_rows = torch.tensor([4, 5, 6, 7, 12, 13, 14, 15])
    destination_rows = torch.tensor([8, 9, 10, 11, 16, 17, 18, 19])
    anchor.binding.update_request_pages(torch.tensor([1, 3]))
    indexer.backup_from_device_all_layer(
        target, host_rows, source_rows, "kernel_ascend"
    )
    for pool in (target, draft):
        pool.index_k_buffer.fill_(-1)
    anchor.binding.release_device_pages(torch.tensor([1, 3]))
    anchor.binding.prepare_restore(torch.empty(0, dtype=torch.int64), destination_rows)
    for pool, first_layer, is_draft in ((target, 0, False), (draft, 2, True)):
        for local_layer in range(pool.layer_num):
            indexer.load_to_device_per_layer(
                pool,
                host_rows,
                destination_rows,
                first_layer + local_layer,
                "kernel_ascend",
                is_draft=is_draft,
            )
            restored = pool.index_k_buffer[local_layer].view(-1, 4)
            torch.testing.assert_close(
                restored[8:12], expected[first_layer + local_layer]
            )
            assert torch.all(restored[:8] == -1)


def test_generic_packed_host_retains_target_and_draft_pointers(resident_pool):
    target, draft = resident_pool(2), resident_pool(1)
    for pool in (target, draft):
        pool.kv_buffer = [
            torch.zeros(
                pool.size + pool.page_size, 1, pool.kv_lora_rank, dtype=pool.dtype
            )
            for _ in range(pool.layer_num)
        ]
        pool.data_ptrs = torch.tensor(
            [buffer.data_ptr() for buffer in pool.kv_buffer],
            dtype=torch.uint64,
        )
    import msgspec

    infos = tuple(
        msgspec.structs.replace(
            pool.get_device_pool_infos()[0].buffer_info,
            buffers=tuple(pool.kv_buffer),
            data_ptrs=pool.data_ptrs,
            native_buffers=None,
        )
        for pool in (target, draft)
    )
    host = mla.MLATokenToKVPoolHost(
        infos[0],
        2,
        0,
        4,
        "layer_first",
        pin_memory=False,
        draft_buffer_infos=infos[1:],
        device_capacity_tokens=target.size,
        start_layer=target.start_layer,
        end_layer=target.end_layer,
        target_layer_num=target.layer_num,
    )
    assert host.device_data_ptrs.tolist() == (
        target.data_ptrs.tolist() + draft.data_ptrs.tolist()
    )
    assert len(host.device_kv_buffers) == 3
    assert host.device_kv_buffers[-1] is draft.kv_buffer[0]


if __name__ == "__main__":
    run_pytest(__file__)
