"""Real NPU pool and Host copy lifecycle; transport ACK is injected locally."""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_ATTENTION") != "1",
    reason="requires explicitly selected idle NPU and native helpers",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_integrated_prefill_decode_share_host_and_reuse_slots(dtype):
    exercise_integrated_slots(dtype, int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"]))


def exercise_integrated_slots(dtype, device_id, shared_group=None):
    import torch_npu
    from sglang.srt.hardware_backend.npu.integrated_offload import IntegratedOffloadRuntime
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    pool = NPUMLATokenToKVPool(
        size=512, page_size=128, dtype=dtype, kv_lora_rank=512,
        qk_rope_head_dim=0, index_head_dim=128, layer_num=4, device=device,
        enable_memory_saver=False, start_layer=0, end_layer=4,
        index_kpool=4, index_kpool_compress=True, max_running_requests=2,
        prefill_layer_slots=2, decode_host_budget=1 << 24, integrated_offload=True,
        shared_host_group=shared_group,
    )
    request_pool = SimpleNamespace(req_to_token=torch.arange(128, 384, device=device).view(1, -1))
    runtime = IntegratedOffloadRuntime(pool, request_pool)
    pool.prefill_offload_runtime = runtime
    assert pool.k_buffer.shape[0] == 2 and pool.index_k_buffer.shape[0] == 4
    graph = None
    try:
        for prefix, length in ((0, 128), (128, 192)):
            batch = SimpleNamespace(
                extend_prefix_lens_cpu=[prefix], req_pool_indices=torch.tensor([0], device=device),
                forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
            )
            with runtime.forward(batch):
                for layer_id in range(4):
                    key, rope = pool.get_kv_buffer(layer_id)
                    if prefix:
                        torch.testing.assert_close(key.view(-1, 512)[128:256].cpu(),
                                                   torch.full((128, 512), layer_id + 1, dtype=dtype))
                    values = torch.full((length - prefix, 1, 512), layer_id + 1, dtype=dtype, device=device)
                    pool.set_kv_buffer(SimpleNamespace(layer_id=layer_id),
                                       request_pool.req_to_token[0, prefix:length], values,
                                       values.new_empty(length - prefix, 1, 0))
                    torch.testing.assert_close(key.view(-1, 512)[128:128 + length].cpu(),
                                               torch.full((length, 512), layer_id + 1, dtype=dtype))
                    runtime.finish_layer(layer_id)
            torch.npu.synchronize()
            for layer_id in range(4):
                key, _ = pool.get_kv_buffer(layer_id)
                assert key.device.type == "cpu"
                torch.testing.assert_close(key.view(-1, 512)[128:128 + length],
                                           torch.full((length, 512), layer_id + 1, dtype=dtype))
        values = torch.full((1, 1, 512), 7, dtype=dtype, device=device)
        location = torch.tensor([320], dtype=torch.int64, device=device)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            pool.set_kv_buffer(SimpleNamespace(layer_id=3), location, values, values.new_empty(1, 1, 0))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(pool.get_kv_buffer(3)[0].view(-1, 512)[320], values[0, 0].cpu())
        batch.extend_prefix_lens_cpu = [193]
        with runtime.forward(batch):
            for layer_id in range(4):
                key, _ = pool.get_kv_buffer(layer_id)
                if layer_id == 3:
                    torch.testing.assert_close(key.view(-1, 512)[320].cpu(), values[0, 0].cpu())
                runtime.finish_layer(layer_id)
        assert not runtime.active and not runtime.failed
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        pool.decode_host_backing.pool.close(drained=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_real_prefill_slots_restore_prefix_and_keep_kpool_resident(dtype):
    exercise_prefill_slots(dtype, int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"]))


def exercise_prefill_slots(dtype, device_id, shared_group=None):
    import torch_npu
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.hardware_backend.npu.prefill_offload_adapter import NpuPrefillOffloadAdapter
    from sglang.srt.hardware_backend.npu.prefill_offload_runtime import NpuPrefillOffloadRuntime
    from sglang.srt.hardware_backend.npu.registered_host_kv import NpuRegisteredHostKvPool

    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    pool = NPUMLATokenToKVPool(
        size=512, page_size=128, dtype=dtype, kv_lora_rank=512,
        qk_rope_head_dim=0, index_head_dim=128, layer_num=4, device=device,
        enable_memory_saver=False, start_layer=0, end_layer=4,
        index_kpool=4, index_kpool_compress=True, max_running_requests=3,
        prefill_layer_slots=2,
    )
    assert pool.k_buffer.shape[0] == 2
    assert pool.index_k_buffer.shape[0] == 4
    request_pool = SimpleNamespace(req_to_token=torch.zeros(3, 256, dtype=torch.int64, device=device))
    request_pool.req_to_token[1] = torch.arange(128, 384, device=device)
    host = NpuRegisteredHostKvPool(device_id, shared_group=shared_group)
    adapter = NpuPrefillOffloadAdapter(host, page_size=128)
    publications = []

    def publish(sender, work, binding, event):
        assert event.query()
        publications.append((work.token.logical_layer, work.valid_range.start_token, work.valid_range.end_token))
        return [SimpleNamespace(state="acknowledged")]

    runtime = NpuPrefillOffloadRuntime(pool, request_pool, adapter, publish=publish)
    runtime.register_host_layers()
    pool.prefill_offload_runtime = runtime
    sender = SimpleNamespace(require_prefill_layer_targets=lambda layers: None,
                             prefill_layer_operations=lambda: ())
    try:
        for prefix, length in ((0, 128), (128, 192)):
            batch = SimpleNamespace(
                rids=["request"], seq_lens_cpu=torch.tensor([length], device="cpu"),
                extend_prefix_lens_cpu=[prefix], req_pool_indices=torch.tensor([1], device=device),
                prefill_offload_senders=[sender],
                forward_mode=SimpleNamespace(is_extend=lambda: True, is_target_verify=lambda: False,
                                             is_draft_extend_v2=lambda: False),
            )
            with runtime.forward(batch):
                for layer in range(4):
                    indexer = pool.get_index_k_buffer(layer)
                    if prefix:
                        torch.testing.assert_close(indexer.cpu(), torch.full(indexer.shape, layer + 10, dtype=dtype))
                    else:
                        indexer.fill_(layer + 10)
                    latent, rope = pool.get_kv_buffer(layer)
                    assert rope.numel() == 0
                    rows = request_pool.req_to_token[1, :length]
                    if prefix:
                        torch.testing.assert_close(latent.view(-1, 512)[rows[:prefix]].cpu(),
                                                   torch.full((prefix, 512), layer + 1, dtype=dtype))
                    values = torch.full((length - prefix, 1, 512), layer + 1, dtype=dtype, device=device)
                    pool.set_kv_buffer(SimpleNamespace(layer_id=layer), rows[prefix:], values,
                                       values.new_empty(length - prefix, 1, 0))
                    runtime.finish_layer(layer)
            assert runtime.slots.active_tokens() == ()
        assert len(publications) == 8
        runtime.release_request("request")
        assert runtime.host_pages.free_page_count == runtime.host_pages.page_count
    finally:
        torch.npu.synchronize()
        host.close(drained=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_real_decode_current_write_graph_and_large_staging_copy(dtype):
    import torch_npu
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.hardware_backend.npu.layer_staging import create_npu_layer_staging
    from sglang.srt.hardware_backend.npu.prefill_offload import KvComponent

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    pool = NPUMLATokenToKVPool(
        size=512, page_size=128, dtype=dtype, kv_lora_rank=512,
        qk_rope_head_dim=0, index_head_dim=128, layer_num=2, device=device,
        enable_memory_saver=False, start_layer=0, end_layer=2,
        index_kpool=4, index_kpool_compress=True, max_running_requests=3,
        decode_host_budget=1 << 24,
    )
    graph = None
    try:
        key, rope = pool.get_kv_buffer(1)
        assert key.device.type == "cpu" and rope.numel() == 0
        assert pool.get_index_k_buffer(1).device.type == "npu"
        region = pool.decode_host_backing.regions[1][0]
        region.tensor.zero_()
        locations = torch.tensor([129, 131, 0], dtype=torch.int64, device=device)
        values = torch.randn(3, 1, 512, dtype=dtype, device=device)
        layer = SimpleNamespace(layer_id=1)

        def write():
            pool.set_kv_buffer(layer, locations, values, values.new_empty(3, 1, 0))

        write()
        torch.npu.synchronize()
        torch.testing.assert_close(region.tensor[[129, 131]], values[:2, 0].cpu(), rtol=0, atol=0)
        assert torch.count_nonzero(region.tensor[0]).item() == 0
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            write()
        values.add_(1)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(region.tensor[[129, 131]], values[:2, 0].cpu(), rtol=0, atol=0)
        lease = pool.layer_receive_lease(7, "peer", [1], [1])
        assert lease.receive_components() == {KvComponent.LATENT_K}
        staging = create_npu_layer_staging(65536, device_id)
        payload = torch.randint(0, 255, (65536,), dtype=torch.uint8, device=device)
        staging.owner[0].copy_(payload)
        torch.npu.synchronize()
        destination = lease.layers[1][KvComponent.LATENT_K]
        offset = 128 * 512 * 2
        event = staging.copy(destination, offset, payload.numel())
        event.synchronize()
        torch.testing.assert_close(region.tensor.view(torch.uint8).flatten()[offset:offset + 65536],
                                   payload.cpu(), rtol=0, atol=0)
        staging.drain()
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        pool.decode_host_backing.pool.close(drained=True)
