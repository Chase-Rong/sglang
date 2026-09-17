"""Production allocator and stream-ordered DMA, with no lab copy adapters."""

import os

import pytest
import torch

from test_fsa_shared_tp_npu import tp as tp

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_UNIFIED_HOST_TEST") != "1",
    reason="requires explicitly selected idle devices and the MemFabric A5 build",
)


def exercise_dma(device_id, shared=None):
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops
    from sglang.srt.hardware_backend.npu.memfabric_host_kv import MemFabricHostKvPool

    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    pool = MemFabricHostKvPool(
        device_id, required_bytes=256 * 512 * 2, shared_group=shared
    )
    region = pool.allocate_component(
        "kv", (256, 512), torch.bfloat16, page_size=128
    ).region
    plan_rows = torch.tensor([128, 129, 130, 7, 8, 18, 0, -1, 256], device=device)
    plan, _ = pool.plan_prefill_rows(plan_rows, plan_rows[:0], 256)
    valid = torch.tensor([128, 129, 130, 7, 8, 18])
    pattern = (torch.arange(256 * 512).view(256, 512) % 97).to(torch.bfloat16) / 128
    source = pattern.to(device)
    restored = torch.zeros_like(source)
    copy_stream = torch.npu.Stream(device=device)
    compute_stream = torch.npu.current_stream()
    graph = None
    try:
        if shared is not None:
            sizes = shared.group.all_gather_object(pool.physical_bytes)
            assert sizes[0] == 1 << 30 and not any(sizes[1:])
        with pytest.raises(RuntimeError, match="drain"):
            pool.close()
        for shift in (0, 1, 2):
            source.copy_((pattern + shift).to(device))
            ready = torch.npu.Event()
            ready.record(compute_stream)
            with torch.npu.stream(copy_stream):
                copy_stream.wait_event(ready)
                pool.copy_prefill_to_host("kv", source, plan)
                if shared is not None:
                    shared.publish_write()
                saved = torch.npu.Event()
                saved.record(copy_stream)
            compute_stream.wait_event(saved)
            pool.copy_prefill_from_host("kv", restored, plan)
            torch.npu.synchronize()
            torch.testing.assert_close(
                restored.cpu()[valid], (pattern + shift)[valid], atol=0, rtol=0
            )
            if shared is None or shared.is_writer:
                torch.testing.assert_close(
                    region.tensor[valid], (pattern + shift)[valid], atol=0, rtol=0
                )
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            pool.copy_prefill_to_host("kv", source, plan)
            if shared is not None:
                shared.publish_write()
            pool.copy_prefill_from_host("kv", restored, plan)
        source.copy_((pattern + 3).to(device))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            restored.cpu()[valid], (pattern + 3)[valid], atol=0, rtol=0
        )
        if shared is not None:
            result = restored.float()
            torch.distributed.all_reduce(result, group=shared.group.device_group)
            torch.npu.synchronize()
            torch.testing.assert_close(
                result.cpu()[valid],
                (pattern + 3)[valid].float() * shared.group.world_size,
            )
        for bad in (
            torch.tensor([[256, 1]]),
            torch.tensor([[1, 256]]),
            torch.tensor([[0, 1]]),
        ):
            with pytest.raises(RuntimeError, match="outside"):
                pool.copy_prefill_from_host("kv", restored, bad)
        with pytest.raises(RuntimeError, match="live MemFabric"):
            native_ops().prefill_dma_copy(
                restored, region.tensor, region.device_ptr + 1, plan, False
            )
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        pool.close(drained=True)


def test_local_production_dma():
    import torch_npu

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch_npu.npu.set_device(device_id)
    for _ in range(2):
        exercise_dma(device_id)


@pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_SHARED_TP") != "1", reason="requires torchrun"
)
def test_shared_production_dma(tp):
    from sglang.srt.hardware_backend.npu.shared_host_kv import SharedHostGroup

    exercise_dma(tp.device_id, SharedHostGroup(tp.group, tp.device_id))
