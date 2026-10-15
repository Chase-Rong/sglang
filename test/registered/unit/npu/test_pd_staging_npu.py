"""Validate PD staging DMA against the production MemFabric Host allocation."""

import os
from types import SimpleNamespace

import pytest
import torch

from test_fsa_shared_tp_npu import tp as tp

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_UNIFIED_HOST_TEST") != "1",
    reason="requires explicitly selected idle devices and the MemFabric A5 build",
)


def exercise_staging(device_id, shared=None):
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops
    from sglang.srt.hardware_backend.npu.layer_staging import create_npu_layer_staging
    from sglang.srt.hardware_backend.npu.memfabric_host_kv import MemFabricHostKvPool
    from sglang.srt.hardware_backend.npu.prefill_offload import PdSourceKind

    torch.npu.set_device(device_id)
    pool = MemFabricHostKvPool(
        device_id, required_bytes=4 * 1024 * 1024, shared_group=shared
    )
    receiver = None
    try:
        host = pool.allocate_component(
            "pd-staging", (4096, 512), torch.bfloat16, page_size=128
        ).region
        receiver = create_npu_layer_staging(4096, device_id)
        buffer = receiver.owner[0]
        pattern = (torch.arange(4096, dtype=torch.int32) % 251).to(torch.uint8)
        buffer.copy_(pattern.to(buffer.device))
        region = SimpleNamespace(kind=PdSourceKind.HOST_REGISTERED, owner=host)
        entries = (
            (region, 128 * 1024, 128, 0),
            (region, 2 * 1024 * 1024 - 512, 1024, 128),
            (region, 2 * 1024 * 1024 + 8192, 1024, 1152),
        )
        if shared is None or shared.is_writer:
            receiver.copy_batch(entries).synchronize()
        if shared is not None:
            shared.publish_write()
        output = torch.zeros_like(buffer)
        plan = torch.tensor(
            [(destination, source, count) for _, destination, count, source in entries],
            dtype=torch.int64,
        )
        native_ops().prefill_dma_copy(
            output.view(-1, 1),
            host.tensor.view(torch.uint8).view(-1, 1),
            host.device_ptr,
            plan,
            False,
        )
        torch.npu.synchronize()
        actual = output.cpu()
        for _, _, count, source in entries:
            torch.testing.assert_close(
                actual[source : source + count], pattern[source : source + count]
            )
    finally:
        if receiver is not None:
            receiver.drain()
        torch.npu.synchronize()
        pool.close(drained=True)


def test_local_pd_staging_dma():
    pytest.importorskip("torch_npu")

    torch.npu.set_device(0)
    exercise_staging(0)


@pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_SHARED_TP") != "1", reason="requires torchrun"
)
def test_shared_pd_staging_dma(tp):
    from sglang.srt.hardware_backend.npu.shared_host_kv import SharedHostGroup

    exercise_staging(tp.device_id, SharedHostGroup(tp.group, tp.device_id))
