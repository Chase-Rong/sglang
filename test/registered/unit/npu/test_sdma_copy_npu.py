"""Opt-in SDMA compatibility gates for the existing registered Host allocator."""

import os
from pathlib import Path

import pytest
import torch

from sdma_copy_lab import AclCopyBridge, BoundRowCopy

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_SDMA_TEST") != "1",
    reason="requires verified idle NPU, TASK_QUEUE_ENABLE=0 and the test SDMA library",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [64, 512])
def test_existing_registered_host_sdma_roundtrip(dtype, width):
    import torch_npu

    from sglang.srt.hardware_backend.npu.fsa_native import RegisteredHostRegion

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch_npu.npu.set_device(device_id)
    bridge = AclCopyBridge(os.environ["SGLANG_SDMA_TEST_LIBRARY"])
    region = RegisteredHostRegion.allocate((1024, width), dtype, device_id=device_id)
    buffer = torch.empty((1024, width), dtype=dtype, device=f"npu:{device_id}")
    host_rows = [128, 129, 130, 19, 20, 700, 701, -1]
    device_rows = [256, 257, 258, 37, 38, 5, 6, -1]
    valid = [True] * 7 + [False]
    load = BoundRowCopy(host_rows, device_rows, valid, 1024, 1024, width * 2)
    save = BoundRowCopy(device_rows, host_rows, valid, 1024, 1024, width * 2)
    compute = torch.npu.current_stream()
    h2d = torch.npu.Stream(device=device_id)
    d2h = torch.npu.Stream(device=device_id)
    template = (torch.arange(1024 * width).reshape(1024, width) % 97).to(dtype) / 128
    try:
        for generation in range(3):
            region.tensor.copy_(template + generation)
            expected = region.tensor.clone()
            expected[host_rows[:-1]] += 0.25
            input_ready = torch.npu.Event()
            input_ready.record(compute)
            with torch.npu.stream(h2d):
                h2d.wait_event(input_ready)
                load.submit(bridge, region.tensor, buffer, 1, h2d, region)
                loaded = h2d.record_event()
            compute.wait_event(loaded)
            buffer.add_(0.25)
            written = compute.record_event()
            with torch.npu.stream(d2h):
                d2h.wait_event(written)
                save.submit(bridge, buffer, region.tensor, 2, d2h, region)
                saved = d2h.record_event()
            compute.wait_event(saved)
            compute.synchronize()
            torch.testing.assert_close(region.tensor, expected, rtol=0, atol=0)
            bridge.release_after_drain()
    finally:
        torch.npu.synchronize()
        bridge.release_after_drain()
        region.close(drained=True)


def test_sdma_against_aiv_with_same_sfa_workload(monkeypatch):
    from test_prefill_overlap_sfa_npu import exercise_sfa_pipeline

    prefix = int(os.environ.get("SGLANG_FSA_TEST_PREFIX", "131072"))
    chunk = int(os.environ.get("SGLANG_FSA_TEST_CHUNK", "8192"))
    repeats = int(os.environ.get("SGLANG_FSA_TEST_REPEATS", "7"))
    destination = Path(os.environ["SGLANG_FSA_WORK_DIR"])
    results = []
    candidate = os.environ.get("SGLANG_SDMA_LAB_CANDIDATE", "sdma")
    assert candidate in ("sdma", "sdma_pinned_staging")
    for backend in ("aiv", candidate, candidate, "aiv"):
        monkeypatch.setenv("SGLANG_SDMA_LAB_BACKEND", backend)
        ordinal = len(results)
        profile_root = os.environ.get("SGLANG_SDMA_TEST_TRACE")
        trace = (
            None
            if profile_root is None
            else str(Path(profile_root) / f"{ordinal}_{backend}")
        )
        result = exercise_sfa_pipeline(prefix, chunk, repeats, profile_dir=trace)
        results.append({"backend": backend, "result": result})
    import json

    assert len({entry["result"]["output_sha256"] for entry in results}) == 1
    (destination / f"{candidate}_abba_{prefix}_{chunk}.json").write_text(
        json.dumps(results, indent=2)
    )
    print(json.dumps(results), flush=True)
