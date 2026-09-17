"""Disambiguate host-pointer fallback from device-alias and pinned SDMA copies."""

import json
import os
from pathlib import Path
import statistics
import time

import pytest
import torch

from sdma_copy_lab import AclCopyBridge

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_SDMA_TEST") != "1", reason="requires idle NPU"
)


def test_sdma_pointer_variants():
    import torch_npu

    from sglang.srt.hardware_backend.npu.fsa_native import RegisteredHostRegion

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch_npu.npu.set_device(device_id)
    bridge = AclCopyBridge(os.environ["SGLANG_SDMA_TEST_LIBRARY"])
    size = 128 * 1024 * 1024
    region = RegisteredHostRegion.allocate((size,), torch.uint8, device_id=device_id)
    device = torch.empty(size, dtype=torch.uint8, device=f"npu:{device_id}")
    pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    stream = torch.npu.Stream(device=device_id)
    runs = ((0, 0, size),)
    work = Path(os.environ["SGLANG_FSA_WORK_DIR"])
    results = {}
    try:
        for name, host, host_pointer, direction in (
            ("registered_host", region.tensor, region.host_ptr, 1),
            ("registered_host_pinned", region.tensor, region.host_ptr, 1),
            ("pinned_host", pinned, pinned.data_ptr(), 1),
        ):
            added_pin = name == "registered_host_pinned"
            if added_pin and os.environ.get("SGLANG_SDMA_REPRO_UNSAFE_PIN") != "1":
                results[name] = {"not_run": "previously crashed; requires explicit reproduction flag"}
                continue
            if added_pin:
                code = bridge.library.fsa_sdma_pin_host(region.host_ptr, region.nbytes)
                print("PIN_RESULT", code, flush=True)
                if code:
                    results[name] = {"pin_error": code}
                    continue
            host.fill_(37)
            samples = []
            destination = str(work / "pointer_trace" / name)
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    destination
                ),
            ):
                with torch.npu.stream(stream):
                    for _ in range(4):
                        start = torch.npu.Event(enable_timing=True)
                        end = torch.npu.Event(enable_timing=True)
                        start.record(stream)
                        submitted_at = time.perf_counter()
                        with torch.profiler.record_function("candidate_" + name):
                            bridge.submit(
                                host_pointer,
                                device.data_ptr(),
                                runs,
                                direction,
                                stream,
                                (region, host, device),
                            )
                        submit_ms = (time.perf_counter() - submitted_at) * 1000
                        end.record(stream)
                        end.synchronize()
                        total_ms = (time.perf_counter() - submitted_at) * 1000
                        samples.append(
                            {
                                "submit_ms": submit_ms,
                                "total_ms": total_ms,
                                "event_ms": start.elapsed_time(end),
                            }
                        )
                actual = device.cpu()
                correct = bool(torch.all(actual == 37).item())
                print(
                    "POINTER_ACCURACY",
                    name,
                    correct,
                    "FIRST_BYTES",
                    actual[:16].tolist(),
                    flush=True,
                )
            bridge.release_after_drain()
            results[name] = {
                "correct": correct,
                "samples": samples,
                "submit_median_ms": statistics.median(
                    item["submit_ms"] for item in samples
                ),
            }
            (work / "pointer_results.json").write_text(json.dumps(results, indent=2))
            if added_pin:
                code = bridge.library.fsa_sdma_unpin_host(region.host_ptr)
                print("UNPIN_RESULT", code, flush=True)
                if code:
                    raise RuntimeError(f"Host unpin failed: {code}")
        print(json.dumps(results), flush=True)
    finally:
        torch.npu.synchronize()
        bridge.release_after_drain()
        region.close(drained=True)
