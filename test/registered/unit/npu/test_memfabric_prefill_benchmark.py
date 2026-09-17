"""Matched no-mirror MemFabric DMA/AIV component test, not model TPOT."""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_UNIFIED_HOST_TEST") != "1",
    reason="requires idle NPU and candidate binding",
)


def test_memfabric_prefill_abba():
    from test_prefill_overlap_sfa_npu import exercise_sfa_pipeline

    results = []
    destination = Path(os.environ["SGLANG_FSA_WORK_DIR"]) / (
        "production_prefill_abba_queue" + os.environ.get("TASK_QUEUE_ENABLE", "1")
    )
    destination.mkdir(exist_ok=True)
    for position, backend in enumerate(("aiv", "sdma", "sdma", "aiv")):
        result = exercise_sfa_pipeline(
            131072,
            8192,
            7,
            modes=[(backend == "sdma", backend == "sdma")],
        )
        results.append({"position": position, "backend": backend, **result})
        (destination / "timings.json").write_text(json.dumps(results, indent=2))
        print("MEMFABRIC_ABBA_ARM", position, backend, result["modes"], flush=True)
    assert len({result["output_sha256"] for result in results}) == 1
