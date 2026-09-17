"""CPU-only checks for the paired performance report definitions."""

import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).parents[4] / "scripts/npu/fsa_matrix_metrics.py"
SPEC = importlib.util.spec_from_file_location("fsa_matrix_metrics", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_throughput_uses_batch_duration_and_eight_cards():
    records = [
        {"prompt_tokens": 100, "completion_tokens": 128, "ttft_s": 1,
         "tpot_ms": 10, "itl_ms": [10, 20], "meta_info": {"num_retractions": 0, "forward_entry_time": 100, "prefill_finished_time": 100.25}},
        {"prompt_tokens": 200, "completion_tokens": 128, "ttft_s": 3,
         "tpot_ms": 30, "itl_ms": [30], "meta_info": {"num_retractions": 1}},
    ]
    result = MODULE.summarize(records, 20)
    assert result["ttft_ms"] is None
    assert result["client_ttft_ms"] == 2000
    assert result["tpot_ms"] == 20
    assert result["itl_ms"] == 22.5
    assert result["output_tokens_per_s"] == 12.8
    assert result["e2e_tokens_per_s"] == 27.8
    assert result["output_tokens_per_s_per_card"] == 1.6
    assert result["e2e_tokens_per_s_per_card"] == 3.475
    assert result["num_retractions"] == 1
    assert result["requests_per_s"] == 0.1
    assert result["ttft_p95_ms"] is None
    assert result["client_ttft_p95_ms"] == 2900
    assert result["itl_p50_ms"] == 22.5
    assert result["token_itl_p50_ms"] == 20
    longer_batch = MODULE.summarize(records, 40)
    for metric in ("ttft_ms", "itl_ms", "tpot_ms"):
        assert longer_batch[metric] == result[metric]
    assert longer_batch["output_tokens_per_s"] == result["output_tokens_per_s"] / 2


def test_empty_or_nonpositive_measurement_is_not_reported_as_throughput():
    with pytest.raises(ValueError):
        MODULE.summarize([], 10)
    with pytest.raises(ValueError):
        MODULE.summarize([{}], 0)


def test_interpolated_percentile_keeps_single_sample():
    assert MODULE.percentile([7], 0.95) == 7
    assert MODULE.percentile([10, 30], 0.5) == 20


def test_inference_ttft_excludes_initial_queue_and_never_guesses_missing_start():
    record = {"ttft_s": 900, "tpot_ms": 10, "itl_ms": [10],
              "meta_info": {"forward_entry_time": 100, "prefill_finished_time": 100.25}}
    assert MODULE.request_latency_means([record])["ttft_ms"] == 250
    record["ttft_s"] = 1800
    assert MODULE.request_latency_means([record])["ttft_ms"] == 250
    record["meta_info"] = {}
    assert MODULE.request_latency_means([record])["ttft_ms"] is None
