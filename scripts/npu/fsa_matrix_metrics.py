"""Metric definitions shared by the benchmark and workbook exporter."""

import statistics
import math


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def inference_ttft_ms(record):
    metadata = record.get("meta_info", {})
    start = metadata.get("forward_entry_time")
    finish = metadata.get("prefill_finished_time")
    if metadata.get("num_retractions", 0):
        return None
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
               for value in (start, finish)) or finish < start:
        return None
    return (finish - start) * 1000


def request_latency_means(records):
    request_itls = [statistics.mean(record["itl_ms"]) for record in records if record["itl_ms"]]
    ttfts = [inference_ttft_ms(record) for record in records]
    return {
        "ttft_ms": statistics.mean(ttfts) if ttfts and all(value is not None for value in ttfts) else None,
        "tpot_ms": statistics.mean(record["tpot_ms"] for record in records),
        "itl_ms": statistics.mean(request_itls) if request_itls else None,
    }


def summarize(records, elapsed_s, cards=8):
    if not records or elapsed_s <= 0 or cards <= 0:
        raise ValueError("Successful requests, positive duration and cards are required")
    output = sum(record["completion_tokens"] for record in records)
    inputs = sum(record["prompt_tokens"] for record in records)
    intervals = [interval for record in records for interval in record["itl_ms"]]
    request_itls = [statistics.mean(record["itl_ms"]) for record in records if record["itl_ms"]]
    result = {
        **request_latency_means(records),
        "client_ttft_ms": statistics.mean(record["ttft_s"] * 1000 for record in records),
        "ttft_basis": "scheduler_forward_entry_to_prefill_finished",
        "output_tokens_per_s": output / elapsed_s,
        "e2e_tokens_per_s": (inputs + output) / elapsed_s,
        "requests_per_s": len(records) / elapsed_s,
        "elapsed_s": elapsed_s,
        "completed_requests": len(records),
        "total_output_tokens": output,
        "total_input_tokens": inputs,
        "num_retractions": sum(record["meta_info"].get("num_retractions", 0) for record in records),
    }
    result["output_tokens_per_s_per_card"] = result["output_tokens_per_s"] / cards
    result["e2e_tokens_per_s_per_card"] = result["e2e_tokens_per_s"] / cards
    ttfts = [inference_ttft_ms(record) for record in records]
    for name, values in (
        ("ttft", ttfts if all(value is not None for value in ttfts) else []),
        ("client_ttft", [record["ttft_s"] * 1000 for record in records]),
        ("tpot", [record["tpot_ms"] for record in records]),
        ("itl", request_itls),
        ("token_itl", intervals),
    ):
        for label, fraction in (("p50", 0.5), ("p95", 0.95)):
            result[f"{name}_{label}_ms"] = percentile(values, fraction) if values else None
    return result
