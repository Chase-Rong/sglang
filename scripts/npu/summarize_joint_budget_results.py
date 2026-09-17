"""Summarize collected capacity, generation, and sampled memory evidence."""

import argparse
import json
from pathlib import Path
import re


def summarize(directory):
    info = json.loads((directory / "joint_server_info.json").read_text())
    state = info["internal_states"][0]
    generation = json.loads((directory / "joint_generation_summary.json").read_text())
    report = {
        "source": str(directory.resolve()),
        "token_capacity": info["max_total_num_tokens"],
        "request_capacity": state["effective_max_running_requests_per_dp"],
        "startup_memory": state["memory_usage"],
        "semantic_matches": sum(record["matches"] for record in generation),
        "semantic_cases": len(generation),
        "long_context": [],
    }
    extended_path = directory / "joint_extended_generation_summary.json"
    if extended_path.exists():
        extended = json.loads(extended_path.read_text())
        report["extended_semantic_smoke"] = {
            "matches": sum(record["matches"] for record in extended),
            "cases": len(extended),
            "all_finished_normally": all(record["finish_reason"] == "stop" for record in extended),
            "max_output_tokens": sorted({record["max_output_tokens"] for record in extended}),
        }
    for case in sorted((directory / "long_context").glob("tokens_*_repeat_*")):
        if not (case / "timing.json").exists():
            continue
        timing = json.loads((case / "timing.json").read_text())
        hbm_samples = []
        host_samples = []
        for line in (case / "memory_samples.jsonl").read_text().splitlines():
            sample = json.loads(line)
            usage = [int(match.group(1))
                     for row in sample.get("npu_smi", "").splitlines()
                     if "| NA " in row
                     and (match := re.search(r"(\d+)\s*/\s*(\d+)\s*\|\s*$", row))]
            if len(usage) == 8:
                hbm_samples.append(sum(usage) / 1024)
            processes = sample.get("host", {})
            if len(processes) == 8 and all("Pss_kib" in entry for entry in processes.values()):
                host_samples.append(sum(entry["Pss_kib"] for entry in processes.values()) / (1024 ** 2))
        report["long_context"].append({
            "prompt_tokens": timing["prompt_tokens"],
            "completion_tokens": timing["completion_tokens"],
            "ttft_s": timing["ttft_s"],
            "tpot_ms": timing["tpot_ms"],
            "num_retractions": timing["meta_info"]["num_retractions"],
            "sampled_hbm_peak_8_devices_gib": max(hbm_samples, default=None),
            "sampled_scheduler_pss_peak_gib": max(host_samples, default=None),
            "hbm_sample_count": len(hbm_samples),
        })
    report["long_context"].sort(key=lambda record: record["prompt_tokens"])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = summarize(args.directory)
    (args.directory / "qualification_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
