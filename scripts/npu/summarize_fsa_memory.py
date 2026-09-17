"""Summarize sampled HBM/PSS and logical KV allocation without summing shared RSS."""

import argparse
import json
from pathlib import Path
import re


def hbm_usage(report):
    result = {}
    device = None
    for line in report.splitlines():
        header = re.search(r"\|\s*(\d+)\s*\|\s*Ascend950\w*\s*\|", line)
        if header:
            device = header.group(1)
            continue
        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
        if device is not None and len(pairs) >= 2:
            result[device] = int(pairs[-1][0])
            device = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    result = {}
    info = json.loads((directory / "server_info_before.json").read_text())
    result["capacity"] = {name: info.get(name) for name in (
        "context_length", "max_total_num_tokens", "max_total_tokens", "tp_size",
        "chunked_prefill_size", "mem_fraction_static", "speculative_algorithm")}
    result["scheduler_memory_usage"] = [entry.get("memory_usage") for entry in info["internal_states"]]
    result["cases"] = {}
    cases = sorted(directory.glob("tokens_*"))
    if (directory / "memory_samples.jsonl").exists():
        cases.append(directory)
    for case in cases:
        samples = []
        for name in ("memory_before.json", "memory_after.json"):
            if (case / name).exists():
                samples.append(json.loads((case / name).read_text()))
        samples.extend(json.loads(line) for line in (case / "memory_samples.jsonl").read_text().splitlines())
        hbm = [hbm_usage(sample["npu_smi"]) for sample in samples]
        valid = [sample for sample in hbm if len(sample) == 8]
        per_device = {str(device): max(sample[str(device)] for sample in valid) for device in range(8)}
        pss = [sum(rank.get("Pss_kib", 0) for rank in sample["host"].values()) / 1024 ** 2
               for sample in samples if len(sample["host"]) == 8]
        result["cases"][case.name] = {
            "samples": len(samples), "valid_hbm_samples": len(valid),
            "sampled_peak_hbm_mib_per_device": per_device,
            "sampled_peak_total_hbm_gib": max(sum(sample.values()) for sample in valid) / 1024,
            "sampled_peak_scheduler_pss_gib": max(pss),
            "before_hbm_mib": hbm_usage(samples[0]["npu_smi"]),
            "after_hbm_mib": hbm_usage(json.loads((case / "memory_after.json").read_text())["npu_smi"])
            if (case / "memory_after.json").exists() else hbm_usage(samples[-1]["npu_smi"]),
        }
    (directory / "memory_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
