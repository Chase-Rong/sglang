"""Collect actual device kernel evidence from selected TP profiler exports."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.directory.glob(f"*_{args.pid}_*/ASCEND_PROFILER_OUTPUT/kernel_details.csv")):
        selected = defaultdict(list)
        with path.open() as source:
            for row in csv.DictReader(source):
                if row["Name"] in ("SparseFlashAttention", "FusedSparseAttentionOverlap", "unidex_copy"):
                    selected[row["Name"]].append(row)
        records = {}
        for name, rows in selected.items():
            durations = [float(row["Duration(us)"]) for row in rows]
            records[name] = {
                "count": len(rows), "duration_us_mean": statistics.mean(durations),
                "duration_us_min": min(durations), "duration_us_max": max(durations),
                "accelerator_cores": dict(Counter(row["Accelerator Core"] for row in rows)),
                "input_shapes": dict(Counter(row["Input Shapes"] for row in rows)),
            }
        summaries.append({"file": str(path), "device_kernels": records})
    if not summaries or not any(
        "FusedSparseAttentionOverlap" in record["device_kernels"]
        for record in summaries
    ):
        raise RuntimeError("No fused device kernel found in the selected TP traces")
    output = args.directory.parent / "trace_kernel_evidence.json"
    output.write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
