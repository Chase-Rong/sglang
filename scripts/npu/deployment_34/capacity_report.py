"""Summarize retained capacity evidence without importing accelerator libraries."""
import argparse
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
reports = []
for run in sorted(a.root.glob('capacity256k_*')):
    if not run.is_dir():
        continue
    logpath = run / 'server.log'
    if not logpath.exists():
        continue
    log = logpath.read_text(errors='replace')
    capacities = re.findall(r'max_total_num_tokens=(\d+)', log)
    decode = []
    for line in log.splitlines():
        m = re.search(r'Decode batch, #running-req: (\d+), #full token: (\d+)', line)
        if m and int(m[2]) >= int(m[1]) * 262144:
            decode.append((int(m[1]), 'npu graph: True' in line))
    summaries = []
    for path in run.glob('*/summary.json'):
        s = json.loads(path.read_text())
        summaries.append({k: v for k, v in s.items() if k not in ('requests', 'decode_evidence')})
    hbm = {}
    for path in run.glob('*/memory_samples.jsonl'):
        for line in path.read_text().splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            for used, total in re.findall(r'(\d+)\s*/\s*(98304)', sample.get('npu_smi', '')):
                hbm['sampled_peak_hbm_mib_any_card'] = max(hbm.get('sampled_peak_hbm_mib_any_card', 0), int(used))
    reports.append({'run': run.name, 'pool_tokens': int(capacities[-1]) if capacities else None,
                    'max_full_decode': max((n for n, _ in decode), default=0),
                    'max_full_graph_decode': max((n for n, g in decode if g), default=0),
                    'summary': summaries, **hbm})
print(json.dumps(reports, indent=2))
