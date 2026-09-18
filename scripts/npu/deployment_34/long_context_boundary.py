"""Measure simultaneous full-context Decode, including oversubscribed batches."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
import threading
import time

SOURCE = Path('/home/j00628475/sglang_release_20260917_f524c79a/sglang_remote')
sys.path.insert(0, str(SOURCE / 'scripts/npu'))
from benchmark_fsa_long_context import MemorySampler, build_tokens, request_json, scheduler_pids, stream_case

p = argparse.ArgumentParser()
p.add_argument('service', type=Path)
p.add_argument('--count', type=int, required=True)
p.add_argument('--label', required=True)
a = p.parse_args()
manifest = json.loads((a.service / 'manifest.json').read_text())
pid = manifest['server_pid']
assert Path(f'/proc/{pid}/stat').read_text().split(')', 1)[1].split()[19] == manifest['start_ticks']
directory = a.service / a.label
directory.mkdir(exist_ok=False)
base = f"http://127.0.0.1:{manifest['port']}"
info = request_json(base, '/server_info')
assert info['tp_size'] == 8 and info['speculative_num_steps'] == 3
assert info['speculative_num_draft_tokens'] == 4 and info['disable_radix_cache']
(directory / 'server_info_before.json').write_text(json.dumps(info, indent=2))
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(info['tokenizer_path'], trust_remote_code=True)
tokens = build_tokens(tokenizer, 262144)
pids = scheduler_pids(pid)
assert len(pids) == 8, pids
barrier = threading.Barrier(a.count)
records, errors = [], []
offset = (a.service / 'server.log').stat().st_size

def execute(index):
    case = directory / f'request_{index}'
    case.mkdir()
    barrier.wait(timeout=120)
    return {'request': index, **stream_case(base, case, tokens, 1024)}

started = time.monotonic()
with MemorySampler(directory, pids), ThreadPoolExecutor(max_workers=a.count) as pool:
    futures = {pool.submit(execute, i): i for i in range(a.count)}
    for future in as_completed(futures):
        try:
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
        except Exception as error:
            errors.append({'request': futures[future], 'error': repr(error)})
            print(json.dumps(errors[-1]), flush=True)
        (directory / 'progress.json').write_text(json.dumps({'completed': records, 'errors': errors}, indent=2))
with (a.service / 'server.log').open('rb') as stream:
    stream.seek(offset)
    log = stream.read().decode(errors='replace')
decode = []
for line in log.splitlines():
    match = re.search(r'Decode batch, #running-req: (\d+), #full token: (\d+)', line)
    if match:
        n, t = map(int, match.groups())
        decode.append({'running': n, 'tokens': t, 'full_context': t >= n * 262144,
                       'graph': 'npu graph: True' in line, 'line': line})
full = [row for row in decode if row['full_context']]
summary = {'submitted': a.count, 'input_tokens': 262144, 'output_tokens': 1024,
           'elapsed_s': time.monotonic() - started, 'completed': len(records), 'errors': errors,
           'max_full_context_decode': max((r['running'] for r in full), default=0),
           'max_full_context_graph_decode': max((r['running'] for r in full if r['graph']), default=0),
           'requests': records, 'decode_evidence': decode}
(directory / 'summary.json').write_text(json.dumps(summary, indent=2))
try:
    after = request_json(base, '/server_info')
except Exception as error:
    after = {'unavailable': repr(error)}
(directory / 'server_info_after.json').write_text(json.dumps(after, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k not in ('requests', 'decode_evidence')}), flush=True)
raise SystemExit(bool(errors) or len(records) != a.count)
