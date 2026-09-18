"""Read-only evidence for the fixed library and steady-state Decode batches."""
import json
from pathlib import Path
import re
import sys
import psutil

root = Path('/home/j00628475/sglang_release_20260917_f524c79a')
run = root / sys.argv[1]
manifest = json.loads((run / 'manifest.json').read_text())
expected = root / 'fsa-binding-dma-segments/libsgl_fsa_extension.so'
log = (run / 'server.log').read_text(errors='replace')
pid = manifest['server_pid']
live = Path(f'/proc/{pid}/stat').exists()
mapped = []
if live:
    assert Path(f'/proc/{pid}/stat').read_text().split(')', 1)[1].split()[19] == manifest['start_ticks']
    for process in psutil.Process(pid).children(recursive=True):
        try:
            maps = Path(f'/proc/{process.pid}/maps').read_text()
        except FileNotFoundError:
            continue
        if str(expected) in maps:
            mapped.append({'pid': process.pid, 'name': process.name(), 'library': str(expected)})
    (run / 'loaded_extension_maps.json').write_text(json.dumps(mapped, indent=2))
decodes = []
for line in log.splitlines():
    match = re.search(r'Decode batch, #running-req: (\d+), #full token: (\d+)', line)
    if match and int(match[2]) >= int(match[1]) * 262144:
        decodes.append({'batch': int(match[1]), 'graph': 'npu graph: True' in line, 'line': line})
errors = [line for line in log.splitlines()
          if 'server_args=' not in line and
          re.search(r'Prefill DMA submission failed|out of memory|OutOfMemoryError|retract(?:ed|ing)\s+\d+', line, re.I)]
pool = re.findall(r'max_total_num_tokens=(\d+)', log)
report = {'server_live': live, 'fixed_library_processes': len(mapped),
          'pool_tokens': int(pool[-1]) if pool else None,
          'max_full_decode': max((row['batch'] for row in decodes), default=0),
          'last_decode': decodes[-1:] or [], 'capacity_error_lines': errors[-8:]}
summary = run / 'boundary/summary.json'
if summary.exists():
    data = json.loads(summary.read_text())
    report['result'] = {key: value for key, value in data.items() if key not in ('requests', 'decode_evidence')}
    report['retractions'] = sum(item['meta_info'].get('num_retractions', 0) for item in data['requests'])
    report['mtp_accept_lengths'] = [item['meta_info'].get('spec_accept_length') for item in data['requests']]
print(json.dumps(report, indent=2))
