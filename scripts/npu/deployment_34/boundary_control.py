"""Own launch and benchmark process groups for the 256K capacity experiment."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/home/j00628475/sglang_release_20260917_f524c79a')
p = argparse.ArgumentParser()
p.add_argument('action', choices=['start', 'bench', 'stop', 'status'])
p.add_argument('name')
p.add_argument('--role', choices=['offload', 'resident'])
p.add_argument('--count', type=int)
p.add_argument('--label', default='boundary')
a = p.parse_args()
assert '/' not in a.name and a.name.startswith('capacity256k_')
run = ROOT / a.name

def ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None

def spawn(cmd, record, logfile):
    with logfile.open('wb') as output:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    owner = {'pid': process.pid, 'start_ticks': ticks(process.pid), 'pgid': os.getpgid(process.pid)}
    record.write_text(json.dumps(owner))
    print(json.dumps(owner))

if a.action == 'start':
    run.mkdir(exist_ok=False)
    launcher = 'handoff_launch_offload_mtp_graph32.sh' if a.role == 'offload' else 'handoff_launch_resident_mtp_graph4.sh'
    spawn(['bash', str(ROOT / launcher), str(run)], run / 'owner.json', run / 'driver.log')
elif a.action == 'bench':
    spawn(['python3', str(ROOT / 'long_context_boundary.py'), str(run), '--count', str(a.count), '--label', a.label],
          run / (a.label + '_owner.json'), run / (a.label + '_driver.log'))
elif a.action == 'stop':
    import psutil
    records = list(run.glob('*owner.json'))
    tracked = {}
    for path in records:
        owner = json.loads(path.read_text())
        anchors = [(owner['pid'], owner['start_ticks'])]
        if path.name == 'owner.json' and (run / 'manifest.json').exists():
            m = json.loads((run / 'manifest.json').read_text())
            anchors.append((m['server_pid'], m['start_ticks']))
        live = [pid for pid, start in anchors if ticks(pid) == start]
        for pid in live:
            assert os.getpgid(pid) == owner['pgid']
            for proc in [psutil.Process(pid)] + psutil.Process(pid).children(recursive=True):
                tracked[proc.pid] = ticks(proc.pid)
        if live:
            os.killpg(owner['pgid'], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        alive = [pid for pid, start in tracked.items() if ticks(pid) == start and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE]
        if not alive:
            break
        time.sleep(1)
    for pid in alive:
        if ticks(pid) == tracked[pid]:
            os.kill(pid, signal.SIGKILL)
    print(json.dumps({'tracked': list(tracked), 'killed': alive}))
else:
    for filename in ('driver.log', 'server.log', a.label + '_driver.log'):
        path = run / filename
        if path.exists():
            print(filename)
            print('\n'.join(path.read_text(errors='replace').splitlines()[-6:])[:6000])
    for path in run.glob('*/summary.json'):
        report = json.loads(path.read_text())
        print(str(path), json.dumps({k: v for k, v in report.items() if k not in ('requests', 'decode_evidence')}))
