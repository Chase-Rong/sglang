"""Run the resident arm only after the owned offload arm releases the host."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmark_fsa_matrix import identity, owned_members, save, stop_owned
from run_fsa_pd_e2e import preflight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path("/home/j00628475")):
        raise ValueError("Use the assigned work directory")
    own_run = root / "baseline_coordinator"
    own_run.mkdir(exist_ok=False)
    record = {"status": "waiting_for_offload", "pid": os.getpid(), "start_ticks": identity(os.getpid())[0]}
    save(own_run / "status.json", record)
    offload = root / "offload64"
    runner = json.loads((offload / "matrix_runner.json").read_text())
    final_path = offload / "matrix/final.json"
    while not final_path.exists():
        process = identity(runner["pid"])
        if process is None or process[0] != runner["start_ticks"] or process[1] == "Z":
            record.update(status="blocked", reason="Owned offload runner exited without a final receipt")
            save(own_run / "status.json", record)
            return 1
        time.sleep(30)
    offload_result = json.loads(final_path.read_text())
    if offload_result.get("cleanup_error") or offload_result.get("cleanup", {}).get("remaining_owned_pids"):
        record.update(status="blocked", reason="Offload cleanup is not confirmed")
        save(own_run / "status.json", record)
        return 1
    deadline = time.monotonic() + 8 * 3600
    while True:
        try:
            preflight(own_run, [19244, 29244])
            break
        except RuntimeError as error:
            record.update(status="waiting_for_idle", reason=str(error))
            save(own_run / "status.json", record)
            if time.monotonic() >= deadline:
                return 1
            time.sleep(30)
    baseline = root / "baseline64"
    environment = os.environ.copy()
    environment.update(
        SGLANG_FSA_WORK_DIR="/home/j00628475/fsa_glm53_pd_20260915",
        FSA_CONTEXT_LENGTH="1048576", FSA_MAX_TOTAL_TOKENS="auto", FSA_MAX_RUNNING_REQUESTS="64",
        FSA_INTEGRATED_PORT="19244", FSA_INTEGRATED_DIST_PORT="29244", FSA_MEM_FRACTION="0.85",
        SGLANG_NPU_DECODE_HOST_KV_BYTES="0",
    )
    record.update(status="starting_baseline")
    save(own_run / "status.json", record)
    launch = [sys.executable, str(root / "source/scripts/npu/run_fsa_integrated_e2e.py"),
              "--launch", "--resident", "--skip-smoke", "--run-dir", str(baseline)]
    with (own_run / "launch.log").open("w") as output:
        launch_result = subprocess.run(launch, env=environment, stdout=output, stderr=subprocess.STDOUT)
    if launch_result.returncode:
        record.update(status="baseline_start_failed", exit_code=launch_result.returncode)
        manifest_path = baseline / "manifest.json"
        if manifest_path.exists():
            owner = json.loads(manifest_path.read_text()).get("processes", {}).get("integrated")
            if owner and (current := identity(owner["pid"])) is not None and current[0] == owner["start_ticks"]:
                record["cleanup"] = stop_owned(owner, own_run, owned_members(owner))
        save(own_run / "status.json", record)
        return 1
    for name in ("benchmark_fsa_matrix.py", "benchmark_fsa_long_context.py", "fsa_matrix_metrics.py"):
        shutil.copy2(offload / name, baseline / name)
    command = [sys.executable, str(baseline / "benchmark_fsa_matrix.py"), "--run-dir", str(baseline / "matrix"),
               "--service-manifest", str(baseline / "manifest.json"), "--port", "19244", "--arm", "baseline", "--host", args.host]
    with (baseline / "matrix_runner.log").open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        save(baseline / "matrix_runner.json", {"pid": process.pid, "start_ticks": identity(process.pid)[0]})
        record.update(status="baseline_running", runner_pid=process.pid)
        save(own_run / "status.json", record)
        exit_code = process.wait()
    record.update(status="completed" if exit_code == 0 else "baseline_failed", exit_code=exit_code)
    save(own_run / "status.json", record)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
