"""Guarded same-host TP4+TP4 bring-up; never stop existing services."""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def port(name, default):
    return int(os.environ.get(name, default))


def preflight(run_dir, service_ports=None):
    report = subprocess.check_output(["npu-smi", "info"], text=True)
    (run_dir / "npu_preflight.txt").write_text(report)
    failures = []
    for device in range(8):
        idle = f"No running processes found in NPU {device} " in report
        present = re.search(rf"\|\s*{device}\s*\|\s*Ascend950\w*\s*\|", report)
        if not idle or present is None:
            failures.append(f"NPU{device}: idle={idle}, present={present is not None}")
    ports = [port(name, default) for name, default in (
        ("FSA_PREFILL_PORT", 31181), ("FSA_DECODE_PORT", 32181),
        ("FSA_ROUTER_PORT", 6681), ("FSA_BOOTSTRAP_PORT", 8991),
        ("FSA_PREFILL_DIST_PORT", 29581), ("FSA_DECODE_DIST_PORT", 29591),
        ("FSA_STORE_PORT", 24681),
    )]
    if service_ports is not None:
        ports = service_ports
    if len(set(ports)) != len(ports):
        failures.append("Service ports must be distinct")
    for address in ports:
        with socket.socket() as probe:
            try:
                probe.bind(("0.0.0.0", address))
            except OSError as error:
                failures.append(f"Port {address}: {error}")
    if failures:
        raise RuntimeError("Resource gate refused launch: " + "; ".join(failures))


def ready(address):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{address}/health", timeout=3) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_ready(processes, address, deadline):
    while time.monotonic() < deadline:
        for role, process in processes.items():
            if process.poll() is not None:
                raise RuntimeError(f"{role} exited {process.returncode}; inspect its log")
        if ready(address):
            return
        time.sleep(5)
    raise TimeoutError(f"Service on port {address} did not become ready")


def smoke(run_dir, address=None):
    if address is None:
        address = port("FSA_ROUTER_PORT", 6681)
    prompts = [
        "Please answer briefly: what is two plus three?",
        "Remember the number 7391. " + "This is a long context validation passage. " * 1300
        + " What number were you asked to remember?",
    ]
    for index, prompt in enumerate(prompts):
        payload = {"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": 64}}
        (run_dir / f"request_{index}.json").write_text(json.dumps(payload))
        request = urllib.request.Request(
            f"http://127.0.0.1:{address}/generate", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
        (run_dir / f"response_{index}.json").write_text(json.dumps(result, ensure_ascii=False))
        if not isinstance(result, dict) or not result.get("text"):
            raise RuntimeError(f"Request {index} returned no generated text")
        meta = result.get("meta_info", {})
        if meta.get("finish_reason", {}).get("type") in ("abort", "error"):
            raise RuntimeError(f"Request {index} failed: {meta}")
        print(json.dumps({"request": index, "elapsed_s": time.monotonic() - started,
                          "meta_info": meta}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ready-timeout", type=int, default=2400)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(Path("/home/j00628475")):
        raise ValueError("Run directory must stay under /home/j00628475")
    run_dir.mkdir(parents=True, exist_ok=False)
    role_script = Path(__file__).with_name("fsa_pd_role.sh").resolve()
    manifest = {"started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source_script": str(role_script), "processes": {}, "status": "checking",
                "context_length": os.environ.get("FSA_CONTEXT_LENGTH", "32768"),
                "mtp_steps": os.environ.get("FSA_MTP_STEPS", "0")}

    def save():
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    save()
    try:
        for role in ("prefill", "decode", "router"):
            with (run_dir / f"{role}_check.log").open("w") as output:
                subprocess.run(["bash", str(role_script), role, "check"],
                               stdout=output, stderr=subprocess.STDOUT, timeout=120, check=True)
        preflight(run_dir)
        manifest["status"] = "preflight_passed"
        save()
        if not args.launch:
            print("Configuration and idle/health gates passed; no service launched", flush=True)
            return
        processes = {}
        for role in ("prefill", "decode"):
            with (run_dir / f"{role}.log").open("w") as output:
                process = subprocess.Popen(["bash", str(role_script), role],
                                           stdout=output, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            processes[role] = process
            manifest["processes"][role] = {"pid": process.pid,
                "start_ticks": Path(f"/proc/{process.pid}/stat").read_text().split(")", 1)[1].split()[19]}
            save()
        deadline = time.monotonic() + args.ready_timeout
        wait_ready(processes, port("FSA_PREFILL_PORT", 31181), deadline)
        wait_ready(processes, port("FSA_DECODE_PORT", 32181), deadline)
        with (run_dir / "router.log").open("w") as output:
            process = subprocess.Popen(["bash", str(role_script), "router"],
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        processes["router"] = process
        manifest["processes"]["router"] = {"pid": process.pid,
            "start_ticks": Path(f"/proc/{process.pid}/stat").read_text().split(")", 1)[1].split()[19]}
        manifest["status"] = "waiting_for_router"
        save()
        wait_ready(processes, port("FSA_ROUTER_PORT", 6681), time.monotonic() + 120)
        smoke(run_dir)
        manifest["status"] = "smoke_passed"
        save()
    except Exception as error:
        manifest.update(status="blocked_or_failed", error=str(error))
        save()
        print(str(error), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
