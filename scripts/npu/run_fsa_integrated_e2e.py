"""Start one idle-node TP8 service with local Host KV offload and fused Decode."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from run_fsa_pd_e2e import port, preflight, smoke, wait_ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ready-timeout", type=int, default=2400)
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(Path("/home/j00628475")):
        raise ValueError("Run directory must stay under /home/j00628475")
    run_dir.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("fsa_pd_role.sh").resolve()
    role = "resident" if args.resident else "integrated"
    manifest = {"started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": "checking", "processes": {}, "tp": 8,
                "disaggregation_mode": "null", "mtp_steps": os.environ.get("FSA_MTP_STEPS", "0")}
    manifest.update(role=role, context_length=os.environ.get("FSA_CONTEXT_LENGTH", "32768"),
                    max_total_tokens=os.environ.get("FSA_MAX_TOTAL_TOKENS", "32768"),
                    max_running_requests=os.environ.get("FSA_MAX_RUNNING_REQUESTS", "1"),
                    host_kv_bytes=os.environ.get("SGLANG_NPU_DECODE_HOST_KV_BYTES", "17179869184"))
    manifest["native_extension"] = os.environ.get("SGLANG_NPU_FSA_EXTENSION")

    def save():
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    save()
    try:
        with (run_dir / "check.log").open("w") as output:
            subprocess.run(["bash", str(script), role, "check"],
                           stdout=output, stderr=subprocess.STDOUT, check=True, timeout=120)
        address = port("FSA_INTEGRATED_PORT", 33181)
        preflight(run_dir, [address, port("FSA_INTEGRATED_DIST_PORT", 29611)])
        manifest["status"] = "preflight_passed"
        save()
        if not args.launch:
            print("Single-service configuration passed; no model started", flush=True)
            return
        with (run_dir / "server.log").open("w") as output:
            process = subprocess.Popen(["bash", str(script), role],
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        manifest["processes"]["integrated"] = {
            "pid": process.pid,
            "start_ticks": Path(f"/proc/{process.pid}/stat").read_text().split(")", 1)[1].split()[19],
        }
        manifest["status"] = "loading_model"
        save()
        wait_ready({"integrated": process}, address, time.monotonic() + args.ready_timeout)
        manifest["status"] = "ready"
        save()
        if args.skip_smoke:
            return
        smoke(run_dir, address)
        manifest["status"] = "smoke_passed"
        save()
    except Exception as error:
        manifest.update(status="blocked_or_failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
