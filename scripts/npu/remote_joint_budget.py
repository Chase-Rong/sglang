"""Deploy an isolated source snapshot and inspect an authorized shared NPU host."""

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import time

import paramiko

try:
    from ssh_auth import connect_host
except ImportError:  # script directory not on sys.path
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from ssh_auth import connect_host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--container", default="jzx_sglang_5.3_flash")
    parser.add_argument("--root", default="/home/j00628475/joint_budget_20260916_v2")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--port", type=int, default=33226)
    parser.add_argument("--dist-port", type=int, default=29726)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--total-tokens", default="auto")
    parser.add_argument("--host-kv-gib", type=int, default=128)
    parser.add_argument("--extension")
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--lengths", type=int, nargs="+", default=[500000, 900000, 1000000])
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--benchmark-name", default="long_context")
    parser.add_argument("--profile-first", action="store_true")
    parser.add_argument("--watch-count", type=int, default=1)
    parser.add_argument("--semantic-output-tokens", type=int, default=64)
    parser.add_argument("--semantic-concurrency", type=int, default=8)
    parser.add_argument("--semantic-first-index", type=int, default=0)
    parser.add_argument("--validation-name", default="joint")
    parser.add_argument("--expected-requests", type=int)
    parser.add_argument("--minimum-tokens", type=int, default=1048576)
    parser.add_argument("--maximum-tokens", type=int)
    parser.add_argument("--action", choices=("inspect", "inspect-runtime", "deploy", "control", "launch", "status", "diagnose", "validate", "benchmark", "concurrent", "matrix", "queue-baseline", "trace-summary", "progress", "stop", "collect"), required=True)
    args = parser.parse_args()
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("Run name must be a single directory name")
    if not args.benchmark_name or Path(args.benchmark_name).name != args.benchmark_name or args.benchmark_name in (".", ".."):
        raise ValueError("Benchmark name must be a single directory name")
    run_dir = args.root + "/" + args.run_name
    if not args.root.startswith("/home/j00628475/") or ".." in args.root.split("/"):
        raise ValueError("Remote output must remain within the assigned work directory")
    client = connect_host(args.host)

    def run(command, check=True, quiet=False):
        print("COMMAND", shlex.join(command), flush=True)
        _, output, errors = client.exec_command(shlex.join(command))
        stdout = output.read().decode(errors="replace")
        stderr = errors.read().decode(errors="replace")
        code = output.channel.recv_exit_status()
        if not quiet:
            print(stdout, end="", flush=True)
        print(stderr, end="", flush=True)
        if check and code:
            raise RuntimeError(f"Remote exit {code}")
        return stdout

    def container(command, **kwargs):
        return run(["docker", "exec", args.container, *command], **kwargs)

    try:
        if args.action == "inspect-runtime":
            code = "import os,json; print(json.dumps({key:value for key,value in os.environ.items() if key in ('LD_LIBRARY_PATH','LD_PRELOAD','ASCEND_CUSTOM_PACKAGE_PATH','ASCEND_AICPU_PATH','ASCEND_OPP_PATH')}))"
            container(["bash", "-c", "\n".join([
                "source /usr/local/Ascend/cann-9.1.0/set_env.sh",
                shlex.join(["python3", "-c", code]),
                "source /usr/local/memfabric_hybrid/set_env.sh",
                shlex.join(["python3", "-c", code]),
            ])])
        elif args.action == "inspect":
            run(["npu-smi", "info"])
            run(["free", "-g"])
            metadata = json.loads(run(["docker", "inspect", args.container], quiet=True))[0]
            print("CONTAINER_STATE", json.dumps({"id": metadata["Id"], "started_at": metadata["State"]["StartedAt"],
                                                "oom_killed": metadata["State"].get("OOMKilled"), "status": metadata["State"]["Status"]}))
            print("CONTAINER_STARTED_AT", metadata["State"]["StartedAt"])
            print("MOUNTS", metadata["Mounts"])
            container(["ps", "-eo", "pid,ppid,stat,etime,args"])
            container(["ls", "-ld", "/mnt/share/w00936111/weights/GLM-5.3-Flash",
                       "/home/j00628475/fsa_glm53_pd_20260915/native-build-v5/libsgl_fsa_extension.so",
                       "/home/j00628475/fsa_glm53_pd_20260915/attention-opp-v8/install/packages/vendors/sgl_fsa_a5"])
        elif args.action == "deploy":
            run(["free", "-g"])
            repo = Path(__file__).resolve().parents[2]
            files = subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=repo
            ).decode().split("\0")
            with tempfile.TemporaryDirectory() as temporary:
                archive = Path(temporary) / "source.tar.gz"
                with tarfile.open(archive, "w:gz") as output:
                    for name in sorted(set(files)):
                        if name and (name.startswith(("python/", "native/fsa/", "scripts/npu/", "test/registered/unit/npu/"))
                                     or name in ("pyproject.toml", "README.md")):
                            path = repo / name
                            if path.is_file() and "__pycache__" not in path.parts:
                                output.add(path, arcname=name)
                container(["mkdir", args.root])
                remote_archive = args.root + "/source.tar.gz"
                with client.open_sftp() as transfer:
                    transfer.put(str(archive), remote_archive)
                container(["mkdir", args.root + "/source"])
                container(["tar", "-xzf", remote_archive, "-C", args.root + "/source"])
        elif args.action == "control":
            container(["bash", "-n", args.root + "/source/scripts/npu/fsa_pd_role.sh"])
            container(["bash", "-n", args.root + "/source/scripts/npu/fsa_offload_env.sh"])
            repo = Path(__file__).resolve().parents[2]
            paths = ["python/sglang/srt/hardware_backend/npu/decode_memory_budget.py",
                     "python/sglang/srt/hardware_backend/npu/fsa_backend.py",
                     "python/sglang/srt/mem_cache/kv_cache_configurator.py",
                     "python/sglang/srt/model_executor/pool_configurator.py",
                     "python/sglang/srt/managers/scheduler.py",
                     "test/registered/unit/npu/test_decode_memory_budget.py",
                     "test/registered/unit/npu/test_joint_budget_configurator.py",
                     "test/registered/unit/npu/test_fsa_runtime_config.py",
                     "test/registered/unit/npu/test_fsa_chunked_admission.py"]
            remote_hashes = container(["sha256sum", *[args.root + "/source/" + path for path in paths]], quiet=True)
            records = {}
            for line, path in zip(remote_hashes.splitlines(), paths, strict=True):
                digest = hashlib.sha256((repo / path).read_bytes()).hexdigest()
                if digest != line.split()[0]:
                    raise RuntimeError(f"Remote control source differs from local file: {path}")
                records[path] = digest
            result = container(["env", "PYTHONPATH=" + args.root + "/source/python", "python3", "-m", "pytest",
                       args.root + "/source/test/registered/unit/npu/test_decode_memory_budget.py",
                       args.root + "/source/test/registered/unit/npu/test_joint_budget_configurator.py",
                       args.root + "/source/test/registered/unit/npu/test_fsa_runtime_config.py",
                       args.root + "/source/test/registered/unit/npu/test_fsa_chunked_admission.py", "-q"])
            destination = repo / "results/joint_budget" / args.host / Path(args.root).name
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "control_source_sha256.json").write_text(json.dumps(records, indent=2))
            (destination / "control_result.txt").write_text(result)
        elif args.action == "queue-baseline":
            repo = Path(__file__).resolve().parents[2]
            offload = args.root + "/offload64"
            with client.open_sftp() as transfer:
                for name in ("continue_fsa_matrix_baseline.py", "run_fsa_pd_e2e.py"):
                    transfer.put(str(repo / "scripts/npu" / name), offload + "/" + name)
            command = ["python3", offload + "/continue_fsa_matrix_baseline.py", "--root", args.root, "--host", args.host]
            code = (
                "import json,pathlib,subprocess; "
                f"root=pathlib.Path({args.root!r}); "
                "assert not (root/'coordinator_owner.json').exists(); "
                "output=(root/'coordinator.log').open('w'); "
                f"process=subprocess.Popen({command!r},stdout=output,stderr=subprocess.STDOUT,start_new_session=True); "
                "identity=pathlib.Path(f'/proc/{process.pid}/stat').read_text().split(')',1)[1].split()[19]; "
                "record={'pid':process.pid,'start_ticks':identity}; (root/'coordinator_owner.json').write_text(json.dumps(record)); print(json.dumps(record))"
            )
            container(["python3", "-c", code])
        elif args.action == "matrix":
            repo = Path(__file__).resolve().parents[2]
            with client.open_sftp() as transfer:
                for name in ("benchmark_fsa_matrix.py", "benchmark_fsa_long_context.py", "fsa_matrix_metrics.py"):
                    transfer.put(str(repo / "scripts/npu" / name), run_dir + "/" + name)
            command = ["python3", run_dir + "/benchmark_fsa_matrix.py", "--run-dir", run_dir + "/matrix",
                       "--service-manifest", run_dir + "/manifest.json", "--port", str(args.port),
                       "--arm", "baseline" if args.resident else "offload", "--host", args.host]
            launch_code = (
                "import json,os,pathlib,subprocess; "
                f"directory=pathlib.Path({run_dir!r}); "
                "assert not (directory/'matrix_runner.json').exists(); "
                "output=(directory/'matrix_runner.log').open('w'); "
                f"process=subprocess.Popen({command!r},stdout=output,stderr=subprocess.STDOUT,start_new_session=True); "
                "identity=pathlib.Path(f'/proc/{process.pid}/stat').read_text().split(')',1)[1].split()[19]; "
                "record={'pid':process.pid,'start_ticks':identity}; "
                "(directory/'matrix_runner.json').write_text(json.dumps(record)); print(json.dumps(record))"
            )
            container(["python3", "-c", launch_code])
        elif args.action == "concurrent":
            repo = Path(__file__).resolve().parents[2]
            with client.open_sftp() as transfer:
                for name in ("benchmark_fsa_concurrent.py", "benchmark_fsa_long_context.py"):
                    transfer.put(str(repo / "scripts/npu" / name), run_dir + "/" + name)
            container(["python3", run_dir + "/benchmark_fsa_concurrent.py",
                       "--run-dir", run_dir + "/" + args.benchmark_name,
                       "--service-manifest", run_dir + "/manifest.json", "--port", str(args.port),
                       "--requests", str(args.requests), "--input-tokens", str(args.lengths[0]),
                       "--output-tokens", str(args.output_tokens)])
        elif args.action in ("validate", "benchmark"):
            repo = Path(__file__).resolve().parents[2]
            name = ("validate_joint_budget_service.py" if args.action == "validate"
                    else "benchmark_fsa_long_context.py")
            script = run_dir + "/" + name
            with client.open_sftp() as transfer:
                transfer.put(str(repo / "scripts/npu" / name), script)
            if args.action == "validate":
                container(["python3", script, "--run-dir", run_dir, "--port", str(args.port),
                           "--max-output-tokens", str(args.semantic_output_tokens),
                           "--concurrency", str(args.semantic_concurrency),
                           "--first-index", str(args.semantic_first_index),
                           "--output-prefix", args.validation_name,
                           "--minimum-tokens", str(args.minimum_tokens),
                           *(["--maximum-tokens", str(args.maximum_tokens)] if args.maximum_tokens is not None else []),
                           *(["--expected-requests", str(args.expected_requests)] if args.expected_requests is not None else [])])
            else:
                container(["python3", script, "--run-dir", run_dir + "/" + args.benchmark_name,
                           "--service-manifest", run_dir + "/manifest.json", "--port", str(args.port),
                           "--lengths", *map(str, args.lengths), "--repeats", "1",
                           "--output-tokens", str(args.output_tokens),
                           *(["--profile-first"] if args.profile_first else [])])
        elif args.action == "trace-summary":
            repo = Path(__file__).resolve().parents[2]
            script = run_dir + "/summarize_fsa_trace.py"
            with client.open_sftp() as transfer:
                transfer.put(str(repo / "scripts/npu/summarize_fsa_trace.py"), script)
                with transfer.open(run_dir + "/manifest.json") as source:
                    owner = json.loads(source.read())["processes"]["integrated"]
            actual = container(["cat", f"/proc/{owner['pid']}/stat"], quiet=True)
            if actual.split(")", 1)[1].split()[19] != owner["start_ticks"]:
                raise RuntimeError("Owned process identity changed")
            processes = container(["ps", "-eo", "pid,ppid,args"], quiet=True)
            selected = [fields[0] for line in processes.splitlines()
                        if len(fields := line.split()) >= 3
                        and fields[1] == str(owner["pid"])
                        and fields[2] == "sglang::scheduler_TP0"]
            if len(selected) != 1:
                raise RuntimeError("Expected exactly one owned TP0 worker")
            container(["python3", script, run_dir + "/" + args.benchmark_name + "/profile",
                       "--pid", selected[0]])
        elif args.action == "progress":
            with client.open_sftp() as transfer:
                paths = [run_dir + "/manifest.json", run_dir + "/server.log",
                         run_dir + "/" + args.benchmark_name + "/summary.json",
                         run_dir + "/" + args.benchmark_name + "/progress.json"]
                if args.benchmark_name == "matrix":
                    paths.extend([run_dir + "/matrix_runner.log", run_dir + "/matrix/final.json", args.root + "/baseline_coordinator/status.json"])
                if args.benchmark_name != "matrix":
                    paths.extend(run_dir + "/" + args.benchmark_name
                                 + f"/tokens_{length}_repeat_0/stream_events.jsonl"
                                 for length in args.lengths)
                if args.benchmark_name.startswith("concurrent"):
                    paths = [path for path in paths if "/tokens_" not in path]
                    paths.extend(run_dir + "/" + args.benchmark_name
                                 + f"/request_{index}/stream_events.jsonl"
                                 for index in range(args.requests))
                for iteration in range(args.watch_count):
                    processes = container(["ps", "-eo", "pid,ppid,stat,etime,args"], quiet=True)
                    print("\n".join(line[:300] for line in processes.splitlines()
                                    if "<defunct>" not in line and any(term in line for term in
                                        ("launch_server", "scheduler_TP0", "benchmark_fsa_concurrent", "benchmark_fsa_matrix"))), flush=True)
                    for path in paths:
                        try:
                            with transfer.open(path) as source:
                                if not path.endswith(".json"):
                                    source.seek(max(0, source.stat().st_size - 8192))
                                lines = source.read().decode(errors="replace").splitlines()
                            if path.endswith(("summary.json", "progress.json")):
                                records = json.loads("\n".join(lines))
                                if isinstance(records, dict):
                                    details = json.dumps({"completed": len(records.get("completed", records.get("requests", records.get("cases", [])))),
                                        "errors": records.get("errors", []),
                                        "phase": records.get("phase"), "input_tokens": records.get("input_tokens"),
                                        "max_decode_running_requests": records.get("max_decode_running_requests", records.get("actual_max_decode_batch"))})
                                else:
                                    details = json.dumps([{key:record.get(key) for key in
                                        ("prompt_tokens", "completion_tokens", "ttft_s", "tpot_ms")}
                                        for record in records])
                            elif path.endswith("manifest.json"):
                                record = json.loads("\n".join(lines))
                                details = json.dumps({key:record.get(key) for key in ("status", "processes")})
                            else:
                                details = "\n".join(line[:500] for line in lines[-2:])
                            print(path, details, sep="\n", flush=True)
                        except FileNotFoundError:
                            print(path, "not yet created", flush=True)
                    if iteration + 1 < args.watch_count:
                        time.sleep(30)
        elif args.action == "collect":
            destination = Path(__file__).resolve().parents[2] / "results/joint_budget" / args.host / Path(args.root).name / args.run_name
            destination.mkdir(parents=True, exist_ok=True)
            import stat

            def download(transfer, remote, local):
                for entry in transfer.listdir_attr(remote):
                    source = remote + "/" + entry.filename
                    target = local / entry.filename
                    if stat.S_ISDIR(entry.st_mode) and entry.filename not in ("profile", "tmp"):
                        target.mkdir(exist_ok=True)
                        download(transfer, source, target)
                    elif stat.S_ISREG(entry.st_mode):
                        transfer.get(source, str(target))

            with client.open_sftp() as transfer:
                download(transfer, run_dir, destination)
            print("Collected", destination)
        elif args.action == "stop":
            with client.open_sftp() as transfer:
                with transfer.open(run_dir + "/manifest.json") as manifest_file:
                    manifest = json.loads(manifest_file.read())
            owner = manifest["processes"]["integrated"]
            stop_code = (
                "import os, pathlib, signal; "
                f"pid={int(owner['pid'])}; "
                "stat=pathlib.Path(f'/proc/{pid}/stat').read_text(); "
                f"assert stat.split(')',1)[1].split()[19]=={owner['start_ticks']!r}; "
                "assert os.getpgid(pid)==pid; os.killpg(pid, signal.SIGTERM)"
            )
            container(["python3", "-c", stop_code])
        elif args.action in ("status", "diagnose"):
            processes = container(["ps", "-eo", "pid,ppid,stat,etime,args"], quiet=True)
            print("\n".join(line[:350] for line in processes.splitlines()
                            if "<defunct>" not in line and any(term in line for term in
                                ("PID", "launch_server", "scheduler_TP", "run_fsa_integrated", "benchmark_fsa", "validate_joint"))))
            container(["cat", run_dir + "/manifest.json"], check=False)
            with client.open_sftp() as transfer:
                with transfer.open(run_dir + "/server.log") as log:
                    lines = log.read().decode(errors="replace").splitlines()
            selected = [line for line in lines if any(term in line for term in (
                "Joint DSA", "Joint request", "Mamba Cache", "KV Cache", "Memory pool", "Traceback",
                "Error:", "Uvicorn running", "max_total_num_tokens=", "Init torch distributed"))]
            print("\n".join(selected[-30:] + [line[:400] for line in lines[-15:]]))
            print("\n".join(line for line in lines if "Joint DSA/KDA budget:" in line))
            if args.action == "diagnose":
                executable = container(["which", "py-spy"], check=False).strip()
                with client.open_sftp() as transfer:
                    with transfer.open(run_dir + "/manifest.json") as manifest_file:
                        manifest = json.loads(manifest_file.read())
                owner = manifest["processes"]["integrated"]
                actual = container(["cat", f"/proc/{owner['pid']}/stat"], quiet=True, check=False)
                if not actual:
                    print("Owned service already exited; diagnostics are in server.log")
                    return
                if actual.split(")", 1)[1].split()[19] != owner["start_ticks"]:
                    raise RuntimeError("Owned process identity changed")
                for line in processes.splitlines():
                    fields = line.split()
                    if len(fields) >= 5 and fields[1] == str(owner["pid"]) and "scheduler_TP" in line:
                        if executable:
                            container([executable, "dump", "--pid", fields[0]], check=False)
                        else:
                            container(["cat", f"/proc/{fields[0]}/wchan"], check=False)
        else:
            container(["env", "SGLANG_FSA_WORK_DIR=/home/j00628475/fsa_glm53_pd_20260915",
                       *(["SGLANG_NPU_FSA_EXTENSION=" + args.extension] if args.extension else []),
                       "FSA_CONTEXT_LENGTH=1048576", "FSA_MAX_TOTAL_TOKENS=" + args.total_tokens,
                       "FSA_MAX_RUNNING_REQUESTS=" + str(args.requests), f"FSA_INTEGRATED_PORT={args.port}",
                       f"FSA_INTEGRATED_DIST_PORT={args.dist_port}", "FSA_MEM_FRACTION=0.85",
                       "SGLANG_NPU_DECODE_HOST_KV_BYTES=" + str(args.host_kv_gib * (1 << 30)),
                       "python3", args.root + "/source/scripts/npu/run_fsa_integrated_e2e.py",
                       "--launch", "--skip-smoke", "--run-dir", run_dir,
                       *(["--resident"] if args.resident else [])])
    finally:
        client.close()


if __name__ == "__main__":
    main()
