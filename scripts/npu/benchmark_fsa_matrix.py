"""Matched offload/resident input matrix with single and saturated workloads."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

from benchmark_fsa_long_context import MemorySampler, build_tokens, request_json, scheduler_pids, stream_case
from fsa_matrix_metrics import summarize


def save(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    temporary.replace(path)


def identity(pid):
    path = Path(f"/proc/{pid}/stat")
    if not path.exists():
        return None
    values = path.read_text().split(")", 1)[1].split()
    return values[19], values[0], int(values[2])


def owned_members(owner):
    report = subprocess.check_output(["ps", "-eo", "pid,pgid"], text=True)
    members = {}
    for line in report.splitlines()[1:]:
        pid, group = map(int, line.split())
        if group == owner["pid"]:
            members[pid] = identity(pid)
    return members


def stop_owned(owner, directory, members):
    initial = identity(owner["pid"])
    if initial is not None and (initial[0] != owner["start_ticks"] or initial[2] != owner["pid"]):
        raise RuntimeError("Refusing cleanup: service process identity changed")
    if initial is not None and initial[1] != "Z":
        members.update(owned_members(owner))
        os.killpg(owner["pid"], signal.SIGTERM)
    else:
        for pid, previous in members.items():
            current = identity(pid)
            if current is not None and previous is not None and current[0] == previous[0] and current[1] != "Z":
                os.kill(pid, signal.SIGTERM)
    for _ in range(45):
        live = [pid for pid, previous in members.items()
                if (current := identity(pid)) is not None and previous is not None
                and current[0] == previous[0] and current[1] != "Z"]
        if not live:
            break
        time.sleep(1)
    for pid in live:
        current = identity(pid)
        if current is not None and current[0] == members[pid][0] and current[1] != "Z":
            os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        remaining = [pid for pid, previous in members.items()
                     if previous is not None and (current := identity(pid)) is not None
                     and current[0] == previous[0] and current[1] != "Z"]
        if not remaining:
            break
        time.sleep(1)
    report = subprocess.check_output(["npu-smi", "info"], text=True, timeout=30)
    (directory / "npu_after_cleanup.txt").write_text(report)
    return {"tracked_pids": list(members), "forced_owned_pids": live, "remaining_owned_pids": remaining}


def measure(base, case, tokens, encoded_payload=None):
    record = stream_case(base, case, tokens, 128, timeout=43200, allow_retractions=True, encoded_payload=encoded_payload)
    previous_elapsed = None
    previous_count = 0
    intervals = []
    for line in (case / "stream_events.jsonl").read_text().splitlines():
        event = json.loads(line)
        count = event["completion_tokens"]
        if count > previous_count:
            if previous_elapsed is not None:
                delta = count - previous_count
                intervals.extend([(event["elapsed_s"] - previous_elapsed) * 1000 / delta] * delta)
            previous_elapsed = event["elapsed_s"]
            previous_count = count
    record.update(itl_ms=intervals)
    save(case / "result.json", record)
    return record


def run_case(base, directory, info, tokenizer, length, mode, pids, server_log):
    directory.mkdir()
    count = 1 if mode == "single" else 64
    required = ((length + 128 + info["page_size"] - 1) // info["page_size"]) * info["page_size"]
    if info["max_total_num_tokens"] < required or info["max_req_input_len"] < length:
        result = {"status": "no_data", "reason": "Resolved token pool or input limit cannot fit one request",
                  "input_tokens": length, "output_tokens": 128, "mode": mode}
        save(directory / "summary.json", result)
        return result
    prototype = build_tokens(tokenizer, length)
    inputs = []
    encoded_inputs = []
    for index in range(count):
        tokens = prototype.copy()
        marker = tokenizer.encode(f" Document record {index:04d}. ", add_special_tokens=False)
        tokens[256:256 + len(marker)] = marker
        inputs.append(tokens)
        encoded_inputs.append(json.dumps({"input_ids": tokens, "stream": True,
                              "sampling_params": {"temperature": 0, "max_new_tokens": 128, "ignore_eos": True}}).encode())
    save(directory / "input_hashes.json", [hashlib.sha256(json.dumps(tokens).encode()).hexdigest() for tokens in inputs])
    barrier = threading.Barrier(count)
    records = []
    errors = []
    offset = server_log.stat().st_size

    def execute(index):
        case = directory / f"request_{index}"
        case.mkdir()
        barrier.wait(timeout=120)
        return {"request": index, **measure(base, case, inputs[index], encoded_inputs[index])}

    with MemorySampler(directory, []), ThreadPoolExecutor(max_workers=count) as executor:
        futures = {executor.submit(execute, index): index for index in range(count)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                errors.append({"request": index, "error": repr(error)})
            progress = {"phase": mode, "input_tokens": length, "completed": records, "errors": errors}
            save(directory / "progress.json", progress)
            save(directory.parent / "progress.json", {**progress, "completed": [record["request"] for record in records]})
            print(json.dumps({"mode": mode, "input_tokens": length, "completed": len(records), "errors": errors}), flush=True)
    log = server_log.read_bytes()[offset:].decode(errors="replace")
    (directory / "service_interval.log").write_text(log)
    batches = [tuple(map(int, match)) for match in re.findall(r"Decode batch, #running-req: (\d+), #full token: (\d+)", log)]
    result = {"status": "ok" if not errors and len(records) == count else "no_data",
              "input_tokens": length, "output_tokens": 128, "mode": mode,
              "submitted_requests": count, "client_concurrency": count,
              "load_shape": "single_request" if mode == "single" else "finite_batch_64",
              "decode_batch_log_samples": len(batches),
              "decode_batch_histogram": {str(size): sum(batch[0] == size for batch in batches) for size in sorted(set(batch[0] for batch in batches))},
              "actual_max_decode_batch": max((batch[0] for batch in batches), default=0),
              "actual_max_full_tokens": max((batch[1] for batch in batches), default=0),
              "estimated_resident_requests": min(64, info["max_total_num_tokens"] // required),
              "errors": errors, "requests": records}
    if result["status"] == "ok":
        elapsed = max(record["request_finished_monotonic_s"] for record in records) - min(record["request_started_monotonic_s"] for record in records)
        result["metrics"] = summarize(records, elapsed)
    save(directory / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--service-manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--arm", choices=("baseline", "offload"), required=True)
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(Path("/home/j00628475")):
        raise ValueError("Use the assigned work directory")
    directory.mkdir(exist_ok=False)
    owner = json.loads(args.service_manifest.read_text())["processes"]["integrated"]
    state = identity(owner["pid"])
    if state is None or state[0] != owner["start_ticks"]:
        raise RuntimeError("Service ownership changed")
    final = {"arm": args.arm, "host": args.host, "status": "running", "cases": []}
    members = owned_members(owner)
    save(directory / "service_identity.json", {"owner": owner, "members": members})
    try:
        base = f"http://127.0.0.1:{args.port}"
        info = request_json(base, "/server_info")
        save(directory / "server_info.json", info)
        assert info["max_running_requests"] == 64 and info["context_length"] == 1048576
        assert info["max_total_tokens"] is None and info["speculative_algorithm"] is None
        assert info["tp_size"] == 8 and info["disable_radix_cache"]
        environment = dict(item.decode().split("=", 1) for item in Path(f"/proc/{owner['pid']}/environ").read_bytes().split(b"\0") if b"=" in item)
        selected = {name: environment.get(name) for name in (
            "SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD", "SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD",
            "SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION", "SGLANG_NPU_FUSED_SPARSE_ATTENTION_MODE",
            "SGLANG_NPU_SHARED_HOST_KV", "SGLANG_NPU_FSA_EXTENSION", "SGLANG_NPU_PROFILING")}
        save(directory / "runtime_env.json", selected)
        expected = "1" if args.arm == "offload" else "0"
        for name in ("SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD", "SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD", "SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION"):
            assert selected[name] == expected, (name, selected[name])
        assert selected["SGLANG_NPU_PROFILING"] == "0"
        pids = scheduler_pids(owner["pid"])
        assert len(pids) == 8
        configs = {}
        for filename in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
            path = Path(info["model_path"]) / filename
            if path.exists():
                configs[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        save(directory / "weight_metadata_sha256.json", configs)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(info["tokenizer_path"], trust_remote_code=True)
        warmup = directory / "warmup"
        warmup.mkdir()
        measure(base, warmup, build_tokens(tokenizer, 16384))
        for mode in ("single", "saturation"):
            for length in (128000, 256000, 512000, 1000000):
                save(directory / "progress.json", {"phase": mode, "input_tokens": length, "completed": [], "errors": []})
                result = run_case(base, directory / f"{mode}_{length}", info, tokenizer,
                                  length, mode, pids, args.service_manifest.parent / "server.log")
                final["cases"].append({key: value for key, value in result.items() if key != "requests"})
                save(directory / "summary.json", final)
                if result["status"] != "ok" and result.get("errors"):
                    raise RuntimeError("Failed request may still be running; stop this service before further cases")
                request_json(base, "/health", timeout=120)
        final["status"] = "completed"
    except Exception as error:
        final.update(status="failed", error=repr(error))
        completed = {(case["mode"], case["input_tokens"]) for case in final["cases"]}
        for length in (128000, 256000, 512000, 1000000):
            for mode in ("single", "saturation"):
                if (mode, length) not in completed:
                    final["cases"].append({"status": "no_data", "input_tokens": length, "output_tokens": 128,
                                           "mode": mode, "reason": "Service or harness failure prevented completion: " + repr(error)})
        raise
    finally:
        try:
            final["cleanup"] = stop_owned(owner, directory, members)
            if final["cleanup"]["remaining_owned_pids"]:
                final.update(status="failed", cleanup_error="Owned service workers remain alive")
        except Exception as error:
            final.update(status="failed", cleanup_error=repr(error))
        save(directory / "final.json", final)
        print(json.dumps(final), flush=True)
        if final["status"] == "failed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
