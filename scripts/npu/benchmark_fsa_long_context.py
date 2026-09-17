"""Exact-token long-context streaming timings and observed host/NPU memory."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request


def request_json(base, route, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + route, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        if not body.strip():
            return {"http_status": response.status}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"http_status": response.status, "body": body.decode(errors="replace")}


def scheduler_pids(parent):
    report = subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True)
    return [int(fields[0]) for line in report.splitlines()[1:]
            if len(fields := line.split(None, 2)) == 3
            and int(fields[1]) == parent and "sglang::scheduler" in fields[2]]


def memory_sample(pids):
    sample = {"wall_time": time.time(), "host": {}, "npu_smi": ""}
    for pid in pids:
        try:
            fields = {}
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                parts = line.split()
                if parts and parts[0] in ("Rss:", "Pss:", "Private_Dirty:", "Shared_Dirty:"):
                    fields[parts[0][:-1] + "_kib"] = int(parts[1])
            sample["host"][str(pid)] = fields
        except OSError as error:
            sample["host"][str(pid)] = {"error": str(error)}
    try:
        sample["npu_smi"] = subprocess.check_output(["npu-smi", "info"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        sample["npu_error"] = str(error)
    return sample


class MemorySampler:
    def __init__(self, directory, pids):
        self.directory = directory
        self.pids = pids
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with (self.directory / "memory_samples.jsonl").open("w") as output:
            while not self.stop.is_set():
                output.write(json.dumps(memory_sample(self.pids)) + "\n")
                output.flush()
                self.stop.wait(5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=20)


def build_tokens(tokenizer, size):
    marker = "CONTEXT_INSERTION_POINT_481927"
    messages = [
        {"role": "system", "content": "Read the context, answer the final question, then briefly explain your answer."},
        {"role": "user", "content": "Remember this access code: 7391.\n" + marker
         + "\nWhat access code was given at the beginning?"},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    before, after = rendered.split(marker)
    prefix = tokenizer.encode(before, add_special_tokens=False)
    suffix = tokenizer.encode(after, add_special_tokens=False)
    paragraph = " ".join(
        f"Record {index}: the archive describes a coastal weather station, its measured wind, rainfall and maintenance schedule."
        for index in range(128)
    )
    filler = tokenizer.encode(paragraph, add_special_tokens=False)
    missing = size - len(prefix) - len(suffix)
    if missing < 0:
        raise ValueError("Requested prompt is shorter than its framing")
    tokens = prefix + (filler * (missing // len(filler) + 1))[:missing] + suffix
    assert len(tokens) == size
    return tokens


def stream_case(base, directory, tokens, output_tokens, timeout=7200, allow_retractions=False, encoded_payload=None):
    payload = {"input_ids": tokens, "stream": True,
               "sampling_params": {"temperature": 0, "max_new_tokens": output_tokens, "ignore_eos": True}}
    encoded = json.dumps(payload).encode() if encoded_payload is None else encoded_payload
    (directory / "input_metadata.json").write_text(json.dumps({
        "prompt_tokens": len(tokens), "output_tokens": output_tokens,
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "sampling_params": payload["sampling_params"],
    }, indent=2))
    request = urllib.request.Request(base + "/generate", data=encoded,
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = last = None
    previous_count = 0
    first_count = 0
    intervals = []
    final = None
    with (directory / "stream_events.jsonl").open("w") as event_file:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                result = json.loads(data)
                timestamp = time.perf_counter()
                if "error" in result:
                    raise RuntimeError(result["error"])
                metadata = result.get("meta_info", {})
                count = int(metadata.get("completion_tokens", 0))
                event_file.write(json.dumps({"elapsed_s": timestamp - started,
                    "completion_tokens": count, "meta_info": metadata}) + "\n")
                event_file.flush()
                if count > previous_count:
                    if first is None:
                        first, first_count = timestamp, count
                    elif last is not None:
                        intervals.append((timestamp - last) / (count - previous_count))
                    last = timestamp
                    previous_count = count
                final = result
    ended = time.perf_counter()
    if final is None or first is None:
        raise RuntimeError("No generated token was returned")
    metadata = final["meta_info"]
    if metadata.get("prompt_tokens") != len(tokens) or previous_count != output_tokens:
        raise RuntimeError(f"Token-count mismatch: {metadata}")
    if (metadata.get("num_retractions", 0) and not allow_retractions) or metadata.get("finish_reason", {}).get("type") in ("abort", "error"):
        raise RuntimeError(f"Generation did not finish normally: {metadata}")
    (directory / "response.json").write_text(json.dumps(final, ensure_ascii=False))
    summary = {
        "prompt_tokens": len(tokens), "completion_tokens": previous_count,
        "request_started_monotonic_s": started, "request_finished_monotonic_s": ended,
        "first_event_tokens": first_count, "ttft_s": first - started,
        "tpot_ms": 1000 * (last - first) / (previous_count - first_count),
        "e2e_s": ended - started,
        "stream_interval_p50_ms": 1000 * statistics.median(intervals),
        "meta_info": metadata,
    }
    (directory / "timing.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--service-manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, default=33181)
    parser.add_argument("--lengths", type=int, nargs="+", default=[500000, 900000])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--profile-first", action="store_true")
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(Path("/home/j00628475")):
        raise ValueError("Keep benchmark artifacts under /home/j00628475")
    directory.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{args.port}"
    service = json.loads(args.service_manifest.read_text())
    owner = service["processes"]["integrated"]
    parent = owner["pid"]
    process_stat = Path(f"/proc/{parent}/stat").read_text()
    if process_stat.split(")", 1)[1].split()[19] != owner["start_ticks"]:
        raise RuntimeError("Service ownership changed")
    pids = scheduler_pids(parent)
    if len(pids) != 8:
        raise RuntimeError(f"Expected the owned TP8 service, found {pids}")
    info = request_json(base, "/server_info")
    if info["context_length"] < max(args.lengths) + args.output_tokens:
        raise ValueError("Service context is too small")
    if info["max_total_num_tokens"] < max(args.lengths) + args.output_tokens:
        raise ValueError("Actual token pool cannot fit the requested generation")
    (directory / "server_info_before.json").write_text(json.dumps(info, indent=2))
    (directory / "shared_memory_before.txt").write_text(Path("/proc/sysvipc/shm").read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(info["tokenizer_path"], trust_remote_code=True)
    summary = []
    for length in args.lengths:
        tokens = build_tokens(tokenizer, length)
        for repeat in range(args.repeats):
            case = directory / f"tokens_{length}_repeat_{repeat}"
            case.mkdir()
            profiled = args.profile_first and not summary
            if profiled:
                result = request_json(base, "/start_profile", {
                    "output_dir": str(directory / "profile"), "num_steps": 2,
                    "profile_by_stage": True, "activities": ["CPU", "GPU"],
                    "with_stack": False, "record_shapes": True,
                }, timeout=180)
                (case / "profile_start.json").write_text(json.dumps(result))
            (case / "memory_before.json").write_text(json.dumps(memory_sample(pids)))
            print(f"START length={length} repeat={repeat} profiled={profiled}", flush=True)
            with MemorySampler(case, pids):
                result = stream_case(base, case, tokens, args.output_tokens)
            result.update(repeat=repeat, profiled=profiled, warmup=(repeat == 0))
            (case / "memory_after.json").write_text(json.dumps(memory_sample(pids)))
            summary.append(result)
            (directory / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(result), flush=True)
    (directory / "server_info_after.json").write_text(json.dumps(request_json(base, "/server_info"), indent=2))


if __name__ == "__main__":
    main()
