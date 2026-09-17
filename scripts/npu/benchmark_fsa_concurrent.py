"""Replay simultaneous long requests against an owned integrated TP8 service."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import threading
import time

from benchmark_fsa_long_context import (
    MemorySampler, build_tokens, request_json, scheduler_pids, stream_case,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--service-manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=1000000)
    parser.add_argument("--output-tokens", type=int, default=1024)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(Path("/home/j00628475")):
        raise ValueError("Keep artifacts in the assigned work directory")
    directory.mkdir(parents=True, exist_ok=False)
    owner = json.loads(args.service_manifest.read_text())["processes"]["integrated"]
    stat = Path(f"/proc/{owner['pid']}/stat").read_text()
    if stat.split(")", 1)[1].split()[19] != owner["start_ticks"]:
        raise RuntimeError("Service ownership changed")
    pids = scheduler_pids(owner["pid"])
    if len(pids) != 8:
        raise RuntimeError("Expected an owned TP8 service")
    base = f"http://127.0.0.1:{args.port}"
    info = request_json(base, "/server_info")
    (directory / "server_info_before.json").write_text(json.dumps(info, indent=2))
    capacity = min(state["effective_max_running_requests_per_dp"] for state in info["internal_states"])
    needed = args.input_tokens + args.output_tokens
    page = info["page_size"]
    if capacity != args.requests or info["max_total_tokens"] is not None:
        raise RuntimeError("Requested capacity or omitted total-token cap was not preserved")
    if info["context_length"] < needed or info["max_total_num_tokens"] < args.requests * ((needed + page - 1) // page * page):
        raise RuntimeError("Resolved pool cannot hold all requested long sequences")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(info["tokenizer_path"], trust_remote_code=True)
    tokens = build_tokens(tokenizer, args.input_tokens)
    barrier = threading.Barrier(args.requests)
    records = []
    errors = []
    log_path = args.service_manifest.parent / "server.log"
    log_offset = log_path.stat().st_size

    def execute(index):
        case = directory / f"request_{index}"
        case.mkdir()
        barrier.wait(timeout=120)
        return {"request": index, **stream_case(base, case, tokens, args.output_tokens)}

    started = time.monotonic()
    with MemorySampler(directory, pids), ThreadPoolExecutor(max_workers=args.requests) as executor:
        futures = {executor.submit(execute, index): index for index in range(args.requests)}
        for future in as_completed(futures):
            try:
                record = future.result()
                records.append(record)
                print(json.dumps(record), flush=True)
            except Exception as error:
                errors.append({"request": futures[future], "error": repr(error)})
            (directory / "progress.json").write_text(json.dumps({"completed": records, "errors": errors}, indent=2))
    elapsed = time.monotonic() - started
    with log_path.open("rb") as source:
        source.seek(log_offset)
        log = source.read().decode(errors="replace")
    active_counts = [int(value) for value in re.findall(r"Decode batch, #running-req: (\d+)", log)]
    full_tokens = [int(value) for value in re.findall(r"Decode batch, #running-req: \d+, #full token: (\d+)", log)]
    concurrency_evidence = [line for line in log.splitlines()
                            if (match := re.search(r"Decode batch, #running-req: (\d+), #full token: (\d+)", line))
                            and int(match.group(1)) == args.requests
                            and int(match.group(2)) >= args.requests * args.input_tokens]
    summary = dict(requests=sorted(records, key=lambda record: record["request"]), errors=errors,
                   elapsed_s=elapsed, aggregate_output_tokens_per_s=sum(record["completion_tokens"] for record in records)/elapsed,
                   max_decode_running_requests=max(active_counts, default=0),
                   max_decode_full_tokens=max(full_tokens, default=0),
                   simultaneous_full_context_evidence=concurrency_evidence)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    (directory / "server_info_after.json").write_text(json.dumps(request_json(base, "/server_info"), indent=2))
    print(json.dumps(summary), flush=True)
    if errors or len(records) != args.requests:
        raise RuntimeError("Concurrent generation did not complete every request")
    if summary["max_decode_running_requests"] < args.requests:
        raise RuntimeError("Did not observe the requested simultaneous Decode concurrency")
    if not concurrency_evidence:
        raise RuntimeError("Did not observe all full contexts resident in the same Decode batch")


if __name__ == "__main__":
    main()
