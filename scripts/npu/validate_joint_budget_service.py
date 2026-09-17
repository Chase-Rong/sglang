"""Check owned-service capacity and real concurrent generation after joint sizing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--first-index", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--output-prefix", default="joint")
    parser.add_argument("--expected-requests", type=int)
    parser.add_argument("--minimum-tokens", type=int, default=1048576)
    parser.add_argument("--minimum-context", type=int, default=1048576)
    parser.add_argument("--maximum-tokens", type=int)
    args = parser.parse_args()
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix or args.output_prefix in (".", ".."):
        raise ValueError("Output prefix must be a single file-name component")
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    owner = manifest["processes"]["integrated"]
    stat = Path(f"/proc/{owner['pid']}/stat").read_text()
    if stat.split(")", 1)[1].split()[19] != owner["start_ticks"]:
        raise RuntimeError("Service ownership changed")
    base = f"http://127.0.0.1:{args.port}"

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        query = urllib.request.Request(base + path, data=data,
                                       headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(query, timeout=1200) as response:
            return json.load(response)

    info = request("/server_info")
    (args.run_dir / f"{args.output_prefix}_server_info.json").write_text(json.dumps(info, indent=2))
    if info["context_length"] < args.minimum_context:
        raise RuntimeError("Single-request context is below the expected minimum")
    if info["max_total_num_tokens"] < args.minimum_tokens:
        raise RuntimeError("Shared token capacity is below the expected minimum")
    if args.maximum_tokens is not None and info["max_total_num_tokens"] > args.maximum_tokens:
        raise RuntimeError("Token capacity exceeds the expected maximum")
    if args.expected_requests is not None:
        for state in info["internal_states"]:
            if state["effective_max_running_requests_per_dp"] != args.expected_requests:
                raise RuntimeError("Configured request capacity was not preserved")
        expected_input_limit = min(info["context_length"], info["max_total_num_tokens"]) - 6
        if info["max_req_input_len"] != expected_input_limit:
            raise RuntimeError("Single-request limit does not match the shared pool and model context")

    def chat(index):
        expected = str(7391 + index)
        payload = {
            "model": "glm53flash", "temperature": 0, "max_completion_tokens": args.max_output_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"Reply with exactly {expected}, no other text."}],
        }
        (args.run_dir / f"{args.output_prefix}_request_{index}.json").write_text(json.dumps(payload))
        started = time.monotonic()
        response = request("/v1/chat/completions", payload)
        (args.run_dir / f"{args.output_prefix}_response_{index}.json").write_text(json.dumps(response))
        content = response["choices"][0]["message"].get("content") or ""
        answer = content.rsplit("</think>", 1)[-1].strip()
        return dict(index=index, expected=expected, actual=answer,
                    matches=answer == expected, elapsed_s=time.monotonic() - started,
                    finish_reason=response["choices"][0].get("finish_reason"),
                    max_output_tokens=args.max_output_tokens)

    records = [chat(args.first_index)]
    if args.concurrency:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            records += list(executor.map(chat, range(args.first_index + 1, args.first_index + args.concurrency + 1)))
    (args.run_dir / f"{args.output_prefix}_generation_summary.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records), flush=True)
    if not all(record["matches"] for record in records):
        raise RuntimeError("Semantic generation check failed")


if __name__ == "__main__":
    main()
