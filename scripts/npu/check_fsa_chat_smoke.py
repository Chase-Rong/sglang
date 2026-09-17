"""Check basic chat-template responses after integrated service bring-up."""

import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=33181)
    args = parser.parse_args()
    results = []
    for index, expected in enumerate(("5", "7391")):
        prompt = json.loads((args.run_dir / f"request_{index}.json").read_text())["text"]
        payload = {
            "model": "glm53flash", "temperature": 0, "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": "Answer the user's question with only the requested number."},
                         {"role": "user", "content": prompt}],
        }
        (args.run_dir / f"chat_request_{index}.json").write_text(json.dumps(payload))
        request = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
        (args.run_dir / f"chat_response_{index}.json").write_text(json.dumps(result, ensure_ascii=False))
        message = result["choices"][0]["message"]
        text = message.get("content") or ""
        final_text = text.rsplit("</think>", 1)[-1].strip()
        record = {"case": index, "elapsed_s": time.monotonic() - started,
                  "expected": expected, "text": text, "final_text": final_text,
                  "matches": final_text == expected,
                  "usage": result.get("usage")}
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    (args.run_dir / "chat_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if not all(record["matches"] for record in results):
        raise RuntimeError("Chat semantic smoke did not match all expected answers")


if __name__ == "__main__":
    main()
