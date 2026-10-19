"""Offline Engine benchmark for complete GLM5Next speculative steps.

Run each arm in a fresh process on idle TP8 hardware. Only Draft planner reuse
differs. Timing includes device completion across all local streams; it is an
isolated worker-step metric, not serving TPOT or a sum of phase timings.
"""

import argparse
import functools
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def accepted_token_rows(tokens, lengths):
    if not lengths or len(tokens) % len(lengths):
        raise ValueError("Invalid speculative output geometry")
    width = len(tokens) // len(lengths)
    if any(length < 1 or length > width for length in lengths):
        raise ValueError("Invalid speculative acceptance length")
    return [
        tokens[row * width : row * width + length] for row, length in enumerate(lengths)
    ]


class StepProbe:
    def __init__(self, synchronize, sink, clock=time.perf_counter_ns):
        self.synchronize = synchronize
        self.sink = sink
        self.clock = clock
        self.active = None
        self.serial = 0

    def phase(self, name, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if self.active is not None:
                self.active["phases"].append(name)
            return function(*args, **kwargs)

        return wrapped

    def graph(self, function):
        @functools.wraps(function)
        def wrapped(worker, runner, *args, **kwargs):
            if self.active is not None:
                phase = self.active["phases"][-1]
                self.active["draft_graph_phases"].append(phase)
            return function(worker, runner, *args, **kwargs)

        return wrapped

    def measure(self, function, worker, batch, *args, **kwargs):
        if not batch.forward_mode.is_decode() or batch.is_extend_in_batch:
            return function(worker, batch, *args, **kwargs)
        if self.active is not None:
            raise RuntimeError("Nested speculative step measurement")
        self.synchronize()
        self.serial += 1
        self.active = dict(
            step=self.serial,
            batch=len(batch.reqs),
            request_ids=[request.rid for request in batch.reqs],
            input_lengths=batch.seq_lens.cpu().tolist(),
            phases=[],
            draft_graph_phases=[],
        )
        try:
            start = self.clock()
            result = function(worker, batch, *args, **kwargs)
            self.synchronize()
            elapsed = self.clock() - start
            record = self.active
            lengths = result.accept_lens.cpu().tolist()
            record.update(
                elapsed_ms=elapsed / 1e6,
                target_graph=bool(result.can_run_cuda_graph),
                accept_lens=lengths,
                accepted_token_ids=accepted_token_rows(
                    result.next_token_ids.cpu().tolist(), lengths
                ),
            )
            self.sink(record)
            return result
        finally:
            self.active = None


def install_probe(output_dir, variant):
    import torch

    from sglang.srt.hardware_backend.npu.hisparse import fsa_backend
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2

    output = Path(output_dir) / f"worker_{os.getpid()}.jsonl"
    stream = output.open("x", buffering=1)

    def emit(record):
        record = dict(record, rank=torch.distributed.get_rank())
        stream.write(json.dumps(record) + "\n")

    probe = StepProbe(
        torch.npu.synchronize, lambda record: emit(dict(kind="step", **record))
    )
    initialize = fsa_backend.initialize_fsa_backend

    @functools.wraps(initialize)
    def initialize_variant(backend, model_runner):
        initialize(backend, model_runner)
        eligible = getattr(backend, "fused_sparse_draft_reuse", False)
        if variant == "baseline":
            backend.fused_sparse_draft_reuse = False
        emit(
            dict(
                kind="backend",
                draft=bool(model_runner.is_draft_worker),
                eligible=bool(eligible),
                enabled=bool(getattr(backend, "fused_sparse_draft_reuse", False)),
                fused=backend.fused_sparse_state is not None,
            )
        )

    fsa_backend.initialize_fsa_backend = initialize_variant
    EagleDraftWorker.draft = probe.phase("draft", EagleDraftWorker.draft)
    EAGLEWorkerV2.verify = probe.phase("verify", EAGLEWorkerV2.verify)
    EagleDraftWorker._draft_extend_for_decode = probe.phase(
        "draft_extend", EagleDraftWorker._draft_extend_for_decode
    )
    EagleDraftWorker._execute_draft_graph = probe.graph(
        EagleDraftWorker._execute_draft_graph
    )
    original = EAGLEWorkerV2.forward_batch_generation

    @functools.wraps(original)
    def measured(worker, batch, *args, **kwargs):
        return probe.measure(original, worker, batch, *args, **kwargs)

    EAGLEWorkerV2.forward_batch_generation = measured


def run_scheduler_with_probe(*args, **kwargs):
    from sglang.srt.managers.scheduler import run_scheduler_process

    install_probe(
        os.environ["MTP_STEP_TEST_OUTPUT"], os.environ["MTP_STEP_TEST_VARIANT"]
    )
    return run_scheduler_process(*args, **kwargs)


def load_records(directory):
    by_rank = {}
    for path in sorted(Path(directory).glob("worker_*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if not records:
            continue
        rank = records[0]["rank"]
        if rank in by_rank:
            raise ValueError(f"Duplicate worker log for rank {rank}")
        if any(record["rank"] != rank for record in records):
            raise ValueError("Mixed ranks in worker log")
        by_rank[rank] = records
    return by_rank


def qualified_steps(by_rank, variant, batch_size, tp_size, warmup, minimum):
    if warmup < 0 or minimum < 1:
        raise ValueError("Warmup must be nonnegative and minimum must be positive")
    if set(by_rank) != set(range(tp_size)):
        raise ValueError("Missing TP rank evidence")
    steps_by_rank = {}
    for rank, records in by_rank.items():
        backends = [record for record in records if record["kind"] == "backend"]
        eligible = [
            record for record in backends if record["draft"] and record["eligible"]
        ]
        if not eligible or not all(record["fused"] for record in backends):
            raise ValueError(f"Rank {rank}: eligible fused Draft backend missing")
        if any(record["enabled"] != (variant == "candidate") for record in eligible):
            raise ValueError("Variant was not installed before graph capture")
        if any(record["enabled"] for record in backends if not record["draft"]):
            raise ValueError("Target backend unexpectedly uses Draft reuse")
        records = [record for record in records if record["kind"] == "step"]
        steps_by_rank[rank] = {record["step"]: record for record in records}
    indices = set(steps_by_rank[0])
    if any(set(steps) != indices for steps in steps_by_rank.values()):
        raise ValueError("TP rank step counts differ")
    full_batch = []
    for index in sorted(indices):
        ranks = [steps_by_rank[rank][index] for rank in range(tp_size)]
        if any(record["batch"] != batch_size for record in ranks):
            continue
        reference = ranks[0]
        for record in ranks:
            if record["phases"] != ["draft", "verify", "draft_extend"]:
                raise ValueError("Incomplete speculative step")
            for key in ("input_lengths", "accept_lens", "accepted_token_ids"):
                if record[key] != reference[key]:
                    raise ValueError(f"TP rank mismatch: {key}")
        full_batch.append(ranks)
    selected = full_batch[warmup:]
    if len(selected) < minimum:
        raise ValueError(f"Only {len(selected)} full-batch steps; require {minimum}")
    for ranks in selected:
        for record in ranks:
            if not record["target_graph"] or record["draft_graph_phases"] != [
                "draft",
                "draft_extend",
            ]:
                raise ValueError("A measured phase did not execute its Graph")
            if any(length < 1 or length > 6 for length in record["accept_lens"]):
                raise ValueError("Invalid acceptance length for MTP5/draft6")
    if not any(length > 1 for ranks in selected for length in ranks[0]["accept_lens"]):
        raise ValueError("Zero-acceptance run is not a qualified workload")
    return selected


def compare_runs(baseline_dir, candidate_dir, warmup=3, minimum=10):
    manifests = [
        json.loads((Path(directory) / "manifest.json").read_text())
        for directory in (baseline_dir, candidate_dir)
    ]
    for key in (
        "source_hashes",
        "server_cli",
        "input_sha256",
        "batch_size",
        "output_len",
        "environment",
    ):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Unmatched A/B configuration: {key}")
    outputs = [
        json.loads((Path(directory) / "outputs.json").read_text())
        for directory in (baseline_dir, candidate_dir)
    ]
    if [row["output_ids"] for row in outputs[0]] != [
        row["output_ids"] for row in outputs[1]
    ]:
        raise ValueError("A/B generated token IDs differ")
    batch_size = manifests[0]["batch_size"]
    chains = []
    for directory, variant in (
        (baseline_dir, "baseline"),
        (candidate_dir, "candidate"),
    ):
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        if manifest["variant"] != variant:
            raise ValueError("Incorrect A/B variant")
        chains.append(
            qualified_steps(
                load_records(directory), variant, batch_size, 8, warmup, minimum
            )
        )
    if len(chains[0]) != len(chains[1]):
        raise ValueError("A/B full-batch step counts differ")
    samples = []
    for original, updated in zip(*chains):
        for key in ("input_lengths", "accept_lens", "accepted_token_ids"):
            if original[0][key] != updated[0][key]:
                raise ValueError(f"A/B step workloads differ: {key}")
        samples.append(
            dict(
                baseline_ms=max(record["elapsed_ms"] for record in original),
                candidate_ms=max(record["elapsed_ms"] for record in updated),
                accepted_tokens=sum(original[0]["accept_lens"]),
            )
        )
    baseline = statistics.median(sample["baseline_ms"] for sample in samples)
    candidate = statistics.median(sample["candidate_ms"] for sample in samples)
    return dict(
        scope="Complete synchronized TP8 worker MTP step; not serving TPOT",
        batch_size=batch_size,
        steps=len(samples),
        baseline_ms=baseline,
        candidate_ms=candidate,
        saved_ms=baseline - candidate,
        reduction_percent=(baseline - candidate) / baseline * 100,
        samples=samples,
    )


def require_idle_tp8():
    report = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
    if report.count("No running processes found in NPU") != 8:
        raise RuntimeError("This full-model benchmark requires all eight NPUs idle")
    return report


def source_hashes():
    import sglang

    root = Path(sglang.__file__).resolve().parent
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "srt").rglob("*.py"))
    }
    hashes["__benchmark__"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def run(args, server_cli):
    for name in (
        "SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD",
        "SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION",
    ):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1 in both variants")
    input_bytes = args.inputs.read_bytes()
    inputs = json.loads(input_bytes)
    if len(inputs) != args.batch_size or not all(
        isinstance(row, list)
        and row
        and all(isinstance(token, int) and token >= 0 for token in row)
        for row in inputs
    ):
        raise ValueError(
            "Inputs must contain exactly batch_size nonempty token-ID lists"
        )
    idle_report = require_idle_tp8()
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import prepare_server_args

    config = prepare_server_args(server_cli)
    required = dict(
        device="npu",
        tp_size=8,
        dp_size=1,
        nnodes=1,
        pp_size=1,
        max_running_requests=args.batch_size,
        speculative_num_steps=5,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=6,
        speculative_algorithm="NEXTN",
    )
    for name, expected in required.items():
        if getattr(config, name) != expected:
            raise ValueError(f"Expected {name}={expected}")
    if max(map(len, inputs)) + args.output_len > config.context_length:
        raise ValueError("Context length must include input and generated output")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "occupancy_before.txt").write_text(idle_report)
    selected_env = {
        name: value
        for name, value in os.environ.items()
        if name.startswith(
            ("SGLANG_NPU_", "SGLANG_ENABLE_", "HCCL_", "ASCEND_RT_VISIBLE")
        )
    }
    manifest = dict(
        variant=args.variant,
        batch_size=args.batch_size,
        output_len=args.output_len,
        input_sha256=hashlib.sha256(input_bytes).hexdigest(),
        server_cli=server_cli,
        source_hashes=source_hashes(),
        environment=selected_env,
        input_lengths=list(map(len, inputs)),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    os.environ["MTP_STEP_TEST_OUTPUT"] = str(output)
    os.environ["MTP_STEP_TEST_VARIANT"] = args.variant
    Engine.run_scheduler_process_func = staticmethod(run_scheduler_with_probe)
    engine = Engine(server_args=config)
    try:
        results = engine.generate(
            input_ids=inputs,
            sampling_params=dict(
                temperature=0, max_new_tokens=args.output_len, ignore_eos=True
            ),
        )
        if len(results) != args.batch_size or any(
            len(result.get("output_ids", [])) != args.output_len for result in results
        ):
            raise RuntimeError("Incomplete generation or missing output_ids")
        (output / "outputs.json").write_text(json.dumps(results, indent=2))
    finally:
        engine.shutdown()
    qualified_steps(
        load_records(output),
        args.variant,
        args.batch_size,
        8,
        args.warmup,
        args.minimum,
    )
    (output / "validated.json").write_text(json.dumps(dict(status="PASS")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch = subparsers.add_parser("run")
    launch.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    launch.add_argument("--inputs", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--batch-size", type=int, choices=(1, 10), required=True)
    launch.add_argument("--output-len", type=int, default=128)
    launch.add_argument("--warmup", type=int, default=3)
    launch.add_argument("--minimum", type=int, default=10)
    launch.add_argument("server_cli", nargs=argparse.REMAINDER)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        report = compare_runs(args.baseline, args.candidate)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    else:
        server_cli = (
            args.server_cli[1:] if args.server_cli[:1] == ["--"] else args.server_cli
        )
        run(args, server_cli)


if __name__ == "__main__":
    main()
