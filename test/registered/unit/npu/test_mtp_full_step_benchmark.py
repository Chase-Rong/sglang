"""CPU contracts for the full-step NPU benchmark; no model timing claims."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[3] / "manual/npu/bench_mtp_full_step.py"
SPEC = importlib.util.spec_from_file_location("mtp_full_step_benchmark", PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class HostValue:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


def test_timing_includes_three_phases_and_final_completion():
    timeline, records = [], []
    now = [0]

    def synchronize():
        timeline.append("sync")
        now[0] += 10_000_000

    probe = BENCH.StepProbe(synchronize, records.append, lambda: now[0])
    result = SimpleNamespace(
        can_run_cuda_graph=True,
        accept_lens=HostValue([4]),
        next_token_ids=HostValue([1, 2, 3, 4]),
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        is_extend_in_batch=False,
        reqs=[SimpleNamespace(rid="request")],
        seq_lens=HostValue([8192]),
    )

    def phase(name):
        timeline.append(name)
        now[0] += 1_000_000
        return result

    def full_step(worker, batch):
        for name in ("draft", "verify", "draft_extend"):
            probe.phase(name, lambda: phase(name))()
        return result

    assert probe.measure(full_step, None, batch) is result
    assert timeline == ["sync", "draft", "verify", "draft_extend", "sync"]
    assert records[0]["elapsed_ms"] == 13
    assert records[0]["phases"] == ["draft", "verify", "draft_extend"]
    assert probe.active is None


def test_failed_step_is_not_recorded_and_scope_is_restored():
    records = []
    probe = BENCH.StepProbe(lambda: None, records.append)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        is_extend_in_batch=False,
        reqs=[],
        seq_lens=HostValue([]),
    )

    def fail(*args):
        raise RuntimeError("failed inference")

    with pytest.raises(RuntimeError, match="failed inference"):
        probe.measure(fail, None, batch)
    assert not records
    assert probe.active is None


def test_prefill_bypasses_timer():
    probe = BENCH.StepProbe(
        lambda: pytest.fail("synchronized prefill"), lambda record: None
    )
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False))
    assert probe.measure(lambda *args: "prefill", None, batch) == "prefill"


def test_graph_observer_records_execution_not_capability():
    probe = BENCH.StepProbe(lambda: None, lambda record: None)
    wrapped = probe.graph(lambda worker, runner, batch: "executed")
    assert wrapped(None, None, None) == "executed"
    probe.active = dict(phases=["draft"], draft_graph_phases=[])
    wrapped(None, None, None)
    assert probe.active["draft_graph_phases"] == ["draft"]


def rank_records(variant="candidate", batch_size=10):
    return {
        rank: [
            dict(
                kind="backend",
                rank=rank,
                draft=True,
                eligible=True,
                enabled=variant == "candidate",
                fused=True,
            ),
            dict(
                kind="backend",
                rank=rank,
                draft=False,
                eligible=False,
                enabled=False,
                fused=True,
            ),
            *[
                dict(
                    kind="step",
                    rank=rank,
                    step=step,
                    batch=batch_size,
                    phases=["draft", "verify", "draft_extend"],
                    draft_graph_phases=["draft", "draft_extend"],
                    target_graph=True,
                    elapsed_ms=20 + rank,
                    input_lengths=[8192 + 4 * step] * batch_size,
                    accept_lens=[4] * batch_size,
                    accepted_token_ids=[[1, 2, 3, 4]] * batch_size,
                )
                for step in range(15)
            ],
        ]
        for rank in range(8)
    }


def test_qualification_discards_warmup_and_requires_actual_batch():
    records = rank_records()
    for rows in records.values():
        rows[2]["batch"] = 1
    assert len(BENCH.qualified_steps(records, "candidate", 10, 8, 3, 10)) == 11
    with pytest.raises(ValueError, match="full-batch"):
        BENCH.qualified_steps(rank_records(batch_size=1), "candidate", 10, 8, 3, 10)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_rank",
        "target_reuse",
        "disabled",
        "graph",
        "phase",
        "acceptance",
        "rank_tokens",
    ],
)
def test_qualification_rejects_invalid_evidence(failure):
    records = rank_records()
    if failure == "missing_rank":
        del records[7]
    elif failure == "target_reuse":
        records[0][1]["enabled"] = True
    elif failure == "disabled":
        records[0][0]["enabled"] = False
    elif failure == "graph":
        records[0][-1]["draft_graph_phases"] = ["draft"]
    elif failure == "phase":
        records[0][-1]["phases"] = ["draft", "verify"]
    elif failure == "acceptance":
        for rows in records.values():
            for row in rows[2:]:
                row["accept_lens"] = [1] * 10
    else:
        records[0][-1]["accepted_token_ids"] = [[99] * 4] * 10
    with pytest.raises(ValueError):
        BENCH.qualified_steps(records, "candidate", 10, 8, 3, 10)


def write_run(directory, variant):
    directory.mkdir()
    manifest = dict(
        source_hashes={"test.py": "hash"},
        server_cli=["--tp-size", "8"],
        input_sha256="input",
        batch_size=10,
        output_len=128,
        environment={},
        variant=variant,
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "outputs.json").write_text(json.dumps([dict(output_ids=[1, 2])]))
    records = rank_records(variant)
    for rank, rows in records.items():
        if variant == "candidate":
            for row in rows[2:]:
                row["elapsed_ms"] -= 2
        (directory / f"worker_{rank}.jsonl").write_text(
            "\n".join(map(json.dumps, rows))
        )


def test_comparison_uses_slowest_rank_and_does_not_sum_phases(tmp_path):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    write_run(baseline, "baseline")
    write_run(candidate, "candidate")
    report = BENCH.compare_runs(baseline, candidate)
    assert report["steps"] == 12
    assert report["baseline_ms"] == 27
    assert report["candidate_ms"] == 25
    assert report["saved_ms"] == 2
    assert report["reduction_percent"] == pytest.approx(200 / 27)


@pytest.mark.parametrize("failure", ["config", "outputs", "workload"])
def test_comparison_rejects_unmatched_workloads(tmp_path, failure):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    write_run(baseline, "baseline")
    write_run(candidate, "candidate")
    if failure == "config":
        path = candidate / "manifest.json"
        data = json.loads(path.read_text())
        data["input_sha256"] = "other input"
        path.write_text(json.dumps(data))
    elif failure == "outputs":
        (candidate / "outputs.json").write_text(json.dumps([dict(output_ids=[3, 4])]))
    else:
        for path in candidate.glob("worker_*.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[-1]["input_lengths"] = [12345] * 10
            path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError):
        BENCH.compare_runs(baseline, candidate)


def test_idle_gate_does_not_accept_free_memory_as_idle(monkeypatch):
    monkeypatch.setattr(
        BENCH.subprocess, "check_output", lambda *args, **kwargs: "98% util, 50GiB free"
    )
    with pytest.raises(RuntimeError, match="idle"):
        BENCH.require_idle_tp8()


def test_rejected_tail_is_not_compared_as_accepted_output():
    assert BENCH.accepted_token_rows([10, 11, 999, 20, 999, 999], [2, 1]) == [
        [10, 11],
        [20],
    ]
    with pytest.raises(ValueError):
        BENCH.accepted_token_rows([1, 2], [3])


def test_normal_run_keeps_individual_zero_acceptance_steps():
    records = rank_records()
    for rows in records.values():
        rows[-1]["accept_lens"] = [1] * 10
        rows[-1]["accepted_token_ids"] = [[1]] * 10
    assert len(BENCH.qualified_steps(records, "candidate", 10, 8, 3, 10)) == 12
