"""Regression for continuing Prefill rows at the configured request limit."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def scheduler():
    source = Path(__file__).parents[4] / "python/sglang/srt/managers/scheduler.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definition = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    method = next(node for node in definition.body
                  if isinstance(node, ast.FunctionDef) and node.name == "get_num_allocatable_reqs")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"get_parallel": lambda: SimpleNamespace(pp_max_micro_batch_size=8)}
    exec(compile(module, str(source), "exec"), namespace)
    instance = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(available_size=lambda: 1),
        beam_coordinator=SimpleNamespace(pending_member_rows=lambda batch: 0),
        running_batch=object(),
    )
    return instance, namespace["get_num_allocatable_reqs"]


def test_last_chunk_does_not_charge_its_existing_slot_twice(scheduler):
    instance, capacity = scheduler
    continuing = SimpleNamespace(kv=SimpleNamespace(holds_kv=True))
    admitted = [continuing]
    assert len(admitted) < capacity(instance, 6, prefill_reqs=admitted)


def test_new_prefill_request_still_consumes_the_last_free_slot(scheduler):
    instance, capacity = scheduler
    admitted = [SimpleNamespace(kv=SimpleNamespace(holds_kv=False))]
    assert len(admitted) >= capacity(instance, 6, prefill_reqs=admitted)


def test_continuing_request_still_counts_toward_running_limit(scheduler):
    instance, capacity = scheduler
    admitted = [SimpleNamespace(kv=SimpleNamespace(holds_kv=True))]
    assert len(admitted) >= capacity(instance, 7, prefill_reqs=admitted)


def test_no_new_slot_cannot_admit_another_request(scheduler):
    instance, capacity = scheduler
    instance.req_to_token_pool.available_size = lambda: 0
    admitted = [SimpleNamespace(kv=SimpleNamespace(holds_kv=True))]
    assert len(admitted) >= capacity(instance, 6, prefill_reqs=admitted)


def test_beam_width_applies_to_new_slots_only(scheduler):
    instance, capacity = scheduler
    admitted = [SimpleNamespace(kv=SimpleNamespace(holds_kv=True))]
    assert len(admitted) >= capacity(instance, 6, beam_width=2, prefill_reqs=admitted)
    instance.req_to_token_pool.available_size = lambda: 2
    assert len(admitted) < capacity(instance, 6, beam_width=2, prefill_reqs=admitted)


def test_pending_beam_rows_remain_reserved(scheduler):
    instance, capacity = scheduler
    instance.beam_coordinator.pending_member_rows = lambda batch: 1
    admitted = [SimpleNamespace(kv=SimpleNamespace(holds_kv=True))]
    assert len(admitted) >= capacity(instance, 6, prefill_reqs=admitted)
