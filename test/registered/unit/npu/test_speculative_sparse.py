"""MTP admission and mutable suffix bounds; no device execution."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).parents[4] / "python/sglang/srt/hardware_backend/npu/speculative_sparse.py"
SPEC = importlib.util.spec_from_file_location("speculative_sparse_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config(name="EAGLE", **changes):
    algorithm = SimpleNamespace(name=name, is_none=lambda: name == "NONE")
    spec = SimpleNamespace(speculative_eagle_topk=1, speculative_num_steps=3,
                           speculative_num_draft_tokens=4, enable_multi_layer_eagle=False)
    spec.__dict__.update(changes)
    return algorithm, spec, SimpleNamespace(num_nextn_predict_layers=1)


@pytest.mark.parametrize("name", ["EAGLE", "EAGLE3"])
def test_chain_mtp_bounds_cover_target_verify_and_draft_steps(name):
    arguments = config(name)
    assert MODULE.sparse_mtp_tail(*arguments) == 8
    assert MODULE.sparse_mtp_rows(*arguments) == 4


def test_non_speculative_path_has_one_query_row_and_no_mutable_tail():
    arguments = config("NONE")
    assert MODULE.sparse_mtp_tail(*arguments) == 0
    assert MODULE.sparse_mtp_rows(*arguments) == 1


@pytest.mark.parametrize("changes", [
    {"speculative_eagle_topk": 2}, {"speculative_num_steps": 0},
    {"speculative_num_draft_tokens": 5}, {"enable_multi_layer_eagle": True},
    {"speculative_adaptive": True}, {"speculative_draft_window_size": 4096},
])
def test_non_chain_layouts_do_not_enter_fused_mtp(changes):
    with pytest.raises(ValueError):
        MODULE.sparse_mtp_tail(*config(**changes))


@pytest.mark.parametrize("name", ["FROZEN_KV_MTP", "DSPARK", "DFLASH", "STANDALONE"])
def test_other_speculative_algorithms_require_their_own_cache_contract(name):
    with pytest.raises(ValueError, match="chains"):
        MODULE.sparse_mtp_tail(*config(name))


def test_target_sizing_requires_actual_draft_inventory():
    values = SimpleNamespace(spec_algorithm=config()[0], is_draft_worker=False,
                             spec_aux_config=SimpleNamespace(eagle_draft_num_layers=1))
    assert MODULE.sparse_mtp_draft_layers(values) == 1
    values.spec_aux_config.eagle_draft_num_layers = None
    with pytest.raises(ValueError, match="layer count"):
        MODULE.sparse_mtp_draft_layers(values)
    values.is_draft_worker = True
    assert MODULE.sparse_mtp_draft_layers(values) == 0


def test_prefill_draft_cannot_capture_bounded_slots_outside_an_active_batch():
    source = SOURCE.parents[2] / "speculative/eagle_worker_v2.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(module)
                  if isinstance(node, ast.FunctionDef) and node.name == "_capture_cuda_graphs")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    worker = SimpleNamespace(draft_runner=SimpleNamespace(prefill_offload_runtime=object()),
                             cuda_graph_runner=object(), cuda_graph_runner_for_draft_extend=object())
    namespace["_capture_cuda_graphs"](worker)
    assert worker.cuda_graph_runner is None
    assert worker.cuda_graph_runner_for_draft_extend is None
