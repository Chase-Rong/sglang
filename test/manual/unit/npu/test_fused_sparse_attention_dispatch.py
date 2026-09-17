import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "hardware_backend"
    / "npu"
    / "fused_sparse_attention.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "fused_attention_under_test", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
FusedSparseAttentionUnavailable = _MODULE.FusedSparseAttentionUnavailable
FusedSparseAttentionInputs = _MODULE.FusedSparseAttentionInputs
FusedSparseAttentionCacheManager = _MODULE.FusedSparseAttentionCacheManager
resolve_op = _MODULE._resolve_fused_op


def test_reference_planner_reuses_only_matching_group_and_inputs():
    planner = _MODULE.ReferenceSparsePlanner()
    args = dict(
        topk_indices=torch.tensor([[0, 3]], dtype=torch.int32),
        block_table=torch.tensor([[10, 11]], dtype=torch.int32),
        kv_lengths=torch.tensor([8], dtype=torch.int32),
        query_lengths=torch.tensor([1], dtype=torch.int32),
        page_size=4,
        request_generation=torch.tensor([2], dtype=torch.int64),
    )
    first = planner.plan("group-0", **args)
    second = planner.plan("group-0", **args)
    assert first is second
    other_group = planner.plan("group-1", **args)
    assert other_group is not first
    changed = dict(args)
    changed["request_generation"] = torch.tensor([3], dtype=torch.int64)
    assert planner.plan("group-0", **changed) is not first


def test_reference_plan_keeps_logical_ids_and_marks_resident_misses():
    plan = _MODULE.build_reference_sparse_plan(
        torch.tensor([[0, 3], [4, -1]], dtype=torch.int32),
        torch.tensor([[10, 11, 12]], dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        page_size=4,
        resident_token_ids=torch.tensor([0, 4]),
    )
    assert plan.logical_topk.shape == (2, 1, 2)
    assert plan.physical_token_ids[:, 0].tolist() == [[40, 43], [44, -1]]
    assert plan.miss_mask[:, 0].tolist() == [[False, True], [False, False]]


def test_mtp_rows_have_independent_causal_limits_and_padding():
    topk = torch.tensor([[[3, 4, 5, 6]]] * 6 + [[[0, 1, 2, 3]]], dtype=torch.int32)
    requests = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    query_lengths = torch.tensor([3, 6], dtype=torch.int32)
    kv_lengths = torch.tensor([7, 6], dtype=torch.int32)
    masked = _MODULE.mask_causal_topk(topk, requests, query_lengths, kv_lengths)
    assert masked[:6, 0].tolist() == [
        [3, 4, -1, -1],
        [3, 4, 5, -1],
        [3, 4, 5, 6],
        [3, -1, -1, -1],
        [3, 4, -1, -1],
        [3, 4, 5, -1],
    ]
    positions = torch.tensor([4, 5, 6, 3, 4, 5, -1])
    masked = _MODULE.mask_causal_topk(
        topk, requests, query_lengths, kv_lengths, positions=positions
    )
    assert torch.all(masked[-1] == -1)


def test_mtp_selection_requires_external_validity_tracking():
    pool = SimpleNamespace(req_generation=torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="external planner"):
        _MODULE.FusedSparseAttentionState("cpu", pool, mutable_tail_tokens=4)


def test_missing_fused_binary_fails_explicitly(monkeypatch):
    monkeypatch.setattr(torch, "ops", SimpleNamespace())
    with pytest.raises(FusedSparseAttentionUnavailable, match="not loaded"):
        resolve_op()


def test_dispatch_rejects_non_npu_before_op_lookup():
    with pytest.raises(ValueError, match="requires NPU"):
        _MODULE.fused_sparse_attention_overlap(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int16),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            None,
            scale_value=1.0,
        )


def test_input_abi_defaults_are_stable():
    fields = FusedSparseAttentionInputs.__dataclass_fields__
    assert fields["sparse_block_size"].default == 1
    assert fields["selection_topk_block_size"].default == 1
    assert fields["layout_kv"].default == "PA_BSND"


def test_cache_manager_normalizes_two_dimensional_topk():
    manager = FusedSparseAttentionCacheManager(torch.device("cpu"))
    topk = torch.tensor([[0, 3, 7]], dtype=torch.int32)
    full_rope = torch.empty((8, 64, 64), dtype=torch.bfloat16)
    full_kv = torch.empty((8, 64, 512), dtype=torch.bfloat16)

    cache, normalized = manager.ensure(topk, full_rope, full_kv)

    assert normalized.shape == (1, 1, 3)
    assert cache.max_heads == 1
    assert cache.max_topk == 3


def test_cache_manager_reuses_state_and_invalidates_changed_owner():
    manager = FusedSparseAttentionCacheManager(torch.device("cpu"))
    topk = torch.tensor([[0, 3], [1, 4]], dtype=torch.int32)
    full_rope = torch.empty((8, 64, 64), dtype=torch.bfloat16)
    full_kv = torch.empty((8, 64, 512), dtype=torch.bfloat16)

    cache, normalized = manager.ensure(topk, full_rope, full_kv)
    _, _, _, status, membership = manager.views(cache, normalized)
    owners = torch.tensor([10, 20], dtype=torch.int32)
    manager.invalidate_changed_rows(cache, status, membership, owners, 2, 1)
    status[0, 0, 0] = 9

    reused, _ = manager.ensure(topk, full_rope, full_kv)
    assert reused is cache
    manager.invalidate_changed_rows(cache, status, membership, owners, 2, 1)
    assert status[0, 0, 0].item() == 9

    changed_owners = torch.tensor([11, 20], dtype=torch.int32)
    manager.invalidate_changed_rows(cache, status, membership, changed_owners, 2, 1)
    assert status[0, 0, 0].item() == -1


def test_cache_manager_reset_drops_persistent_state():
    manager = FusedSparseAttentionCacheManager(torch.device("cpu"))
    topk = torch.zeros((1, 2), dtype=torch.int32)
    full_rope = torch.empty((2, 64, 64), dtype=torch.bfloat16)
    full_kv = torch.empty((2, 64, 512), dtype=torch.bfloat16)
    manager.ensure(topk, full_rope, full_kv)

    manager.reset()

    assert manager.cache is None


def make_state_and_inputs():
    pool = SimpleNamespace(req_generation=torch.tensor([0, 1, 1], dtype=torch.int64))
    state = _MODULE.FusedSparseAttentionState("cpu", pool)
    full_kv = torch.empty((8, 64, 1, 512), dtype=torch.bfloat16)
    full_rope = torch.empty((8, 64, 1, 64), dtype=torch.bfloat16)
    inputs = dict(
        layer_id=0,
        q_nope=torch.zeros((2, 8, 512), dtype=torch.bfloat16),
        q_rope=torch.zeros((2, 8, 64), dtype=torch.bfloat16),
        topk_indices=torch.tensor([[1, 65], [0, 70]], dtype=torch.int32),
        full_kv=_MODULE.paged_kv_view(full_kv, 64, 512),
        full_rope=_MODULE.paged_kv_view(full_rope, 64, 64),
        block_table=torch.tensor([[3, 7], [2, 4]], dtype=torch.int32),
        kv_lengths=torch.tensor([100, 120], dtype=torch.int32),
        query_lengths=torch.tensor([1, 2], dtype=torch.int32),
        request_slots=torch.tensor([1, 2], dtype=torch.int32),
        scale=0.125,
    )
    return state, inputs


def test_builder_preserves_logical_topk_and_zero_copy_backing():
    state, arguments = make_state_and_inputs()
    inputs = state.build_inputs(**arguments)
    torch.testing.assert_close(
        inputs.selection_topk_indices[:, 0], arguments["topk_indices"]
    )
    torch.testing.assert_close(inputs.full_kv_block_table, arguments["block_table"])
    assert inputs.full_q_actual_seq.tolist() == [1, 2]
    assert inputs.full_kv_cache.data_ptr() == arguments["full_kv"].data_ptr()
    assert inputs.full_k_rope.data_ptr() == arguments["full_rope"].data_ptr()


def test_layer_state_is_not_shared_even_with_shared_topk():
    state, arguments = make_state_and_inputs()
    first = state.build_inputs(**arguments)
    first.selection_kv_block_status.fill_(42)
    second = state.build_inputs(**dict(arguments, layer_id=1))
    assert second.selection_kv_cache.data_ptr() != first.selection_kv_cache.data_ptr()
    assert torch.all(second.selection_kv_block_status == -1)
    assert torch.all(state.build_inputs(**arguments).selection_kv_block_status == 42)


def test_reallocated_request_slot_invalidates_selection():
    state, arguments = make_state_and_inputs()
    inputs = state.build_inputs(**arguments)
    inputs.selection_kv_block_status.fill_(42)
    state.pool.req_generation[1] += 1
    state.refresh_generations()
    updated = state.build_inputs(**arguments)
    assert torch.all(updated.selection_kv_block_status[0] == -1)
    assert torch.all(updated.selection_kv_block_status[1] == 42)


def test_pool_generation_reset_changes_epoch():
    state, arguments = make_state_and_inputs()
    inputs = state.build_inputs(**arguments)
    inputs.selection_kv_block_status.fill_(42)
    state.pool.req_generation.zero_()
    state.refresh_generations()
    state.pool.req_generation[1:] = 1
    state.refresh_generations()
    updated = state.build_inputs(**arguments)
    assert torch.all(updated.selection_kv_block_status == -1)


def test_sequence_rollback_invalidates_selection():
    state, arguments = make_state_and_inputs()
    inputs = state.build_inputs(**arguments)
    inputs.selection_kv_block_status.fill_(42)
    arguments["kv_lengths"][0] -= 1
    updated = state.build_inputs(**arguments)
    assert torch.all(updated.selection_kv_block_status[0] == -1)
    assert torch.all(updated.selection_kv_block_status[1] == 42)


def test_pool_clear_and_reallocate_between_forwards_invalidates_selection():
    state, arguments = make_state_and_inputs()
    inputs = state.build_inputs(**arguments)
    inputs.selection_kv_block_status.fill_(42)
    state.pool.req_generation_epoch = 1
    state.refresh_generations()
    updated = state.build_inputs(**arguments)
    assert torch.all(updated.selection_kv_block_status == -1)


def test_builder_maps_cumulative_query_rows_and_masks_padding():
    state, arguments = make_state_and_inputs()
    arguments.update(
        q_nope=torch.zeros((4, 8, 512), dtype=torch.bfloat16),
        q_rope=torch.zeros((4, 8, 64), dtype=torch.bfloat16),
        topk_indices=torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32),
        query_lengths=torch.tensor([2, 3], dtype=torch.int32),
    )
    inputs = state.build_inputs(**arguments)
    cache = state.layer(0).cache
    assert cache.row_owner.tolist() == [4, 4, 5, -1]
    assert inputs.full_q_actual_seq.tolist() == [2, 3]
    assert inputs.selection_topk_indices[-1].tolist() == [[-1, -1]]


def test_graph_reservation_preserves_storage_for_smaller_batch():
    state, arguments = make_state_and_inputs()
    state.reserve(0, 8, 2, arguments["full_rope"], arguments["full_kv"])
    cache = state.layer(0).cache
    inputs = state.build_inputs(**arguments)
    assert inputs.selection_kv_cache.data_ptr() == cache.selection_kv_cache.data_ptr()
    assert inputs.selection_kv_block_status.is_contiguous()
    assert inputs.selection_kv_block_table.is_contiguous()
    assert (
        inputs.selection_kv_block_table.max().item()
        < inputs.selection_kv_cache.shape[0]
    )
    arguments["topk_indices"] = torch.zeros((2, 3), dtype=torch.int32)
    with pytest.raises(RuntimeError, match="after graph reservation"):
        state.build_inputs(**arguments)


def test_topk_geometry_change_does_not_reuse_stale_block_table():
    manager = FusedSparseAttentionCacheManager(torch.device("cpu"))
    full_rope = torch.empty((8, 64, 64), dtype=torch.bfloat16)
    full_kv = torch.empty((8, 64, 512), dtype=torch.bfloat16)
    old, _ = manager.ensure(
        torch.zeros((2, 129), dtype=torch.int32), full_rope, full_kv
    )
    cache, normalized = manager.ensure(
        torch.zeros((2, 2), dtype=torch.int32), full_rope, full_kv
    )
    assert cache is not old
    _, selection_kv, table, status, _ = manager.views(cache, normalized)
    assert table.max().item() < selection_kv.shape[0]
    assert status.is_contiguous()


def test_full_cache_noncontiguous_view_is_rejected_not_copied():
    combined = torch.empty((8, 64, 576), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous split"):
        _MODULE.paged_kv_view(combined[..., :512], 64, 512)
