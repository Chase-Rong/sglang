"""Decode capacity solver boundaries, independent of Ascend imports."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest


SOURCE = Path(__file__).parents[4] / "python/sglang/srt/hardware_backend/npu/decode_memory_budget.py"
SPEC = importlib.util.spec_from_file_location("decode_budget_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def budget(**overrides):
    values = dict(layers=4, planner_groups=1, latent_dim=512, rope_dim=64,
                  index_dim=128, element_bytes=2, topk=2048, resident_capacity=4096,
                  host_bytes=1 << 34)
    values.update(overrides)
    return MODULE.DecodeKvMemoryBudget(**values)


def test_mtp_sizes_query_rows_instead_of_request_count():
    ordinary = budget()
    mtp = budget(rows_per_request=4)
    assert mtp.row_capacity(2) == 8
    assert mtp.fixed_bytes(2, 64) == ordinary.fixed_bytes(8, 64)
    graph = budget(rows_per_request=4, graph_rows=8)
    assert graph.row_capacity(1) == 32
    assert graph.row_capacity(9) == 36


def test_target_and_draft_budget_includes_both_backings_and_selection():
    target = budget(layers=4, planner_groups=1)
    combined = budget(layers=5, planner_groups=2, rows_per_request=4)
    assert combined.host_bytes_per_token == target.host_bytes_per_token * 5 // 4
    assert combined.device_bytes_per_token == target.device_bytes_per_token * 5 // 4
    assert combined.fixed_bytes(2, 64)[1] > target.fixed_bytes(8, 64)[1]


def test_full_kv_scales_host_while_indexer_scales_npu():
    model = budget()
    before = model.required_bytes(8192, 4, 64)
    after = model.required_bytes(16384, 4, 64)
    assert after[0] - before[0] == 8192 * 4 * 576 * 2
    assert after[1] - before[1] == 8192 * 4 * 128 * 2


def test_integrated_slots_add_device_capacity_without_duplicating_host():
    ordinary = budget()
    integrated = budget(prefill_layer_slots=2)
    before = ordinary.required_bytes(8192, 1, 128)
    after = integrated.required_bytes(8192, 1, 128)
    assert after[0] == before[0]
    assert after[1] - before[1] == (8192 + 128) * 2 * 576 * 2
    assert integrated.max_tokens(after[1], 128, lambda tokens: 1) == 8192


def test_staging_reserves_one_fixed_window_and_copy_metadata():
    baseline = budget()
    staged = budget(staging_bytes=4096)
    for rows in (1, 8):
        before = baseline.required_bytes(8192, rows, 64)
        after = staged.required_bytes(8192, rows, 64)
        assert after[0] == before[0]
        assert after[1] - before[1] == 4096 + 9
    with pytest.raises(ValueError, match="staging"):
        budget(staging_bytes=33)


def test_solver_reserves_padding_page_and_fixed_selection_storage():
    model = budget()
    _, device_bytes = model.required_bytes(8192, 4, 64)
    assert model.max_tokens(device_bytes, 64, lambda tokens: 4) == 8192
    assert model.max_tokens(device_bytes - 1, 64, lambda tokens: 4) == 8128


def test_host_limit_includes_planner_and_padding():
    model = budget()
    host_required, device_required = model.required_bytes(8192, 4, 64)
    constrained = replace(model, host_bytes=host_required)
    assert constrained.max_tokens(device_required * 100, 64, lambda tokens: 4) == 8192
    with pytest.raises(MemoryError, match="Host"):
        constrained.validate(8256, 4, 64)


def test_graph_reservation_exceeds_real_request_count():
    model = budget(graph_rows=8)
    assert model.fixed_bytes(1, 64) == model.fixed_bytes(8, 64)
    assert model.fixed_bytes(9, 64)[1] > model.fixed_bytes(8, 64)[1]
    assert model.fixed_bytes(1, 64)[1] > budget().fixed_bytes(1, 64)[1]


def test_joint_solver_handles_request_count_threshold():
    model = budget()
    def row_count(tokens):
        return 2 if tokens <= 8192 else 4
    _, device_bytes = model.required_bytes(8192, 2, 64)
    result = model.max_tokens(device_bytes, 64, row_count)
    assert result == 8192
    model.validate(result, row_count(result), 64, device_bytes)
    with pytest.raises(MemoryError, match="Indexer"):
        model.validate(result + 64, row_count(result + 64), 64, device_bytes)


def test_shared_group_reduces_planner_storage_only():
    shared = budget()
    unshared = budget(planner_groups=4)
    shared_host, shared_device = shared.fixed_bytes(4, 64)
    unshared_host, unshared_device = unshared.fixed_bytes(4, 64)
    assert unshared_host > shared_host
    assert unshared_device - shared_device == 3 * 4 * (2048 * 8 + 9)


@pytest.mark.parametrize("page_size", [0, -1])
def test_invalid_page_size_rejected(page_size):
    with pytest.raises(ValueError):
        budget().required_bytes(8192, 4, page_size)


def test_alignment_cannot_overflow_native_slot_encoding():
    with pytest.raises(ValueError, match="aligned"):
        budget(resident_capacity=32767).fixed_bytes(4, 64)


def test_insufficient_npu_space_returns_no_capacity():
    _, fixed = budget().fixed_bytes(4, 64)
    assert budget().max_tokens(fixed, 64, lambda tokens: 4) == 0


def solve_request_priority(**overrides):
    values = dict(budget=budget(prefill_layer_slots=2), available_bytes=1 << 30,
                  token_limit=1048576, page_size=64, requests=8,
                  state_bytes_per_slot=1 << 20, request_bytes_per_row=32768)
    values.update(overrides)
    return MODULE.solve_joint_dsa_kda_tokens(**values)


def test_request_priority_shrinks_tokens_before_concurrency():
    result = solve_request_priority()
    assert result.requests == result.state_slots == 8
    assert 8 * 64 <= result.target_tokens < 1048576
    assert result.target_tokens % 64 == 0
    assert result.unused_bytes < 64 * budget(prefill_layer_slots=2).device_bytes_per_token
    assert result.dsa_bytes + result.kda_bytes + result.request_bytes + result.unused_bytes == 1 << 30


def test_request_priority_more_memory_increases_tokens_when_requests_fit():
    small = solve_request_priority(available_bytes=1 << 29)
    large = solve_request_priority()
    assert small.requests == large.requests == 8
    assert small.state_slots == large.state_slots == 8
    assert small.target_tokens < large.target_tokens


def test_request_priority_token_cap_is_shared_and_not_multiplied_or_divided():
    result = solve_request_priority(token_limit=8193)
    assert result.target_tokens == 8192
    assert result.requests == 8
    assert result.unused_bytes > 0


def test_request_priority_host_limit_also_reduces_tokens():
    model = budget(prefill_layer_slots=2)
    host_bytes = model.required_bytes(8192, 8, 64)[0]
    result = solve_request_priority(budget=replace(model, host_bytes=host_bytes))
    assert result.target_tokens == 8192
    assert result.requests == 8


@pytest.mark.parametrize("overrides", [dict(available_bytes=1),
                                      dict(token_limit=63),
                                      dict(budget=budget(host_bytes=1))])
def test_request_priority_fails_only_when_one_request_cannot_fit(overrides):
    with pytest.raises(MemoryError, match="cannot fit one request"):
        solve_request_priority(**overrides)


def test_request_priority_caps_requests_to_explicit_kda_capacity():
    result = solve_request_priority(explicit_state_slots=7)
    assert result.requests == result.state_slots == 7


def test_request_priority_charges_extra_kda_slots_and_speculative_storage():
    result = solve_request_priority(slots_per_request=3, explicit_state_slots=32,
                                  ring_bytes_per_slot=1024,
                                  intermediate_bytes_per_request=2048)
    assert result.requests == 8
    assert result.state_slots == 32
    assert result.kda_bytes == 33 * ((1 << 20) + 1024) + 9 * 2048


def test_request_priority_keeps_graph_row_floor():
    model = budget(prefill_layer_slots=2, graph_rows=16)
    result = solve_request_priority(budget=model)
    assert result.dsa_bytes == model.required_bytes(result.target_tokens, 16, 64)[1]


@pytest.mark.parametrize("capacity", [1, 3, 7])
def test_request_priority_finds_largest_concurrency_that_fits_hbm(capacity):
    model = budget(prefill_layer_slots=2)
    device = model.required_bytes(capacity * 64, capacity, 64)[1]
    available = device + (capacity + 1) * ((1 << 20) + 32768)
    result = solve_request_priority(available_bytes=available)
    assert result.requests == result.state_slots == capacity
    assert result.target_tokens == capacity * 64
    assert result.unused_bytes == 0


def test_request_priority_reduces_concurrency_for_host_budget():
    model = budget(prefill_layer_slots=2)
    host_bytes = model.required_bytes(3 * 64, 3, 64)[0]
    result = solve_request_priority(budget=replace(model, host_bytes=host_bytes))
    assert result.requests == 3
    assert result.target_tokens == 3 * 64


def test_request_priority_can_run_one_request_under_small_token_cap():
    result = solve_request_priority(token_limit=64)
    assert result.requests == 1
    assert result.target_tokens == 64


def test_request_priority_kda_ratio_limits_effective_requests():
    result = solve_request_priority(slots_per_request=3, explicit_state_slots=7)
    assert result.requests == 2
    assert result.state_slots == 7


def test_request_priority_does_not_exceed_native_query_row_limit():
    model = budget(rows_per_request=4)
    result = solve_request_priority(budget=model, available_bytes=1 << 42,
                                    requests=10000, token_limit=1 << 30)
    assert result.requests == 32767 // 4


def test_request_priority_page_one_retains_two_tokens_per_request():
    result = solve_request_priority(page_size=1, token_limit=3)
    assert result.requests == 1
    assert result.target_tokens == 3


def test_request_priority_cannot_remove_graph_reservation_to_fit_one_request():
    model = budget(graph_rows=16)
    available = model.required_bytes(64, 1, 64)[1] + 2 * ((1 << 20) + 32768) - 1
    with pytest.raises(MemoryError, match="cannot fit one request"):
        solve_request_priority(budget=model, available_bytes=available)
