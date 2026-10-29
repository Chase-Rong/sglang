"""Decode capacity solver boundaries, independent of Ascend imports."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import npu_module_name, npu_source_path, run_pytest

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

SOURCE = (
    Path(__file__).parents[4]
    / "python/sglang/srt/hardware_backend/npu/hisparse/decode_memory_budget.py"
)
SPEC = importlib.util.spec_from_file_location(
    npu_module_name("decode_budget_under_test"), npu_source_path(SOURCE)
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def budget(**overrides):
    values = dict(
        execution_abi="shared_v3",
        layers=4,
        planner_groups=1,
        latent_dim=512,
        rope_dim=0,
        index_dim=128,
        element_bytes=2,
        topk=4,
        resident_capacity=128,
        host_bytes=1 << 34,
    )
    values.update(overrides)
    return MODULE.DecodeKvMemoryBudget(**values)


def test_mtp_sizes_query_rows_instead_of_request_count():
    ordinary = budget()
    mtp = budget(rows_per_request=4)
    assert mtp.row_capacity(2) == 8
    mtp_host, mtp_device = mtp.fixed_bytes(2, 128)
    ordinary_host, ordinary_device = ordinary.fixed_bytes(8, 128)
    assert mtp_host < ordinary_host  # CPU generation snapshots are request-owned.
    # Row payload storage stays proportional to query rows, while device
    # planner resident-sort workspaces are allocated per active request.
    assert mtp_device < ordinary_device
    graph = budget(rows_per_request=4, graph_rows=8)
    assert graph.row_capacity(1) == 32
    assert graph.row_capacity(9) == 36


def test_target_and_draft_budget_includes_both_backings_and_selection():
    target = budget(layers=4, planner_groups=1)
    combined = budget(layers=5, planner_groups=2, rows_per_request=4)
    assert combined.host_bytes_per_token == target.host_bytes_per_token * 5 // 4
    assert combined.device_bytes_per_token == target.device_bytes_per_token * 5 // 4
    assert combined.fixed_bytes(2, 128)[1] > target.fixed_bytes(2, 128)[1]


def test_full_kv_scales_host_while_indexer_scales_npu():
    model = budget()
    before = model.required_bytes(8192, 4, 128)
    after = model.required_bytes(16384, 4, 128)
    assert after[0] - before[0] == 8192 * 4 * 512 * 2
    assert after[1] - before[1] == 8192 * 4 * 128 * 2


def test_independent_host_capacity_does_not_consume_device_kv_budget():
    ordinary = budget(prefill_layer_slots=2)
    expanded = budget(prefill_layer_slots=2, host_token_capacity=131072)
    before = ordinary.required_bytes(8192, 4, 128)
    after = expanded.required_bytes(8192, 4, 128)
    assert after[0] - before[0] == 122880 * 4 * 512 * 2
    assert after[1] == before[1]
    assert expanded.max_tokens(before[1], 128, lambda _: 4) == 8192
    # This is a minimum Host capacity, not a cap on the device solver.
    assert expanded.required_bytes(262144, 4, 128) == ordinary.required_bytes(
        262144, 4, 128
    )


def test_independent_host_budget_includes_target_draft_and_reserved_page():
    model = budget(layers=5, host_token_capacity=131072)
    host_fixed, _ = model.fixed_bytes(4, 128)
    assert model.required_bytes(8192, 4, 128)[0] == host_fixed + 131200 * 5 * 512 * 2
    required, device_bytes = model.required_bytes(8192, 4, 128)
    too_small = replace(model, host_bytes=required - 1)
    with pytest.raises(MemoryError, match="Host"):
        too_small.validate(8192, 4, 128, device_bytes)
    assert too_small.max_tokens(device_bytes, 128, lambda _: 4) == 0


def test_independent_host_capacity_requires_nonnegative_aligned_tokens():
    with pytest.raises(ValueError, match="Host"):
        budget(host_token_capacity=-1)
    with pytest.raises(ValueError, match="Host"):
        budget(host_token_capacity=129).required_bytes(8192, 4, 128)


def test_integrated_slots_add_device_capacity_without_duplicating_host():
    ordinary = budget()
    integrated = budget(prefill_layer_slots=2)
    before = ordinary.required_bytes(8192, 1, 128)
    after = integrated.required_bytes(8192, 1, 128)
    assert after[0] == before[0]
    assert after[1] - before[1] == (8192 + 128) * 2 * 512 * 2
    assert integrated.max_tokens(after[1], 128, lambda tokens: 1) == 8192


def test_staging_reserves_one_fixed_window_and_copy_metadata():
    baseline = budget()
    staged = budget(staging_bytes=4096)
    for rows in (1, 8):
        before = baseline.required_bytes(8192, rows, 128)
        after = staged.required_bytes(8192, rows, 128)
        assert after[0] == before[0]
        assert after[1] - before[1] == 4096
    with pytest.raises(ValueError, match="staging"):
        budget(staging_bytes=33)


def test_solver_reserves_padding_page_and_fixed_selection_storage():
    model = budget()
    _, device_bytes = model.required_bytes(8192, 4, 128)
    assert model.max_tokens(device_bytes, 128, lambda tokens: 4) == 8192
    assert model.max_tokens(device_bytes - 1, 128, lambda tokens: 4) == 8064


def test_host_limit_includes_planner_and_padding():
    model = budget()
    host_required, device_required = model.required_bytes(8192, 4, 128)
    constrained = replace(model, host_bytes=host_required)
    assert constrained.max_tokens(device_required * 100, 128, lambda tokens: 4) == 8192
    with pytest.raises(MemoryError, match="Host"):
        constrained.validate(8320, 4, 128)


def test_graph_reservation_exceeds_real_request_count():
    model = budget(graph_rows=8)
    assert model.fixed_bytes(1, 128)[0] < model.fixed_bytes(8, 128)[0]
    assert model.row_capacity(1) == model.row_capacity(8)
    assert model.fixed_bytes(1, 128)[1] < model.fixed_bytes(8, 128)[1]
    assert model.fixed_bytes(9, 128)[1] > model.fixed_bytes(8, 128)[1]
    assert model.fixed_bytes(1, 128)[1] > budget().fixed_bytes(1, 128)[1]


def test_joint_solver_handles_request_count_threshold():
    model = budget()

    def row_count(tokens):
        return 2 if tokens <= 8192 else 4

    _, device_bytes = model.required_bytes(8192, 2, 128)
    result = model.max_tokens(device_bytes, 128, row_count)
    assert result == 8192
    model.validate(result, row_count(result), 128, device_bytes)
    with pytest.raises(MemoryError, match="Indexer"):
        model.validate(result + 128, row_count(result + 128), 128, device_bytes)


def test_shared_group_reduces_planner_storage_only():
    shared = budget()
    unshared = budget(planner_groups=4)
    shared_host, shared_device = shared.fixed_bytes(4, 128)
    unshared_host, unshared_device = unshared.fixed_bytes(4, 128)
    assert unshared_host == shared_host
    assert unshared_device == shared_device  # No CPU planner group allocations remain.


@pytest.mark.parametrize("page_size", [0, -1])
def test_invalid_page_size_rejected(page_size):
    with pytest.raises(ValueError):
        budget().required_bytes(8192, 4, page_size)


def test_alignment_cannot_overflow_native_slot_encoding():
    with pytest.raises(ValueError, match="C<=16384"):
        budget(resident_capacity=32767).fixed_bytes(4, 128)


def test_insufficient_npu_space_returns_no_capacity():
    _, fixed = budget().fixed_bytes(4, 128)
    assert budget().max_tokens(fixed, 128, lambda tokens: 4) == 0


def solve_request_priority(**overrides):
    values = dict(
        budget=budget(prefill_layer_slots=2),
        available_bytes=1 << 30,
        token_limit=1048576,
        page_size=128,
        requests=8,
        state_bytes_per_slot=1 << 20,
        request_bytes_per_row=32768,
    )
    values.update(overrides)
    return MODULE.solve_joint_dsa_kda_tokens(**values)


def test_request_priority_shrinks_tokens_before_concurrency():
    result = solve_request_priority()
    assert result.requests == result.state_slots == 8
    assert 8 * 128 <= result.target_tokens < 1048576
    assert result.target_tokens % 128 == 0
    assert (
        result.unused_bytes < 128 * budget(prefill_layer_slots=2).device_bytes_per_token
    )
    assert (
        result.dsa_bytes + result.kda_bytes + result.request_bytes + result.unused_bytes
        == 1 << 30
    )


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
    host_bytes = model.required_bytes(8192, 8, 128)[0]
    result = solve_request_priority(budget=replace(model, host_bytes=host_bytes))
    assert result.target_tokens == 8192
    assert result.requests == 8


@pytest.mark.parametrize(
    "overrides",
    [dict(available_bytes=1), dict(token_limit=63), dict(budget=budget(host_bytes=1))],
)
def test_request_priority_fails_only_when_one_request_cannot_fit(overrides):
    with pytest.raises(MemoryError, match="cannot fit one request"):
        solve_request_priority(**overrides)


def test_request_priority_caps_requests_to_explicit_kda_capacity():
    result = solve_request_priority(explicit_state_slots=7)
    assert result.requests == result.state_slots == 7


def test_request_priority_charges_extra_kda_slots_and_speculative_storage():
    result = solve_request_priority(
        slots_per_request=3,
        explicit_state_slots=32,
        ring_bytes_per_slot=1024,
        intermediate_bytes_per_request=2048,
    )
    assert result.requests == 8
    assert result.state_slots == 32
    assert result.kda_bytes == 33 * ((1 << 20) + 1024) + 9 * 2048


def test_request_priority_keeps_graph_row_floor():
    model = budget(prefill_layer_slots=2, graph_rows=16)
    result = solve_request_priority(budget=model)
    assert model.row_capacity(result.requests) == 16
    assert (
        result.dsa_bytes
        == model.required_bytes(result.target_tokens, result.requests, 128)[1]
    )


@pytest.mark.parametrize("capacity", [1, 3, 7])
def test_request_priority_finds_largest_concurrency_that_fits_hbm(capacity):
    model = budget(prefill_layer_slots=2)
    device = model.required_bytes(capacity * 128, capacity, 128)[1]
    available = device + (capacity + 1) * ((1 << 20) + 32768)
    result = solve_request_priority(available_bytes=available)
    assert result.requests == result.state_slots == capacity
    assert result.target_tokens == capacity * 128
    assert result.unused_bytes == 0


def test_request_priority_reduces_concurrency_for_host_budget():
    model = budget(prefill_layer_slots=2)
    host_bytes = model.required_bytes(3 * 128, 3, 128)[0]
    result = solve_request_priority(budget=replace(model, host_bytes=host_bytes))
    assert result.requests == 3
    assert result.target_tokens == 3 * 128


def test_request_priority_can_run_one_request_under_small_token_cap():
    result = solve_request_priority(token_limit=128)
    assert result.requests == 1
    assert result.target_tokens == 128


def test_request_priority_kda_ratio_limits_effective_requests():
    result = solve_request_priority(slots_per_request=3, explicit_state_slots=7)
    assert result.requests == 2
    assert result.state_slots == 7


def test_request_priority_does_not_exceed_native_query_row_limit():
    model = budget(rows_per_request=4)
    result = solve_request_priority(
        budget=model, available_bytes=1 << 42, requests=10000, token_limit=1 << 30
    )
    assert result.requests == 1024


def test_external_plan_rejects_non_native_page_size():
    with pytest.raises(ValueError, match="geometry"):
        solve_request_priority(page_size=1, token_limit=3)


def test_request_priority_cannot_remove_graph_reservation_to_fit_one_request():
    model = budget(graph_rows=16)
    available = model.required_bytes(128, 1, 128)[1] + 2 * ((1 << 20) + 32768) - 1
    with pytest.raises(MemoryError, match="cannot fit one request"):
        solve_request_priority(budget=model, available_bytes=available)


def test_compact_indexer_and_scales_charge_physical_layers_only():
    compact = budget(index_layers=2, index_scale_bytes=4, prefill_layer_slots=1)
    full = budget(index_layers=4, index_scale_bytes=4, prefill_layer_slots=1)
    before = compact.required_bytes(8192, 3, 128)
    after = full.required_bytes(8192, 3, 128)
    assert before[0] == after[0]
    assert after[1] - before[1] == (8192 + 128) * 2 * (128 * 2 + 4)
    compact.validate(8192, 3, 128, before[1])
    assert compact.max_tokens(before[1] - 1, 128, lambda _: 3) == 8064


def test_tail_slots_follow_requests_while_selection_follows_graph_routes():
    model = budget(graph_rows=8, rows_per_request=6, index_tail_bytes_per_request=4096)
    metadata_growth = model.external_metadata_bytes(
        2, 128
    ) - model.external_metadata_bytes(1, 128)
    assert model.fixed_bytes(2, 128)[1] - model.fixed_bytes(1, 128)[1] == (
        4096 + model.layers * (model.resident_capacity * (512 * 2 + 11) + 12)
        + 34 * model.external_resident_instances
        + 34 * model.layers * model.callback_instances
    )
    assert model.row_capacity(2) == 48
    one_route = replace(model, rows_per_request=1)
    assert model.fixed_bytes(1, 128)[1] > one_route.fixed_bytes(1, 128)[1]


def test_kda_extra_per_slot_and_fixed_replay_storage_count_once():
    result = solve_request_priority(
        slots_per_request=3,
        ring_bytes_per_slot=4096,
        intermediate_bytes_per_request=2048,
        fixed_state_bytes=65536,
    )
    assert result.kda_bytes == 25 * ((1 << 20) + 4096) + 9 * 2048 + 65536
    assert (
        result.dsa_bytes + result.kda_bytes + result.request_bytes + result.unused_bytes
        == 1 << 30
    )


def test_prefill_physical_slots_do_not_reduce_logical_host_payload():
    model = MODULE.PrefillKvMemoryBudget(
        host_bytes_per_token=10 * 512 * 2,
        device_bytes_per_token=2 * 512 * 2 + 7 * 128 * 2,
        host_bytes=(8192 + 128) * 10 * 512 * 2,
        device_fixed_bytes=4096,
    )
    device_bytes = (8192 + 128) * model.device_bytes_per_token + 4096
    assert model.max_tokens(device_bytes, 128) == 8192
    assert model.max_tokens(device_bytes - 1, 128) == 8064
    with pytest.raises(MemoryError):
        model.validate_host_capacity(8320, 128)


def test_dense_decode_can_use_joint_request_priority_without_kda_allocation():
    result = solve_request_priority(state_bytes_per_slot=0)
    assert result.kda_bytes == 0
    assert result.requests == 8


def _startup_budget_config(monkeypatch, *, backend="lim_copy_sfa", extra_slots=0):
    import sglang.srt.hardware_backend.npu.fsa_native as native
    monkeypatch.setattr(native, "shared_v3_workspace_bytes", lambda: 4096)
    monkeypatch.setenv("SGLANG_NPU_FSA_EXECUTION_ABI", "shared_v3")

    def install(name, **values):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**values))

    values = dict(
        SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=True,
        SGLANG_NPU_PREFILL_LAYER_SLOTS=2,
        SGLANG_NPU_FSA_RESIDENT_CAPACITY=4096,
        SGLANG_NPU_DECODE_HOST_KV_BYTES=1 << 34,
        SGLANG_NPU_SHARED_HOST_KV=True,
        SGLANG_NPU_LAYER_STAGING_BYTES=0,
        SGLANG_NPU_HICACHE_HOST_TOKENS=0,
        SGLANG_NPU_DSA_OFFLOAD_BACKEND=backend,
    )
    install(
        "sglang.srt.environ",
        envs=SimpleNamespace(
            **{
                name: SimpleNamespace(get=lambda value=value: value)
                for name, value in values.items()
            }
        ),
    )
    install(
        "sglang.srt.configs.model_config",
        is_deepseek_dsa=lambda _: True,
        dsa_layer_skips_topk=lambda config, layer: layer in (2, 6),
        get_dsa_index_kpool_compress=lambda _: True,
    )
    install(
        "sglang.srt.model_executor.pool_configurator",
        _get_dsa_cache_layer_ids=lambda kvc, layers: [0, 2, 4, 6],
    )
    install(
        "sglang.srt.mem_cache.kv_cache_configurator",
        _should_elide_dsa_index_k=lambda **values: False,
    )
    install(
        "sglang.srt.hardware_backend.npu.hisparse.speculative_sparse",
        sparse_mtp_draft_layers=lambda _: 1,
        sparse_mtp_rows=lambda *values: 6,
    )
    install(
        "sglang.srt.hardware_backend.npu.hisparse.offload_config",
        layer_staging_bytes=lambda *args, **values: 0,
    )
    graph = SimpleNamespace(backend="npu", bs=[1, 2, 8], max_bs=8)
    install(
        "sglang.srt.runtime_context",
        get_disagg=lambda: SimpleNamespace(
            disaggregation_mode="null",
            enable_pd_role_switch=extra_slots > 0,
            disaggregation_decode_extra_slots=extra_slots,
        ),
        get_exec=lambda: SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=graph))
        ),
        get_spec=lambda: SimpleNamespace(speculative_num_steps=5),
        max_speculative_num_draft_tokens=lambda: 6,
    )
    model = SimpleNamespace(
        hf_config=object(),
        hf_text_config=SimpleNamespace(index_topk=2048, index_kpool=4),
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_head_dim=128,
        context_len=8192,
    )
    import torch

    kvc = SimpleNamespace(
        device="npu",
        use_mla_backend=True,
        model_config=model,
        kv_cache_dtype=torch.bfloat16,
        mambaish_config=object(),
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
    )
    return kvc


def test_budget_builder_prices_allocated_index_layers_separately_from_planner_groups(
    monkeypatch,
):
    result = MODULE.budget_from_config(_startup_budget_config(monkeypatch), 4)
    assert result.layers == result.index_layers == 5
    assert result.planner_groups == 3
    assert result.prefill_layer_slots == 3
    assert result.graph_rows == 8
    assert result.rows_per_request == 6
    assert result.index_tail_bytes_per_request == 3 * 10 * 128 * 4
    assert result.device_bytes_per_token == (5 * 128 + 3 * 512) * 2
    assert result.fixed_bytes(2, 128)[1] > result.fixed_bytes(1, 128)[1]


def _allocated_external_metadata_bytes(result, request_capacity):
    """Measure real target + draft metadata tensors, without NPU allocation."""
    from sglang.srt.hardware_backend.npu.hisparse.external_sparse_planner import (
        ExternalPlanResidentState,
    )
    import torch

    total = 0
    for layer_count in (result.layers - 1, 1):
        resident = ExternalPlanResidentState(
            "cpu",
            request_capacity,
            result.resident_capacity,
            max_routes=result.rows_per_request,
            expanded_topk=result.topk,
            page_size=128,
        )
        for layer_id in range(layer_count):
            resident.ensure_layer(layer_id)
        tensors = [v for v in vars(resident).values() if isinstance(v, torch.Tensor)]
        tensors += [v for layer in resident.layers.values() for v in layer.values()]
        total += sum(t.numel() * t.element_size() for t in tensors)
    return (total + 7) // 8 * 8


@pytest.mark.parametrize("extra_slots", [0, 3])
@pytest.mark.parametrize("requests", [1, 2, 7])
def test_startup_solver_prices_candidate_resident_before_request_pool_exists(
    monkeypatch,
    requests,
    extra_slots,
):
    kvc = _startup_budget_config(
        monkeypatch,
        backend="external_plan",
        extra_slots=extra_slots,
    )
    assert not hasattr(kvc, "req_to_token_pool")
    result = MODULE.budget_from_config(kvc, 4)
    # The graph floor fixes row payload/planner geometry at eight requests.
    # Below it, resident storage must still grow with the real request pool.
    before = result.fixed_bytes(requests, 128)[1]
    after = result.fixed_bytes(requests + 1, 128)[1]
    qcap = requests + extra_slots + 1
    metadata_growth = _allocated_external_metadata_bytes(
        result, qcap + 1
    ) - _allocated_external_metadata_bytes(result, qcap)
    # Below the graph floor, only request-owned payload/metadata and per-
    # invocation lease snapshots grow; planner Bcap/Tcap stay fixed.
    payload_growth = result.layers * result.resident_capacity * 512 * 2
    epoch_generation_growth = 8 * result.external_resident_instances
    lease_growth = 34 * result.layers * len(result.shared_graph_executions)
    assert after - before == (
        metadata_growth + payload_growth + epoch_generation_growth
        + lease_growth + result.index_tail_bytes_per_request
    )


@pytest.mark.parametrize("requests", [1, 3, 8])
def test_startup_metadata_matches_actual_allocation_and_reserved_slot(
    monkeypatch,
    requests,
):
    result = MODULE.budget_from_config(
        _startup_budget_config(monkeypatch, backend="external_plan"),
        4,
    )
    measured = _allocated_external_metadata_bytes(result, requests + 1)
    assert result.external_metadata_bytes(requests, 128) == measured
    graph_larger = replace(result, graph_rows=32)
    assert graph_larger.external_metadata_bytes(requests, 128) == measured


def test_existing_request_pool_capacity_is_charged_when_solver_shrinks(monkeypatch):
    import torch

    kvc = _startup_budget_config(monkeypatch, backend="external_plan")
    kvc.req_to_token_pool = SimpleNamespace(req_generation=torch.zeros(12))
    result = MODULE.budget_from_config(kvc, 4)
    assert result.external_metadata_bytes(2, 128) == (
        _allocated_external_metadata_bytes(result, 12)
    )


def test_joint_solver_reduces_requests_for_resident_metadata_under_graph_floor():
    model = budget(layers=1, planner_groups=1, graph_rows=4, native_workspace_bytes=0)
    available = model.required_bytes(3 * 128, 3, 128)[1]
    assert model.required_bytes(4 * 128, 4, 128)[1] > available
    result = MODULE.solve_joint_dsa_kda_tokens(
        budget=model, available_bytes=available, token_limit=1024, page_size=128,
        requests=4, state_bytes_per_slot=0)
    assert result.requests == 3
    assert result.target_tokens == 3 * 128
    assert result.dsa_bytes == available
    assert result.unused_bytes == 0


def test_shared_domain_rounds_once_and_private_sidecars_are_never_shared():
    plan = MODULE.HostMemoryBudgetPlan(
        (1 << 30) + 1, 4 << 30, shared_ranks=4, planner_bytes_per_rank=4096
    )
    assert plan.aligned_reserve_bytes == 2 << 30
    assert plan.physical_charge_bytes(is_writer=True) == 2 << 30
    assert plan.physical_charge_bytes(is_writer=False) == 0
    assert plan.required_scope_bytes(8192) == (2 << 30) + 4 * (4096 + 8192)
    assert plan.sidecar_budget_bytes == (1 << 29) - 4096
    plan.validate(plan.sidecar_budget_bytes)
    with pytest.raises(MemoryError, match="cgroup/OS"):
        plan.validate(plan.sidecar_budget_bytes + 1)


def test_rounding_cannot_hide_a_domain_that_exceeds_scope():
    plan = MODULE.HostMemoryBudgetPlan((1 << 30) + 1, (2 << 30) - 1, shared_ranks=8)
    with pytest.raises(MemoryError, match="aligned MemFabric"):
        plan.validate()


def test_host_sidecar_pressure_reduces_concurrency_before_hbm_runs_out():
    base = budget(
        host_bytes=1 << 30, host_index_sidecar=True, host_index_kpool=4, index_layers=3
    )
    requests = 3
    tokens = requests * 128
    planner, _ = base.fixed_bytes(requests, 128)
    checkpoints = (requests * 2 + 1) * (1 << 20)
    private = planner + base.host_indexer_bytes(tokens, 128) + checkpoints
    plan = MODULE.HostMemoryBudgetPlan(1 << 30, (1 << 30) + 4 * private, shared_ranks=4)
    model = replace(base, host_memory_plan=plan)
    result = solve_request_priority(
        budget=model,
        available_bytes=1 << 40,
        host_state_bytes_per_slot=1 << 20,
        host_to_device_ratio=2,
    )
    assert result.requests == requests
    assert result.target_tokens == tokens
    assert model.host_scope_fits(tokens, requests, 128, checkpoints)
    assert not model.host_scope_fits(tokens + 128, requests, 128, checkpoints)


def test_private_indexer_scales_and_independent_host_capacity_count_per_rank():
    model = budget(
        host_index_sidecar=True,
        host_index_kpool=4,
        index_layers=2,
        index_scale_bytes=4,
        host_token_capacity=131072,
    )
    expected = (131072 // 128 + 1) * 2 * ((128 // 4 * 128 * 2) + (128 // 4 * 4))
    assert model.host_indexer_bytes(8192, 128) == expected
    assert model.host_indexer_bytes(16384, 128) == expected


def test_planner_is_private_host_memory_outside_the_memfabric_payload_budget():
    model = budget()
    plan = MODULE.HostMemoryBudgetPlan(model.host_bytes, 1 << 40, shared_ranks=4)
    scoped = replace(model, host_memory_plan=plan)
    planner, _ = model.fixed_bytes(3, 128)
    assert (
        model.required_bytes(8192, 3, 128)[0] - scoped.required_bytes(8192, 3, 128)[0]
        == planner
    )
    frozen = replace(
        scoped, host_memory_plan=replace(plan, planner_bytes_per_rank=planner)
    )
    assert frozen.host_scope_fits(8192, 3, 128)
    assert (
        frozen.host_memory_plan.required_scope_bytes()
        == plan.aligned_reserve_bytes + 4 * planner
    )


if __name__ == "__main__":
    run_pytest(__file__)
