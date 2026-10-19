"""CPU semantic gates for the alternative A5 pooled offload operators."""

from npu_test_paths import npu_module_name, npu_source_path
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(npu_module_name(name), npu_source_path(ROOT / path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = load_module(
    "lim_contract_under_test",
    "python/sglang/srt/hardware_backend/npu/hisparse/lim_copy_sfa_contract.py",
)
REFERENCE = load_module(
    "lim_reference_under_test", "test/manual/npu/lim_copy_sfa_reference.py"
)


@pytest.mark.parametrize("explicit_planner", [False, True])
def test_legacy_comparator_initializes_selected_planner(monkeypatch, explicit_planner):
    calls = []

    def make_planner(spec, request_capacity, batch_capacity, device):
        calls.append((spec, request_capacity, batch_capacity, device))
        return SimpleNamespace(rows=batch_capacity * spec.max_query_routes)

    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_device_plan"),
        SimpleNamespace(DeviceUnionPlanner=make_planner),
    )
    spec = CONTRACT.LimCopySfaSpec(8192, 1, 2176)
    kwargs = {"planner_type": make_planner} if explicit_planner else {}
    comparator = REFERENCE.LegacyFsaComparator(
        spec, 1, 2, "cpu", torch.bfloat16, **kwargs
    )
    assert calls == [(spec, 1, 2, "cpu")]
    assert comparator.table.shape == (2, 17)
    assert comparator.kv.shape == (17, 128, spec.latent_dim)


@pytest.mark.parametrize("routes,capacity", [(1, 2176), (6, 12416)])
def test_capacity_covers_expanded_union(routes, capacity):
    spec = CONTRACT.LimCopySfaSpec(1 << 20, routes, capacity)
    assert spec.pooled_topk == 512
    assert spec.token_topk == 2051
    memory = spec.storage_bytes(10, 1)
    assert memory["mapping"] == 40 << 20
    assert memory["payload"] == 10 * capacity * 1024
    assert spec.storage_bytes(10, 2)["mapping"] == memory["mapping"] * 2


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_query_routes=7),
        dict(max_query_routes=0),
        dict(resident_tokens=4096),
        dict(resident_tokens=33024),
        dict(resident_tokens=12306),
        dict(pool_size=1),
        dict(index_topk=512),
        dict(rope_dim=64),
        dict(page_size=64),
        dict(max_source_tokens=(1 << 21) + 1),
    ],
)
def test_unsupported_contracts_fail_explicitly(kwargs):
    options = dict(max_source_tokens=1 << 20, max_query_routes=6, resident_tokens=12416)
    options.update(kwargs)
    with pytest.raises(ValueError):
        CONTRACT.LimCopySfaSpec(**options)


def test_graph_outputs_are_budgeted_separately_from_request_maps():
    spec = CONTRACT.LimCopySfaSpec(1 << 18, 6, 12416)
    ordinary = spec.storage_bytes(10, 1)
    padded = spec.storage_bytes(10, 1, graph_rows=96)
    assert ordinary["mapping"] == padded["mapping"] == 10 << 20
    assert padded["outputs"] > ordinary["outputs"]
    assert padded["payload"] == ordinary["payload"]


@pytest.mark.parametrize("value", ["", "auto", "host", "typo"])
def test_unknown_backend_never_silently_falls_back(value):
    with pytest.raises(ValueError):
        CONTRACT.resolve_dsa_offload_backend(value)


def test_known_backend_names():
    for name in ("external_plan", "lim_copy_sfa"):
        assert CONTRACT.resolve_dsa_offload_backend(name) == name


@pytest.mark.parametrize("length", [0, 1, 3, 4, 7, 8, 9, 11, 12, 15])
def test_pool_selection_preserves_exact_causal_tail(length):
    query = torch.ones(1, 2, 1)
    keys = torch.arange(4, dtype=torch.float32)[:, None]
    weights = torch.ones(1, 2)
    ids, scores = REFERENCE.pooled_selection(query, weights, keys, [length], topk=8)
    assert ids.shape == (1, 11)
    selected = ids[ids >= 0].tolist()
    if length <= 8:
        assert selected == list(range(length))
    else:
        pools = length // 4
        expected = set(range((pools - 2) * 4, pools * 4)) | set(
            range(pools * 4, length)
        )
        assert set(selected) == expected
    assert all(source < length for source in selected)
    assert torch.isneginf(scores[0, length // 4 :]).all()


def test_verify_queries_keep_distinct_pool_prefixes():
    query = torch.ones(6, 2, 1)
    keys = torch.arange(5, dtype=torch.float32)[:, None]
    ids, _ = REFERENCE.pooled_selection(
        query, torch.ones(6, 2), keys, [10, 11, 12, 13, 14, 15], topk=8
    )
    assert 8 not in ids[0, :8].tolist()
    assert 8 in ids[2, :8].tolist()
    assert ids[0, 8:].tolist() == [8, 9, -1]
    assert ids[2, 8:].tolist() == [-1, -1, -1]


def test_score_eviction_protects_union_and_prefers_low_score():
    state = REFERENCE.ReferenceResident(4, 64)
    table = torch.tensor([1])
    first = state.plan(torch.tensor([[0, 4, 8, 12]]), table, [64], 1, mutable_tail=0)
    state.commit(first)
    scores = torch.tensor([[50.0, 10.0, 1.0, 20.0, 15.0]])
    second = state.plan(
        torch.tensor([[0, 16], [4, 16]]),
        table,
        [64, 64],
        1,
        mutable_tail=0,
        scores=scores.expand(2, -1),
    )
    assert set(state.sources.tolist()) == {0, 4, 12, 16}
    assert second.copy_sources.tolist() == [144]
    assert second.slots[0, 1] == second.slots[1, 1]
    state.commit(second)


def test_reshuffle_all_hit_and_duplicate_attention_weight():
    state = REFERENCE.ReferenceResident(8, 64)
    table = torch.tensor([1])
    host = torch.randn(256, 4, generator=torch.Generator().manual_seed(7))
    cache = torch.zeros(8, 4)
    query = torch.ones(1, 2, 4)
    ids = torch.tensor([[0, 4, 4, 8]])
    first = state.plan(ids, table, [64], 1, mutable_tail=0)
    result = REFERENCE.copy_attention(query, host, cache, first, 0.5)
    keys = host[128 + ids[0]]
    expected = torch.softmax(query[0] @ keys.T * 0.5, dim=-1) @ keys
    torch.testing.assert_close(result[0], expected)
    assert first.copy_sources.numel() == 3
    state.commit(first)
    second = state.plan(ids.flip(1), table, [64], 1, mutable_tail=0)
    assert not second.misses.any()
    assert second.copy_sources.numel() == 0
    state.commit(second)


@pytest.mark.parametrize("change", ["generation", "rollback", "remap"])
def test_ownership_and_physical_mapping_changes_force_copy(change):
    state = REFERENCE.ReferenceResident(4, 64)
    ids = torch.tensor([[1, 2]])
    first = state.plan(ids, torch.tensor([1]), [64], 1, mutable_tail=0)
    state.commit(first)
    second = state.plan(
        ids,
        torch.tensor([2 if change == "remap" else 1]),
        [32 if change == "rollback" else 64],
        2 if change == "generation" else 1,
        mutable_tail=0,
    )
    assert second.misses.all()
    state.commit(second)


def test_mutable_tail_is_refreshed_after_commit():
    state = REFERENCE.ReferenceResident(4, 64)
    ids = torch.tensor([[1, 62]])
    for _ in range(2):
        plan = state.plan(ids, torch.tensor([1]), [64], 1, mutable_tail=12)
        assert plan.misses[0, 1]
        state.commit(plan)
    assert not plan.misses[0, 0]


def test_padding_never_changes_persistent_owner_state():
    state = REFERENCE.ReferenceResident(4, 64)
    first = state.plan(
        torch.tensor([[1, 2]]), torch.tensor([1]), [64], 1, mutable_tail=0
    )
    state.commit(first)
    sources, physical = state.sources.clone(), state.physical.clone()
    padding = state.plan(
        torch.tensor([[12, 13]]), torch.tensor([0]), [0], 0, active=False
    )
    assert not padding.misses.any()
    state.commit(padding)
    assert state.generation == 1 and state.sequence_length == 64
    torch.testing.assert_close(state.sources, sources)
    torch.testing.assert_close(state.physical, physical)


def test_plan_must_complete_before_next_plan_and_abort_invalidates():
    state = REFERENCE.ReferenceResident(4, 64)
    args = torch.tensor([[1, 2]]), torch.tensor([1]), [64], 1
    first = state.plan(*args, mutable_tail=0)
    with pytest.raises(RuntimeError, match="complete"):
        state.plan(*args, mutable_tail=0)
    state.abort()
    with pytest.raises(RuntimeError, match="foreign"):
        state.commit(first)
    fresh = state.plan(*args, mutable_tail=0)
    assert fresh.misses.all()


def test_union_capacity_is_not_silently_truncated():
    state = REFERENCE.ReferenceResident(3, 64)
    with pytest.raises(ValueError, match="union"):
        state.plan(torch.tensor([[1, 2], [3, 4]]), torch.tensor([1]), [64, 64], 1)


def test_source_ids_above_17_bits_are_preserved():
    state = REFERENCE.ReferenceResident(4, 1 << 21)
    ids = torch.tensor([[(1 << 20) + 1, (1 << 20) + 2]])
    table = torch.arange(1 << 14)
    plan = state.plan(ids, table, [(1 << 20) + 64], 1, mutable_tail=0)
    torch.testing.assert_close(plan.physical_ids, ids.long())


def test_independent_layers_do_not_share_payload_validity():
    first_layer = REFERENCE.ReferenceResident(4, 64)
    second_layer = REFERENCE.ReferenceResident(4, 64)
    args = torch.tensor([[1, 2]]), torch.tensor([1]), [64], 1
    first = first_layer.plan(*args, mutable_tail=0)
    first_layer.commit(first)
    second = second_layer.plan(*args, mutable_tail=0)
    assert second.misses.all()
