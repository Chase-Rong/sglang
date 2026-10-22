from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytest import main as run_pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NPU = Path(__file__).resolve().parents[4] / "python/sglang/srt/hardware_backend/npu"
PREFIX = "sglang.srt.hardware_backend.npu.hisparse."


def load(monkeypatch, name):
    spec = importlib.util.spec_from_file_location(
        PREFIX + name, NPU / "hisparse" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    return load(monkeypatch, "fused_sparse_attention"), load(
        monkeypatch, "external_sparse_planner"
    )


@pytest.mark.parametrize("requests,routes", [(1, 1), (2, 1), (1, 6), (2, 6)])
@pytest.mark.parametrize("hit_percent", [0, 80, 99, 100])
def test_2051_selection_reorders_mask_and_copies_only_misses(
    modules, monkeypatch, requests, routes, hit_percent
):
    _, planner = modules
    rows, width, capacity = requests * routes, 2051, 2176
    applied = torch.full((rows, capacity + 1), -1, dtype=torch.int64)
    cache = SimpleNamespace(
        selection_kv_cache=torch.empty(rows * capacity // 128, 128, 1),
        max_tokens=rows,
        applied_keys=applied[:, :-1],
        applied_keys_with_padding=applied,
    )
    membership = torch.full((rows, 16400), -7, dtype=torch.int16)
    keys = torch.arange(width).repeat(rows, 1) + 128
    slots = torch.arange(width).repeat(rows, 1)
    hits = width * hit_percent // 100
    applied[:, :hits] = keys[:, :hits]
    order = torch.arange(width).roll(37).flip(0)
    keys, slots = keys[:, order], slots[:, order]
    plan = planner.materialize_layer_plan(cache, keys, slots, membership)
    assert plan.miss_mask.sum().item() == rows * (width - hits)
    encoded = membership[:, 16384 - width : 16384]
    assert torch.equal(encoded.abs().long(), slots + 1)
    assert torch.equal(encoded < 0, plan.miss_mask)
    assert membership[0, 16384:16392].tolist() == [
        0x5A4D,
        0x5A45,
        2051,
        14333,
        0,
        0x5A44,
        capacity,
        0,
    ]
    assert (membership[:, 14332] == -7).all()
    source = torch.arange(4096, dtype=torch.float32).reshape(-1, 1)
    destination = torch.full((rows * capacity, 1), -999.0)
    destination.view(rows, capacity, 1)[:, :hits] = source[torch.arange(hits) + 128]
    copied = []

    def copy(source, target, source_ids, destination_ids, valid, **kwargs):
        copied.append(valid.sum().item())
        target[destination_ids[valid]] = source[source_ids[valid]]

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.hardware_backend.npu.fsa_native",
        SimpleNamespace(unidex_copy_inplace=copy),
    )
    import sglang.srt.hardware_backend.npu as npu

    monkeypatch.setattr(
        npu,
        "fsa_native",
        sys.modules["sglang.srt.hardware_backend.npu.fsa_native"],
        raising=False,
    )
    inputs = SimpleNamespace(
        selection_plan=plan,
        selection_kv_cache=destination.reshape(-1, 128, 1),
        selection_k_rope=torch.empty(rows * capacity // 128, 128, 0),
        full_kv_cache=source,
        full_k_rope=torch.empty(4096, 0),
        full_kv_region=None,
        full_rope_region=None,
    )
    planner.gather_selection_misses(inputs)
    assert copied == [rows * (width - hits)]
    assert torch.equal(
        destination.view(rows, capacity, 1)[:, :width, 0],
        torch.arange(width).repeat(rows, 1) + 128,
    )
    assert (destination.view(rows, capacity, 1)[:, width:] == -999).all()
    planner.commit_layer_plan(cache, plan)
    assert not planner.materialize_layer_plan(
        cache, keys, slots, membership
    ).miss_mask.any()


@pytest.mark.parametrize("length", [2047, 2048, 2049])
@pytest.mark.parametrize("routes", [1, 6])
def test_causal_metadata_preserves_tail_and_masks_invalid_columns(
    modules, length, routes
):
    fsa, _ = modules
    rows = routes * 2
    topk = torch.arange(2051, dtype=torch.int32).repeat(rows, 1).unsqueeze(1)
    topk[:, :, 0] = -8
    requests = torch.arange(2).repeat_interleave(routes)
    ends = torch.tensor([routes, rows], dtype=torch.int32)
    lengths = torch.tensor([length, length + 128])
    positions = torch.cat(
        (
            torch.arange(length - routes, length),
            torch.arange(length + 128 - routes, length + 128),
        )
    )
    masked = fsa.mask_causal_topk(topk, requests, ends, lengths, positions=positions)
    assert (masked[:, :, 0] == -1).all()
    assert (masked[:, 0][torch.arange(2051)[None] > positions[:, None]] == -1).all()
    table = torch.stack((torch.arange(1, 19), torch.arange(19, 37))).int()
    plan = fsa.build_reference_sparse_plan(masked, table, lengths, ends, page_size=128)
    assert torch.equal(
        plan.physical_token_ids[:, 0],
        torch.where(
            masked[:, 0] >= 0, masked[:, 0] + (requests[:, None] * 18 + 1) * 128, -1
        ),
    )


@pytest.mark.parametrize("routes,capacity", [(1, 2176), (6, 12416)])
def test_lim_capacity_covers_union_and_native_limit(monkeypatch, routes, capacity):
    contract = load(monkeypatch, "lim_copy_sfa_contract")
    spec = contract.LimCopySfaSpec(1 << 20, routes, capacity)
    assert spec.token_topk == 2051
    assert spec.resident_tokens >= routes * spec.token_topk
    if routes == 6:
        with pytest.raises(ValueError, match="Resident tokens"):
            contract.LimCopySfaSpec(1 << 20, routes, 4096)


def test_generation_upload_is_stable_and_monotonic_after_clear(modules):
    fsa, _ = modules
    pool = SimpleNamespace(req_generation=torch.tensor([0, 4, 9]))
    state = fsa.FusedSparseAttentionState("cpu", pool)
    address = state.generations.data_ptr()
    version = state._generation_version
    state.refresh_generations()
    assert state._generation_version == version
    pool.req_generation.add_(1)
    state.refresh_generations()
    assert state.generations.tolist() == [1, 5, 10]
    assert state.generations.data_ptr() == address
    assert state._generation_version == version + 1


def test_host_table_scope_shares_only_one_preparation_and_mapping_epoch(monkeypatch):
    offload = load(monkeypatch, "hicache_offload")
    binding = offload.OffloadPageBinding.__new__(offload.OffloadPageBinding)
    binding.device_to_host = torch.tensor([0, 5, 9])
    binding.mapping_epoch = 0
    source = torch.tensor([[1, 2], [2, 0]], dtype=torch.int32)
    scope = offload.DraftHostTableScope(4)
    first = scope.host_table(binding, source)
    assert first.tolist() == [[5, 9], [9, 0]]
    assert scope.host_table(binding, source.clone()).data_ptr() == first.data_ptr()
    binding.mapping_epoch += 1
    binding.device_to_host[1] = 7
    changed = scope.host_table(binding, source)
    assert changed.tolist() == [[7, 9], [9, 0]]
    assert changed.data_ptr() != first.data_ptr()
    output = torch.full((2, 8), -1, dtype=torch.int32)
    pointer = output.data_ptr()
    fresh = offload.DraftHostTableScope(4)
    assert fresh.host_table(binding, source.flip(0), output).data_ptr() == pointer
    assert output[:, :2].tolist() == [[9, 0], [7, 9]]
    assert not output[:, 2:].any()


def test_mutable_tail_and_inactive_rows_do_not_publish_hits(modules):
    _, planner = modules
    applied = torch.full((2, 9), -1, dtype=torch.int64)
    cache = SimpleNamespace(
        selection_kv_cache=torch.empty(8, 2, 1),
        max_tokens=2,
        applied_keys=applied[:, :-1],
        applied_keys_with_padding=applied,
    )
    membership = torch.empty(2, 16400, dtype=torch.int16)
    keys = torch.tensor([[8, 9, 10, 11], [-1, -1, -1, -1]])
    slots = torch.arange(4).repeat(2, 1)
    stable = torch.tensor([[True, True, False, False], [False] * 4])
    first = planner.materialize_layer_plan(
        cache, keys, slots, membership, cacheable_mask=stable
    )
    planner.commit_layer_plan(cache, first)
    second = planner.materialize_layer_plan(
        cache, keys, slots, membership, cacheable_mask=stable
    )
    assert second.miss_mask.tolist() == [[False, False, True, True], [False] * 4]
    assert (membership[1, 16380:16384] == 0).all()
    assert (cache.applied_keys[1] == -1).all()


@pytest.mark.parametrize("change", ["generation", "mapping", "requests", "seed"])
def test_external_draft_reuse_identity_change_replans(modules, monkeypatch, change):
    fsa, planner = modules
    pool = SimpleNamespace(req_generation=torch.tensor([0, 1, 2]))
    state = fsa.FusedSparseAttentionState(
        "cpu", pool, external_planner=True, resident_capacity=8, mutable_tail_tokens=2
    )
    enqueue = []

    def ensure(layer, cache, topk):
        def plan(keys, owners, lengths):
            enqueue.append(keys.clone())
            return torch.arange(topk).repeat(keys.shape[0], 1)

        return layer, SimpleNamespace(enqueue=plan)

    monkeypatch.setattr(state, "_ensure_planner", ensure)
    cache = {}
    kwargs = dict(
        layer_id=0,
        q_nope=torch.zeros(2, 1, 4),
        q_rope=torch.empty(2, 1, 0),
        topk_indices=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32),
        full_kv=torch.empty(8, 2, 4),
        full_rope=torch.empty(8, 2, 0),
        block_table=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        kv_lengths=torch.tensor([4, 4]),
        query_lengths=torch.tensor([1, 2]),
        request_slots=torch.tensor([1, 2]),
        scale=0.5,
        draft_plan_cache=cache,
        draft_plan_identity=("generation", 0),
    )
    inputs = state.build_inputs(**kwargs)
    planner.commit_layer_plan(inputs.selection_cache, inputs.selection_plan)
    state.build_inputs(**kwargs)
    assert len(enqueue) == 1
    kwargs["draft_plan_identity"] = (change, 1)
    state.refresh_generations()
    state.build_inputs(**kwargs)
    assert len(enqueue) == 2


if __name__ == "__main__":
    sys.exit(run_pytest([__file__, *sys.argv[1:]]))
