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
    plan = fsa.translate_logical_selection(masked, table, lengths, ends, page_size=128)
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
    state = fsa.FusedSparseAttentionState("cpu", pool, page_size=128)
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






if __name__ == "__main__":
    sys.exit(run_pytest([__file__, *sys.argv[1:]]))
