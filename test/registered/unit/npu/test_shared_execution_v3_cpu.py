"""ABI3 Python contracts. Native/Attention below are explicit CPU doubles."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu.hisparse import (
    config,
    shared_execution as execution,
)
from sglang.srt.hardware_backend.npu.hisparse.decode_memory_budget import (
    shared_storage_bytes,
)
from sglang.srt.hardware_backend.npu.hisparse.external_plan_fused import (
    PLANNER_CAPABILITIES,
)
from test_shared_execution_cpu import make_shared, registered

_spec = importlib.util.spec_from_file_location(
    "protected_hash_test_reference",
    Path(__file__).resolve().parents[4] / "native/fsa/tests/planner_reference.py",
)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)


def install_double(monkeypatch, *, inspect=None):
    import sglang.srt.hardware_backend.npu.fsa_native as native

    calls = []

    def planner(i, o, C, R, host_tokens, version):
        calls.append("planner")
        assert version == 3 and len(i) == 14 and len(o) == 23
        assert o[18].shape[1] == 8 and o[15].shape[1] % 16 == 0
        T, K = i[0].shape
        for n in [0, 4, 5, 6, 7, 8, 9, 17, 19]:
            o[n].fill_(-1)
        for n in [1, 2, 3, 10, 11, 12, 13, 15, 16, 18, 20, 21, 22]:
            o[n].zero_()
        begin = 0
        for b, end in enumerate(i[1].tolist()):
            owner = int(i[2][b])
            if owner <= 0:
                begin = end
                continue
            expected = _reference.reference(
                i[0][begin:end].tolist(),
                i[3][begin:end].tolist(),
                i[6][owner].tolist(),
                i[7][owner].tolist(),
                i[8][owner].tolist(),
                i[9][owner].tolist(),
                length=int(i[5][b]),
            )
            i[9][owner].copy_(torch.tensor(expected["reusable"]))
            count = len(expected["keys"])
            o[12][b] = count
            for n, name in [(7, "keys"), (8, "slots"), (10, "miss"), (11, "cacheable")]:
                o[n][b, :count] = torch.tensor(expected[name], dtype=o[n].dtype)
            o[9][b, :count] = (
                torch.tensor(expected["owner"], dtype=torch.int32) + begin * K
            )
            for local, u in enumerate(expected["row_union"]):
                row, col = begin + local // K, local % K
                o[5][row] = b
                if u < 0:
                    continue
                slot, miss, cacheable = (
                    expected["slots"][u],
                    expected["miss"][u],
                    expected["cacheable"][u],
                )
                o[0][row, col] = slot
                o[1][row, col] = miss
                o[2][row, col] = miss and expected["owner"][u] == local
                o[3][row, col] = cacheable
                o[4][row] = owner
                o[6][row, col] = u
                o[15][b, u] = 1 if miss else 2
                o[19][row, 0, col] = i[4][row, 0, col]
            begin = end

    def attention(*args):
        calls.append("attention")
        assert len(args) == 30 and args[-1] == 3
        assert args[21].shape[1] == 8
        if inspect:
            inspect(args)
        args[18].copy_(
            (torch.arange(args[18].shape[1])[None] < args[13][:, None]).int() * 2
        )
        args[20].copy_(args[17])
        return torch.zeros_like(args[0])

    monkeypatch.setattr(
        native,
        "native_ops",
        lambda: SimpleNamespace(
            external_plan_prepare=planner,
            npu_fused_sparse_attention_overlap=attention,
        ),
    )
    return calls


def test_protected_hash_reference_keeps_full_identity_and_mutable_aliases():
    expected = _reference.reference(
        [[0, (1 << 40), 0, 7]],
        [[False, True, True, True]],
        [0, 0, 7, 100, 200, 300, -1, -1],
        [True] * 6 + [False] * 2,
        [True] * 6 + [False] * 2,
        [True] * 6 + [False] * 2,
        length=128,
    )
    assert expected["keys"] == [0, 1 << 40, 7]
    assert expected["slots"] == [6, 7, 2]
    assert expected["miss"] == [True, True, False]
    assert expected["reusable"][:3] == [False, False, True]
    assert expected["row_union"] == [0, 1, 0, 2]


def test_fused_dispatches_one_planner_and_one_original_attention(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_FSA_LOAD_MODE", "FUSED")
    # Old policy is a legacy control only; no new policy selector is required.
    calls = install_double(monkeypatch)
    state, build = make_shared(monkeypatch)
    inputs = registered(build())
    assert inputs.workspace.planner.replacement_policy == "protected_hash"
    assert inputs.workspace.planner.union_width == 16
    assert not hasattr(inputs.workspace, "load_stream")
    execution.run_shared(inputs)
    assert calls == ["planner", "attention"]
    assert state.external_resident.layers[0]["in_flight"].sum() == 0


def test_final_planner_safe_indices_and_late_lease_gate_both_apply(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_FSA_LOAD_MODE", "FUSED")
    observed = []

    def inspect(args):
        observed.append(args[7].clone())
        assert args[7].data_ptr() != args[22].data_ptr()
        assert args[7][0, 0, 0] == -1  # final planner masking
        assert args[7][2:].eq(-1).all()  # late lease rejects owner1
        assert args[8].tolist() == [3, 0]

    install_double(monkeypatch, inspect=inspect)
    state, build = make_shared(monkeypatch)
    inputs = registered(build())
    assert inputs.selection_topk_indices[0, 0, 0] >= 0
    inputs.workspace.safe_topk[0, 0, 0] = -1
    state.external_resident.blocked_requests[1] = True
    execution.run_shared(inputs)
    assert len(observed) == 1


def test_fused_preserves_prepared_error_even_without_a_lease(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_FSA_LOAD_MODE", "FUSED")

    def inspect(args):
        assert args[19][0] == 17
        assert args[7].eq(-1).all()
        assert args[8].eq(0).all()

    install_double(monkeypatch, inspect=inspect)
    state, build = make_shared(monkeypatch)
    inputs = registered(build())
    inputs.workspace.status[0] = 17
    state.external_resident.blocked_requests.fill_(True)
    with pytest.raises(RuntimeError, match="external_plan"):
        execution.run_shared(inputs)
    assert state.external_resident.lifecycle_error != 0


def test_fused_tensor_budget_matches_actual_storage(monkeypatch):
    install_double(monkeypatch)
    state, build = make_shared(monkeypatch)
    inputs = build()
    inputs.workspace.guard.drained()
    actual = execution.storage_inventory(state)["cpu_bytes"]
    expected = shared_storage_bytes(5, 2, 4, 1, 4, 128, 128, 2)
    assert actual == sum(expected.values())


@pytest.mark.parametrize(
    "field",
    [
        "shared_v3_capabilities",
        "external_plan_prepare_capabilities",
        "external_plan_prepare",
        "npu_fused_sparse_attention_overlap",
    ],
)
def test_fused_rejects_old_binary_before_allocation(monkeypatch, field):
    import sglang.srt.hardware_backend.npu.fsa_native as native

    entries = dict(
        shared_v3_capabilities=lambda: [
            3,
            1,
            32,
            512,
            0,
            128,
            2051,
            32767,
            32767,
            9,
            1024,
            1,
            8,
            16,
            1,
        ],
        external_plan_prepare_capabilities=lambda: PLANNER_CAPABILITIES,
        external_plan_prepare=lambda: None,
        npu_fused_sparse_attention_overlap=lambda: None,
        shared_v3_workspace_bytes=lambda: 128,
    )
    entries.pop(field)
    monkeypatch.setattr(native, "native_ops", lambda: SimpleNamespace(**entries))
    with pytest.raises(RuntimeError, match="ABI3"):
        native.require_shared_v3_capability()


def test_fused_rejects_unsupported_capacity_and_retains_warmup():
    assert config.resolve_shared_load_mode("FUSED") == "FUSED"
    with pytest.raises(ValueError, match="external_plan"):
        shared_storage_bytes(5, 2, 4, 1, 4, 16512, 128, 2, load_mode="FUSED")
    cold = shared_storage_bytes(5, 2, 4, 1, 4, 128, 128, 2, load_mode="FUSED")
    warm = shared_storage_bytes(
        5, 2, 4, 1, 4, 128, 128, 2, load_mode="FUSED", warmup_threshold=128
    )
    assert warm["device_layers"] - cold["device_layers"] == 32 * 128 + 96 + 96 * 5


def test_fused_retains_independent_prefix_warmup_reservation(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_FSA_LOAD_MODE", "FUSED")
    monkeypatch.setenv("SGLANG_NPU_FSA_WARMUP_THRESHOLD", "128")
    install_double(monkeypatch)
    state, build = make_shared(monkeypatch)
    inputs = build()
    ws = inputs.workspace
    assert ws.counters.shape == (2, 8)
    assert ws.warmup.counters.shape == (1, 2)
    assert ws.planner.replacement_policy == "protected_hash"
    assert not hasattr(ws.warmup.planner, "replacement_policy")
    ws.guard.drained()


def test_v3_capability_is_independent_of_legacy_v1_symbols(monkeypatch):
    import sglang.srt.hardware_backend.npu.fsa_native as native

    ops = SimpleNamespace(
        shared_v3_capabilities=lambda: [
            3,
            1,
            32,
            512,
            0,
            128,
            2051,
            32767,
            32767,
            9,
            1024,
            1,
            8,
            16,
            1,
        ],
        external_plan_prepare_capabilities=lambda: PLANNER_CAPABILITIES,
        external_plan_prepare=lambda: None,
        npu_fused_sparse_attention_overlap=lambda: None,
        shared_v3_workspace_bytes=lambda: 4096,
    )
    monkeypatch.setattr(native, "native_ops", lambda: ops)
    assert native.shared_v3_workspace_bytes() == 4096
    assert not hasattr(ops, "shared_capabilities")
    assert not hasattr(ops, "shared_workspace_bytes")


def test_capturing_run_retains_only_current_abi3_inputs(monkeypatch):
    """Exercise run_shared's capture branch with native CPU doubles."""
    from sglang.srt.hardware_backend.npu.hisparse import fsa_graph_lifecycle

    calls = install_double(monkeypatch)
    state, build = make_shared(monkeypatch)
    inputs = registered(build())
    # Keep real resident tensors and guard on CPU; only the stream-capture
    # predicate is simulated, so the whole planner/consumer/complete path runs.
    state.device = SimpleNamespace(type="npu", index=inputs.full_kv_region.device_id)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), raising=False)
    retained = []
    monkeypatch.setattr(fsa_graph_lifecycle, "register_capture_shared", lambda *args: retained.append(args))
    result = execution.run_shared(inputs)
    assert result.shape == inputs.query.shape
    assert calls == ["planner", "attention"]
    assert len(retained) == 3
    assert len(retained[0][4]) == 0
    assert len(retained[1][4]) == 1  # lease retained before native enqueue
    assert len(retained[2][4]) == 9
    assert retained[2][4][-1] is inputs.workspace.native_workspace
    assert not inputs.workspace.guard.active
    assert state.external_resident.layers[0]["in_flight"].sum() == 0
