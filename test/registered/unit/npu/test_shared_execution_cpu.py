"""Phase 3 shared execution contracts, CPU metadata/fault injection only."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu.hisparse import config
from sglang.srt.hardware_backend.npu.hisparse.external_sparse_planner import (
    ExternalPlanResidentState,
)
from external_plan_test_support import install_planner_double


def test_shared_default_and_explicit_old_abi(monkeypatch):
    monkeypatch.delenv("SGLANG_NPU_FSA_EXECUTION_ABI", raising=False)
    assert config.resolve_fsa_execution_abi() == "shared_v3"
    with pytest.raises(ValueError, match="removed"):
        config.resolve_fsa_execution_abi("row_v0")
    with pytest.raises(ValueError):
        config.resolve_fsa_execution_abi("unknown")




def test_forward_gate_raises_before_return_for_normal_native_tensor():
    from sglang.srt.hardware_backend.npu.hisparse.shared_execution import (
        checked_shared_forward,
        register_shared_resident,
    )

    resident = ExternalPlanResidentState(
        "cpu", 3, 4, max_routes=1, expanded_topk=2, page_size=1
    )
    with pytest.raises(RuntimeError, match="external_plan"):
        with checked_shared_forward():
            register_shared_resident(resident)
            resident.lifecycle_error.fill_(8)
            output = torch.ones(1)  # Native returned normally.
    assert output.item() == 1
    with checked_shared_forward():
        pass  # Non-shared forwards are a no-op.


def test_shared_rejects_reference_and_mapped_before_allocation(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        FusedSparseAttentionState,
    )

    pool = SimpleNamespace(req_generation=torch.zeros(3, dtype=torch.int64))
    with pytest.raises(ValueError, match="mapped"):
        FusedSparseAttentionState(
            "cpu", pool, external_planner=True, mapped_host_plan=True
        )


def test_batch_completion_is_checked_then_mapped_to_request_owner():
    from sglang.srt.hardware_backend.npu.hisparse.shared_execution import (
        expected_identity,
        complete_shared,
    )

    r = ExternalPlanResidentState(
        "cpu", 7, 4, max_routes=1, expanded_topk=2, page_size=1
    )
    owners = torch.tensor([5, 2])
    keys = torch.tensor([[20, 21], [30, -1]])
    slots = torch.tensor([[0, 1], [2, -1]])
    r.prepare(9, owners, torch.tensor([4, 8]), 3)
    r.record_plan(9, owners, keys, slots)
    lease = r.begin_execution(9, owners, physical_keys=keys, slots=slots)
    expected = expected_identity(r, lease, owners, 11)
    completion = expected.clone()
    completion[1, 2] += 1  # wrong physical layer for owner2
    complete_shared(
        r,
        lease,
        owners,
        expected,
        completion,
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([[2, 2], [2, 0]], dtype=torch.int32),
        torch.tensor([2, 1]),
    )
    assert r.layers[9]["payload_valid"][5, :2].all()
    assert not r.layers[9]["payload_valid"][2].any()
    assert r.layers[9]["in_flight"].sum() == 0
    assert r.lifecycle_error != 0
    complete_shared(
        r,
        lease,
        owners,
        expected,
        completion,
        torch.zeros(2, dtype=torch.int32),
        torch.full((2, 2), 2, dtype=torch.int32),
        torch.tensor([2, 1]),
    )
    assert r.layers[9]["in_flight"].sum() == 0


def make_shared(monkeypatch, *, routes=2, qcap=5, rows=4, batch=2, layer=0):
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        FusedSparseAttentionState,
    )

    install_planner_double(monkeypatch)
    state = FusedSparseAttentionState(
        "cpu",
        SimpleNamespace(req_generation=torch.arange(qcap)),
        external_planner=True,
        resident_capacity=128,
        row_capacity=rows,
        max_query_routes=routes,
        page_size=128,
        batch_capacity=batch,
    )
    full = (
        torch.arange(4 * 128 * 512, dtype=torch.float32)
        .reshape(4, 128, 512)
        .to(torch.bfloat16)
    )

    def build(topk=None, owners=None, ends=None, epoch=1):
        topk = (
            torch.tensor(
                [[0, 1, 1, 2], [1, 2, 3, -1], [0, 1, -1, -1], [-1] * 4],
                dtype=torch.int32,
            )
            if topk is None
            else topk
        )
        return state.build_inputs(
            layer_id=layer,
            q_nope=torch.zeros(rows, 32, 512, dtype=torch.bfloat16),
            q_rope=torch.empty(rows, 32, 0, dtype=torch.bfloat16),
            topk_indices=topk,
            full_kv=full,
            full_rope=full[:, :, :0],
            block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            kv_lengths=torch.tensor([128, 128]),
            query_lengths=torch.tensor([2, 4]) if ends is None else ends,
            request_slots=torch.tensor([3, 1]) if owners is None else owners,
            scale=0.1,
            mapping_epoch=epoch,
        )

    return state, build


def install_native_simulator(
    monkeypatch, *, status_error=False, fail=False, inspect=None
):
    import sglang.srt.hardware_backend.npu.fsa_native as native

    def op(*args):
        assert len(args) == 30
        assert args[6][-1].item() == args[0].shape[0]
        assert args[0].dtype == args[1].dtype == args[2].dtype == torch.bfloat16
        dtypes = {
            4: torch.int32,
            5: torch.int32,
            6: torch.int32,
            7: torch.int32,
            8: torch.int64,
            9: torch.int32,
            10: torch.int32,
            11: torch.int64,
            12: torch.int32,
            13: torch.int32,
            14: torch.bool,
            15: torch.int32,
            16: torch.bool,
            17: torch.int64,
            18: torch.int32,
            19: torch.int32,
            20: torch.int64,
            21: torch.int64,
            22: torch.int32,
            23: torch.int16,
            24: torch.int32,
            25: torch.int32,
            26: torch.uint8,
        }
        for index, dtype in dtypes.items():
            assert args[index].dtype == dtype, (index, args[index].dtype, dtype)
            assert args[index].is_contiguous(), index
        assert args[25].shape[-1] > args[7].shape[-1]
        if inspect:
            inspect(args)
        if fail:
            raise RuntimeError("simulated launch failure")
        identity, ready, status, completion, counters = args[17:22]
        count = args[13]
        assert ready.dtype == torch.int32
        ready.copy_(
            (torch.arange(ready.shape[1])[None, :] < count[:, None]).to(torch.int32) * 2
        )
        completion.copy_(identity)
        if status_error:
            status[0] = 16
        return torch.ones(args[0].shape, dtype=torch.bfloat16)

    install_planner_double(monkeypatch, attention=op)


def registered(inputs):
    from dataclasses import replace

    full = inputs.full_kv_cache
    return replace(
        inputs,
        full_kv_region=SimpleNamespace(
            host_ptr=full.data_ptr(), device_ptr=123456, nbytes=full.nbytes, device_id=0
        ),
    )


def test_shared_wrapper_reserves_qcap_payload_and_replans_hits(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        run_fused_sparse_attention,
    )

    state, build = make_shared(monkeypatch)
    install_native_simulator(monkeypatch)
    cold = registered(build())
    payload = cold.workspace.payload
    assert payload.shape == (5, 128, 512)
    assert not hasattr(cold.workspace, "applied_keys")
    assert cold.plan.union_count.tolist() == [4, 2]
    run_fused_sparse_attention(cold)
    warm = registered(build())
    assert warm.workspace.payload.data_ptr() == payload.data_ptr()
    assert not warm.plan.union_miss.any()
    assert not hasattr(state.planners[0], "enqueue")
    run_fused_sparse_attention(warm)
    layer1 = state.reserve(1, 4, 4, warm.full_kv_cache[:, :, :0], warm.full_kv_cache)
    assert layer1.payload.data_ptr() != payload.data_ptr()


def test_shared_wrapper_normal_tensor_error_is_raised_and_owner_invalidated(
    monkeypatch,
):
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        run_fused_sparse_attention,
    )

    state, build = make_shared(monkeypatch)
    install_native_simulator(monkeypatch, status_error=True)
    inputs = registered(build())
    with pytest.raises(RuntimeError, match="external_plan"):
        run_fused_sparse_attention(inputs)
    assert state.external_resident.layers[0]["in_flight"].sum() == 0
    assert not state.external_resident.layers[0]["payload_valid"][3].any()
    assert state.external_resident.layers[0]["payload_valid"][1].any()


def test_shared_wrapper_empty_request_is_native_padding(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        run_fused_sparse_attention,
    )

    state, build = make_shared(monkeypatch)

    def inspect(args):
        assert args[8].tolist() == [0, 1]
        assert args[9].tolist() == [-1, -1, 1, 1]
        assert (args[7][:2] == -1).all()
        assert (args[17][0] == -1).all()

    install_native_simulator(monkeypatch, inspect=inspect)
    topk = torch.tensor(
        [[-1] * 4, [-1] * 4, [0, 1, -1, -1], [-1] * 4], dtype=torch.int32
    )
    run_fused_sparse_attention(registered(build(topk=topk)))
    assert state.external_resident.layers[0]["in_flight"].sum() == 0
    assert state.external_resident.lifecycle_error == 0


@pytest.mark.parametrize("point", ["identity", "native", "publication"])
def test_shared_exception_transaction_releases_exact_lease(monkeypatch, point):
    import sglang.srt.hardware_backend.npu.hisparse.shared_execution as shared
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        run_fused_sparse_attention,
    )

    state, build = make_shared(monkeypatch)
    inputs = registered(build())
    install_native_simulator(monkeypatch, fail=point == "native")

    def broken(*a, **kw):
        raise RuntimeError("injected failure")

    if point == "identity":
        monkeypatch.setattr(shared, "expected_identity", broken)
    if point == "publication":
        monkeypatch.setattr(shared, "complete_shared", broken)
    with pytest.raises(RuntimeError):
        run_fused_sparse_attention(inputs)
    assert state.external_resident.layers[0]["in_flight"].sum() == 0
    assert not inputs.workspace.guard.active


def test_workspace_guard_rejects_second_producer_before_prepare(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.shared_execution import WorkspaceGuard

    guard = WorkspaceGuard("cpu")
    guard.acquire("capture")
    with pytest.raises(RuntimeError):
        guard.acquire("second")
    assert guard.stream == "capture"
    guard.active = False
    guard.fence = SimpleNamespace(query=lambda: False)
    with pytest.raises(RuntimeError):
        guard.acquire("replay")
    guard.drained()
    guard.acquire("replay")
    guard.release()
    guard.fail(RuntimeError("drain failed"))
    with pytest.raises(RuntimeError, match="quarantined"):
        guard.acquire("replay")


def test_budget_matches_actual_shared_storage_for_pd_graph_and_two_owners(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.shared_execution import (
        storage_inventory,
    )
    from sglang.srt.hardware_backend.npu.hisparse.decode_memory_budget import (
        shared_storage_bytes,
    )

    target, build = make_shared(monkeypatch, qcap=9, rows=6, batch=3)
    full = torch.empty(2, 128, 512, dtype=torch.bfloat16)
    target.reserve(0, 6, 4, full[:, :, :0], full)
    target.reserve(7, 6, 4, full[:, :, :0], full)
    draft, _ = make_shared(monkeypatch, qcap=11, rows=8, batch=4)
    draft.reserve(0, 8, 4, full[:, :, :0], full)
    actual = storage_inventory(target, draft)
    # All tensors are CPU in this allocation test. Logical device/CPU categories
    # are independently specified by shared_storage_bytes.
    t = shared_storage_bytes(9, 3, 6, 2, 4, 128, 128, 2)
    d = shared_storage_bytes(11, 4, 8, 1, 4, 128, 128, 2)
    assert actual["cpu_bytes"] == sum(t.values()) + sum(d.values())
    assert len(actual["storages"]) == len(
        {(v["device"], v["bytes"], v["paths"][0]) for v in actual["storages"]}
    )


@pytest.mark.parametrize("ready_state", [0, 1, 3])
def test_native_pending_or_failed_ready_cannot_publish(ready_state):
    from sglang.srt.hardware_backend.npu.hisparse.shared_execution import (
        expected_identity,
        complete_shared,
    )

    r = ExternalPlanResidentState(
        "cpu", 3, 4, max_routes=1, expanded_topk=2, page_size=1
    )
    owners = torch.tensor([1])
    keys = torch.tensor([[10, -1]])
    slots = torch.tensor([[0, -1]])
    r.prepare(0, owners, torch.tensor([1]), 0)
    r.record_plan(0, owners, keys, slots)
    lease = r.begin_execution(0, owners, physical_keys=keys, slots=slots)
    expected = expected_identity(r, lease, owners, 1)
    complete_shared(
        r,
        lease,
        owners,
        expected,
        expected.clone(),
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([[ready_state, 0]], dtype=torch.int32),
        torch.tensor([1]),
    )
    assert not r.layers[0]["payload_valid"][1].any()
    assert r.lifecycle_error != 0


def test_allocated_native_ready_has_exact_frozen_dtype(monkeypatch):
    state, build = make_shared(monkeypatch)
    inputs = build()
    assert inputs.workspace.ready.dtype == torch.int32


def test_failed_launch_keeps_guard_until_asynchronous_cancel_is_drained(monkeypatch):
    import sglang.srt.hardware_backend.npu.hisparse.shared_execution as shared
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        run_fused_sparse_attention,
    )

    state, build = make_shared(monkeypatch)
    inputs = registered(build())
    guard = inputs.workspace.guard
    install_native_simulator(monkeypatch, fail=True)
    original = state.external_resident.cancel_execution

    def cancel(*args, **kw):
        with pytest.raises(RuntimeError):
            guard.acquire("other-stream")
        return original(*args, **kw)

    monkeypatch.setattr(state.external_resident, "cancel_execution", cancel)
    with pytest.raises(RuntimeError, match="launch failure"):
        run_fused_sparse_attention(inputs)
    assert not guard.active


@pytest.mark.parametrize("update_inputs", [False, True])
@pytest.mark.parametrize("is_draft", [False, True])
def test_actual_graph_replay_result_gate_blocks_native_tensor_errors(
    monkeypatch, update_inputs, is_draft
):
    from concurrent.futures import ThreadPoolExecutor
    from test_external_plan_fsa_lifecycle_cpu import load_graph_backend
    from sglang.srt.hardware_backend.npu.hisparse.fsa_graph_lifecycle import (
        FsaGraphReaders,
    )
    from sglang.srt.hardware_backend.npu.hisparse.shared_execution import (
        WorkspaceGuard,
        checked_shared_forward,
    )

    resident = ExternalPlanResidentState(
        "cpu", 3, 4, max_routes=1, expanded_topk=2, page_size=1
    )
    readers = FsaGraphReaders()
    guard = WorkspaceGuard("cpu")
    readers.add_shared(resident, guard, None, None, ())
    backend = object.__new__(load_graph_backend(monkeypatch))

    def replay():
        resident.lifecycle_error.fill_(16)

    class Event:
        def record(self, *a):
            pass

        def synchronize(self):
            pass

    stream = SimpleNamespace(synchronize=lambda: None)
    backend._graphs = {1: SimpleNamespace(replay=replay, update=lambda **kw: None)}
    backend._outputs = {1: torch.ones(1)}
    backend._fsa_readers = {1: readers}
    backend._device_module = SimpleNamespace(Event=Event, current_stream=lambda: stream)
    backend._rebind_fence = None
    backend._update_executor = ThreadPoolExecutor(1)
    consumed = []
    try:
        with pytest.raises(RuntimeError, match="external_plan"):
            with checked_shared_forward():
                output = (
                    backend.replay_with_input_update(1, [4], "seq_lens")
                    if update_inputs
                    else backend.replay(1, None)
                )
            consumed.append(("draft" if is_draft else "target", output))
        assert not consumed
    finally:
        backend._update_executor.shutdown()


@pytest.mark.parametrize("is_draft", [False, True])
def test_actual_model_runner_forward_gates_before_consumer(monkeypatch, is_draft):
    # Execute the real forward method while replacing unrelated distributed
    # services; no fake forward wrapper can hide a misplaced result gate.
    import ast
    import contextlib
    from pathlib import Path
    from sglang.srt.hardware_backend.npu.hisparse.shared_execution import (
        register_shared_resident,
    )

    root = Path(__file__).resolve().parents[4]
    tree = ast.parse(
        (root / "python/sglang/srt/model_executor/model_runner.py").read_text()
    )
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            fn,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    @contextlib.contextmanager
    def recording(*a, **kw):
        yield {}

    env = {
        "profile_range": lambda *a, **kw: contextlib.nullcontext(),
        "build_step_span_name": lambda *a, **kw: "step",
        "contextlib": contextlib,
        "get_global_expert_distribution_recorder": lambda: SimpleNamespace(
            with_forward_pass=recording
        ),
        "get_schedule": lambda: SimpleNamespace(disable_overlap_schedule=False),
        "get_global_experts_capturer": lambda: None,
        "get_global_indexer_capturer": lambda: None,
        "dumper": SimpleNamespace(may_enable=False),
        "get_exec": lambda: SimpleNamespace(
            moe=SimpleNamespace(elastic_ep_backend=None)
        ),
    }
    exec(compile(module, "model_runner.forward.actual", "exec"), env)
    resident = ExternalPlanResidentState(
        "cpu", 3, 4, max_routes=1, expanded_topk=2, page_size=1
    )

    def raw(*a):
        register_shared_resident(resident)
        resident.lifecycle_error.fill_(8)
        return SimpleNamespace()

    runner = SimpleNamespace(
        forward_pass_id=0,
        msprobe_debugger=None,
        is_draft_worker=is_draft,
        canary_manager=None,
        _forward_raw=raw,
        enable_elastic_ep=False,
        eplb_manager=None,
    )
    batch = SimpleNamespace(apply_deprecated_skip_attn_backend_init=lambda _: None)
    with pytest.raises(RuntimeError, match="external_plan"):
        env["forward"](runner, batch)


def test_actual_capture_snapshot_storage_is_charged_per_bucket_and_step(monkeypatch):
    import sglang.srt.hardware_backend.npu.hisparse.shared_execution as shared
    from sglang.srt.hardware_backend.npu.hisparse.decode_memory_budget import (
        shared_storage_bytes,
        shared_capture_storage_bytes,
    )
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        run_fused_sparse_attention,
    )

    state, build = make_shared(monkeypatch)
    retained = []
    original = shared.expected_identity
    expected_objects = []

    def identity(resident, lease, owners, instance):
        expected = original(resident, lease, owners, instance)
        expected_objects.append((lease, expected))
        return expected

    monkeypatch.setattr(shared, "expected_identity", identity)

    def inspect(args):
        retained.append((args[8], args[7], args[9], args[10]))

    install_native_simulator(monkeypatch, inspect=inspect)
    snapshots = []
    for _ in range(3):
        inputs = registered(build())
        run_fused_sparse_attention(inputs)
        snapshots.append(
            (
                *expected_objects[-1],
                inputs.plan,
                inputs.selection_topk_indices,
                *retained[-1],
            )
        )
    actual = shared.storage_inventory(state, snapshots)["cpu_bytes"]
    base = sum(shared_storage_bytes(5, 2, 4, 1, 4, 128, 128, 2).values())
    assert actual == base + 3 * shared_capture_storage_bytes(5, 2, 4, 4)


@pytest.mark.parametrize("column", range(9))
def test_each_identity_column_is_required_before_publication(column):
    from sglang.srt.hardware_backend.npu.hisparse.shared_execution import (
        expected_identity,
        complete_shared,
    )

    r = ExternalPlanResidentState(
        "cpu", 3, 4, max_routes=1, expanded_topk=2, page_size=1
    )
    owners = torch.tensor([1])
    keys = torch.tensor([[10, -1]])
    slots = torch.tensor([[0, -1]])
    r.prepare(0, owners, torch.tensor([1]), 0)
    r.record_plan(0, owners, keys, slots)
    lease = r.begin_execution(0, owners, physical_keys=keys, slots=slots)
    expected = expected_identity(r, lease, owners, 1)
    actual = expected.clone()
    actual[0, column] += 1
    complete_shared(
        r,
        lease,
        owners,
        expected,
        actual,
        torch.zeros(1, dtype=torch.int32),
        torch.tensor([[2, 0]], dtype=torch.int32),
        torch.tensor([1]),
    )
    assert not r.layers[0]["payload_valid"].any()
    assert r.layers[0]["in_flight"].sum() == 0


def test_begin_snapshot_allocation_failure_never_acquires_reference(monkeypatch):
    r = ExternalPlanResidentState(
        "cpu", 3, 4, max_routes=1, expanded_topk=2, page_size=1
    )
    original = torch.Tensor.clone
    calls = []

    def fail_clone(t, *a, **kw):
        calls.append(t)
        if len(calls) == 5:
            raise RuntimeError("snapshot allocation failure")
        return original(t, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "clone", fail_clone)
    with pytest.raises(RuntimeError, match="snapshot"):
        r.begin_execution(
            0,
            torch.tensor([1]),
            physical_keys=torch.tensor([[4, -1]]),
            slots=torch.tensor([[0, -1]]),
        )
    assert r.layers[0]["in_flight"].sum() == 0
    assert r.layers[0]["execution_serial"].sum() == 0


def test_reentrant_build_is_rejected_before_shared_metadata_changes(monkeypatch):
    state, build = make_shared(monkeypatch)
    first = build()
    keys = first.plan.union_keys.clone()
    epoch = state.mapping_epoch_input.clone()
    with pytest.raises(RuntimeError, match="overlapping"):
        build(epoch=9)
    assert torch.equal(first.plan.union_keys, keys)
    assert torch.equal(state.mapping_epoch_input, epoch)


def test_forward_error_read_is_aggregated_once_for_multiple_layers(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.shared_execution import (
        checked_shared_forward,
        register_shared_resident,
    )

    a = ExternalPlanResidentState(
        "cpu", 3, 4, max_routes=1, expanded_topk=2, page_size=1
    )
    b = ExternalPlanResidentState(
        "cpu", 3, 4, max_routes=1, expanded_topk=2, page_size=1
    )
    original = torch.Tensor.item
    reads = []

    def item(t, *args):
        reads.append(t)
        return original(t, *args)

    monkeypatch.setattr(torch.Tensor, "item", item)
    with checked_shared_forward():
        register_shared_resident(a)
        register_shared_resident(a)
        register_shared_resident(b)
    assert len(reads) == 1


def test_qcap_and_graph_bcap_are_independent_reservations(monkeypatch):
    state, _ = make_shared(monkeypatch, qcap=3, rows=8, batch=4)
    full = torch.empty(1, 128, 512, dtype=torch.bfloat16)
    layer = state.reserve(0, 8, 4, full[:, :, :0], full)
    assert layer.payload.shape == (3, 128, 512)
    assert layer.ready.shape[0] == 4


def test_native_batch_limit_rejected_before_tensor_allocation(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        FusedSparseAttentionState,
    )

    pool = SimpleNamespace(req_generation=torch.zeros(2048, dtype=torch.int64))
    with pytest.raises(ValueError, match="Bcap"):
        FusedSparseAttentionState(
            "cpu",
            pool,
            external_planner=True,
            page_size=128,
            row_capacity=1025,
            batch_capacity=1025,
        )


@pytest.mark.parametrize("update_inputs", [False, True])
def test_partial_graph_refresh_failure_drains_before_releasing_workspace(
    monkeypatch, update_inputs
):
    from concurrent.futures import ThreadPoolExecutor
    from test_external_plan_fsa_lifecycle_cpu import load_graph_backend
    from sglang.srt.hardware_backend.npu.hisparse.fsa_graph_lifecycle import (
        FsaGraphReaders,
    )
    from sglang.srt.hardware_backend.npu.hisparse.shared_execution import WorkspaceGuard

    resident = ExternalPlanResidentState(
        "cpu", 3, 4, max_routes=1, expanded_topk=2, page_size=1
    )
    guard = WorkspaceGuard("cpu")
    readers = FsaGraphReaders()
    readers.add_shared(resident, guard, None, None, ())
    pending = []
    drains = []

    def refresh():
        pending.append("queued-write")
        raise RuntimeError("partial refresh")

    readers.states[1] = SimpleNamespace(refresh_generations=refresh)

    def drain():
        assert guard.active
        drains.append("drain")
        pending.clear()

    backend = object.__new__(load_graph_backend(monkeypatch))
    backend._graphs = {
        1: SimpleNamespace(replay=lambda: None, update=lambda **kw: None)
    }
    backend._outputs = {1: None}
    backend._fsa_readers = {1: readers}
    backend._device_module = SimpleNamespace(
        current_stream=lambda: SimpleNamespace(synchronize=drain)
    )
    backend._rebind_fence = None
    backend._update_executor = ThreadPoolExecutor(1)
    try:
        with pytest.raises(RuntimeError, match="partial refresh"):
            if update_inputs:
                backend.replay_with_input_update(1, [4], "seq_lens")
            else:
                backend.replay(1, None)
        assert drains and not pending and not guard.active
    finally:
        backend._update_executor.shutdown()


def test_invalid_execution_mode_releases_prepared_workspace(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        run_fused_sparse_attention,
    )

    state, build = make_shared(monkeypatch)
    inputs = registered(build())
    with pytest.raises(ValueError, match="fused"):
        run_fused_sparse_attention(inputs, mode="reference")
    assert not inputs.workspace.guard.active


def test_shared_target_draft_geometry_mismatch_rejected_before_allocation(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        FusedSparseAttentionState,
    )

    pool = SimpleNamespace(req_generation=torch.zeros(5, dtype=torch.int64))
    full = torch.empty(1, 128, 512, dtype=torch.bfloat16)
    target = FusedSparseAttentionState(
        "cpu",
        pool,
        external_planner=True,
        page_size=128,
        resident_capacity=128,
        row_capacity=4,
        batch_capacity=2,
        max_query_routes=2,
    )
    target.reserve(0, 4, 4, full[:, :, :0], full)
    draft = FusedSparseAttentionState(
        "cpu",
        pool,
        external_planner=True,
        page_size=128,
        resident_capacity=256,
        row_capacity=4,
        batch_capacity=2,
        max_query_routes=2,
    )
    allocations = []
    original = torch.empty

    def allocate(*args, **kwargs):
        allocations.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    with pytest.raises(ValueError, match="geometry"):
        draft.reserve(0, 4, 129, full[:, :, :0], full)
    assert not allocations


def test_empty_selection_native_error_is_not_hidden_by_absent_lease(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        run_fused_sparse_attention,
    )

    state, build = make_shared(monkeypatch)
    install_native_simulator(monkeypatch, status_error=True)
    inputs = registered(build(topk=torch.full((4, 4), -1, dtype=torch.int32)))
    with pytest.raises(RuntimeError, match="external_plan"):
        run_fused_sparse_attention(inputs)
    assert state.external_resident.layers[0]["in_flight"].sum() == 0


@pytest.mark.parametrize("entry", ["allocator", "budget"])
def test_shared_union_width_limit_rejected_before_allocation(monkeypatch, entry):
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        FusedSparseAttentionState,
    )
    from sglang.srt.hardware_backend.npu.hisparse.decode_memory_budget import (
        shared_storage_bytes,
    )

    pool = SimpleNamespace(req_generation=torch.zeros(3, dtype=torch.int64))
    with pytest.raises(ValueError):
        if entry == "allocator":
            FusedSparseAttentionState("cpu", pool, external_planner=True, page_size=128,
                                     resident_capacity=16512, row_capacity=8,
                                     batch_capacity=1, max_query_routes=8)
        else:
            shared_storage_bytes(3, 1, 8, 1, 2051, 16512, 128, 8)


def test_reserved_native_workspace_is_persistent_once_per_layer_and_shared_across_buckets(
    monkeypatch,
):
    from sglang.srt.hardware_backend.npu.hisparse.shared_execution import (
        storage_inventory,
    )
    from sglang.srt.hardware_backend.npu.hisparse.decode_memory_budget import (
        shared_storage_bytes,
    )
    from sglang.srt.hardware_backend.npu.hisparse.fused_sparse_attention import (
        run_fused_sparse_attention,
    )

    state, build = make_shared(monkeypatch)
    state.shared_native_workspace_bytes = 4096
    install_native_simulator(monkeypatch)
    pointers = []
    for _ in range(3):
        inputs = registered(build())
        assert inputs.workspace.native_workspace.dtype == torch.uint8
        assert inputs.workspace.native_workspace.numel() == 4096
        pointers.append(inputs.workspace.native_workspace.data_ptr())
        run_fused_sparse_attention(inputs)
    assert len(set(pointers)) == 1
    state.reserve(7, 4, 4, inputs.full_kv_cache[:, :, :0], inputs.full_kv_cache)
    assert state.layers[7].native_workspace.data_ptr() != pointers[0]
    expected = shared_storage_bytes(
        5, 2, 4, 2, 4, 128, 128, 2, native_workspace_bytes=4096
    )
    actual = storage_inventory(state, state.layers[0].native_workspace[:4])
    assert actual["cpu_bytes"] == sum(expected.values())


def test_reserved_workspace_budget_is_not_multiplied_by_graph_bucket_count():
    from dataclasses import replace
    from sglang.srt.hardware_backend.npu.hisparse.decode_memory_budget import (
        DecodeKvMemoryBudget,
    )

    budget = DecodeKvMemoryBudget(
        layers=3,
        planner_groups=1,
        latent_dim=512,
        rope_dim=0,
        index_dim=8,
        element_bytes=2,
        topk=4,
        resident_capacity=128,
        host_bytes=1 << 30,
        rows_per_request=2,
        graph_rows=3,
        callback_instances=4,
        native_workspace_bytes=4096,
    )
    before = replace(budget, native_workspace_bytes=0)
    assert budget.fixed_bytes(2, 128)[1] - before.fixed_bytes(2, 128)[1] == 4096 * 3
    assert (
        budget.shared_persistent_bytes(2, 128)["device"]
        - before.shared_persistent_bytes(2, 128)["device"]
        == 4096 * 3
    )


