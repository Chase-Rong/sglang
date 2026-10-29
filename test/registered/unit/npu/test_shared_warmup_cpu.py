"""Phase4 Python contracts. Every native/stream implementation here is a CPU double."""

from contextlib import contextmanager
from dataclasses import replace, FrozenInstanceError
from types import SimpleNamespace
import pytest
import torch

from sglang.srt.hardware_backend.npu.hisparse import (
    config,
    shared_execution as execution,
)
from sglang.srt.hardware_backend.npu.hisparse.shared_warmup import warm_committed_prefix
from test_shared_execution_cpu import make_shared, registered
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class StreamDouble:
    def __init__(self, name, log, fail=False):
        self.npu_stream = name
        self.name, self.log, self.fail = name, log, fail

    def synchronize(self):
        self.log.append((self.name, "drain"))
        if self.fail:
            raise RuntimeError(self.name + " drain fault")

    def wait_event(self, event):
        self.log.append((self.name, "wait", event.name))


class EventDouble:
    def __init__(self, name, log, fail=False):
        self.name, self.log, self.fail = name, log, fail

    def record(self, stream):
        self.log.append((stream.name, "record", self.name))
        if self.fail:
            raise RuntimeError("record fault")

    def query(self):
        return True


def warm_fixture(monkeypatch, threshold=128):
    monkeypatch.setenv("SGLANG_NPU_FSA_WARMUP_THRESHOLD", str(threshold))
    state, build = make_shared(monkeypatch)
    inputs = registered(build())
    inputs.workspace.guard.drained()
    failures = []
    event = object()
    binding = SimpleNamespace(
        ready_event=event,
        mapping_epoch=1,
        check_reader_health=lambda: None,
        mark_reader_failure=failures.append,
    )
    kwargs = dict(
        owner=3,
        generation=3,
        committed_length=6,
        mapping_epoch=1,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        full_kv=inputs.full_kv_cache,
        region=inputs.full_kv_region,
        binding=binding,
        publication_length=128,
        publication_event=event,
    )
    return state, kwargs


def fill_double(monkeypatch, *, fail=False, stale_column=None, on_fill=None):
    import sglang.srt.hardware_backend.npu.fsa_native as native

    calls = []

    def op(*args):
        calls.append(args)
        (
            payload,
            full,
            _,
            _,
            lengths,
            owners,
            keys,
            slots,
            counts,
            misses,
            logical,
            cacheable,
            expected,
            ready,
            status,
            completion,
            counters,
            capacity,
        ) = args
        assert counts.tolist() == lengths.tolist()
        assert keys.shape == (1, capacity)
        assert ready.dtype == torch.int32
        assert len(args) == 18
        for j in range(int(counts[0])):
            assert int(logical[0, j]) < int(lengths[0])
            if bool(misses[0, j]):
                source = int(keys[0, j])
                target = int(owners[0]) * capacity + int(slots[0, j])
                payload.view(-1, 512)[target].copy_(full.view(-1, 512)[source])
        count = int(counts[0])
        assert count == int(lengths[0])
        assert (keys[0, 1:count] > keys[0, :count - 1]).all()
        assert sorted(logical[0, :count].tolist()) == list(range(count))
        expected_keys = args[3][0, logical[0, :count].long() // 128].long() * 128 + logical[0, :count].long() % 128
        assert torch.equal(keys[0, :count], expected_keys)
        assert slots[0, :count].unique().numel() == count
        ready.zero_()
        ready[0, : int(counts[0])] = 2
        status.zero_()
        counters.zero_()
        counters[0] = misses.sum()
        completion.copy_(expected)
        if stale_column is not None:
            completion[0, stale_column] += 1
        if on_fill:
            on_fill()
        if fail:
            status[0] = 8

    ops = native.native_ops()
    ops.external_plan_fill_only = op
    monkeypatch.setattr(native, "native_ops", lambda: ops)
    return calls






@pytest.mark.parametrize("value", [-1, 32768, "bad"])
def test_bad_warmup_threshold(value):
    with pytest.raises(ValueError):
        config.resolve_shared_warmup_threshold(value)






@pytest.mark.parametrize(
    "main_fail,load_fail", [(True, False), (False, True), (True, True)]
)
def test_both_streams_attempted_even_after_failure(main_fail, load_fail):
    log = []
    with pytest.raises(RuntimeError, match="drain failed"):
        execution.drain_streams(
            [StreamDouble("main", log, main_fail), StreamDouble("load", log, load_fail)]
        )
    assert log == [("main", "drain"), ("load", "drain")]




def test_prefix_wider_than_k_fills_and_exact_marker_and_one_shot(monkeypatch):
    state, args = warm_fixture(monkeypatch)
    calls = fill_double(monkeypatch)
    assert warm_committed_prefix(state, 0, **args)
    w = state.layers[0].warmup
    assert w.planner.capacity == 128 and w.planner.union_width == 128
    assert w.marker[3, :4].tolist() == [3, 3, 1, 6]
    assert int(w.counters[0, 0]) == 6
    assert not warm_committed_prefix(state, 0, **args)
    assert not warm_committed_prefix(state, 0, **(args | {"committed_length": 7}))
    assert len(calls) == 1
    # Eviction revokes residency without retriggering full warmup.
    w.marker[3] = -1
    assert not warm_committed_prefix(state, 0, **args)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "length,threshold,expected",
    [
        (0, 128, False),
        (5, 6, True),
        (6, 6, True),
        (7, 6, False),
        (127, 128, True),
        (128, 128, True),
        (129, 256, False),
    ],
)
def test_prefix_threshold_capacity_boundary(monkeypatch, length, threshold, expected):
    state, args = warm_fixture(monkeypatch, threshold)
    calls = fill_double(monkeypatch)
    assert (
        warm_committed_prefix(state, 0, **(args | {"committed_length": length}))
        is expected
    )
    assert len(calls) == int(expected)


@pytest.mark.parametrize("column", range(9))
def test_stale_identity_never_marks_warm(monkeypatch, column):
    state, args = warm_fixture(monkeypatch)
    fill_double(monkeypatch, stale_column=column)
    with pytest.raises(RuntimeError, match="external_plan"):
        warm_committed_prefix(state, 0, **args)
    w = state.layers[0].warmup
    assert (w.marker == -1).all() and (w.stamp == -1).all()
    assert not state.layers[0].guard.active


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", 4),
        ("publication_length", 5),
        ("publication_event", None),
        ("mapping_epoch", 2),
    ],
)
def test_unproven_or_stale_prefix_skipped(monkeypatch, field, value):
    state, args = warm_fixture(monkeypatch)
    calls = fill_double(monkeypatch)
    assert not warm_committed_prefix(state, 0, **(args | {field: value}))
    assert not calls


def test_producer_error_and_content_change_never_publish(monkeypatch):
    state, args = warm_fixture(monkeypatch)
    fill_double(monkeypatch, fail=True)
    with pytest.raises(RuntimeError):
        warm_committed_prefix(state, 0, **args)
    assert (state.layers[0].warmup.marker == -1).all()


def test_content_invalidation_rewarms_but_plain_append_does_not(monkeypatch):
    state, args = warm_fixture(monkeypatch)
    calls = fill_double(monkeypatch)
    assert warm_committed_prefix(state, 0, **args)
    state.invalidate_warmup_content(torch.tensor([3]))
    assert warm_committed_prefix(state, 0, **args)
    assert len(calls) == 2


def test_content_change_during_submission_never_marks(monkeypatch):
    state, args = warm_fixture(monkeypatch)
    fill_double(
        monkeypatch, on_fill=lambda: state.layers[0].warmup.content_version.add_(1)
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        warm_committed_prefix(state, 0, **args)
    assert (state.layers[0].warmup.marker == -1).all()


def test_duplicate_mapping_is_rejected_without_native_fill(monkeypatch):
    state, args = warm_fixture(monkeypatch, 256)
    calls = fill_double(monkeypatch)
    # Capacity128 cannot accept129 at all; reserve256 to exercise true alias.
    state.layers[0].guard.drained()
    args["block_table"] = torch.tensor([[0, 0]])
    args["committed_length"] = 129
    assert not warm_committed_prefix(state, 0, **args)
    assert not calls


def test_exact_extra_storage_inventory(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.decode_memory_budget import (
        shared_storage_bytes,
    )

    state, args = warm_fixture(monkeypatch)
    ws = state.layers[0]
    warm_bytes = execution.storage_inventory(ws.warmup)["cpu_bytes"]
    capacity = ws.capacity
    q = state.generations.numel()
    assert warm_bytes == 32 * capacity + 96 + 96 * q
    plain = shared_storage_bytes(q, 2, 4, 1, 4, capacity, 128, 2)
    warm = shared_storage_bytes(q, 2, 4, 1, 4, capacity, 128, 2, warmup_threshold=128)
    assert warm["device_layers"] - plain["device_layers"] == warm_bytes
    assert warm["cpu_generation_snapshot"] - plain["cpu_generation_snapshot"] == 32 * q
    # Service coordinator table is budgeted, but this component does not own it.
    assert execution.storage_inventory(state)["cpu_bytes"] == sum(warm.values()) - 32 * q
    assert state._generation_snapshot.nbytes == 8 * q
    assert not hasattr(state, "_warmup_publications")






def coordinator_fixture(monkeypatch):
    from sglang.srt.hardware_backend.npu.hisparse.runtime import NpuHiSparseCoordinator

    state, args = warm_fixture(monkeypatch)
    binding = args["binding"]
    binding.device_to_host = torch.tensor([0, 1])
    backing = SimpleNamespace(
        page_binding=binding,
        regions=[(args["region"], None)],
        layer_views=lambda layer: (args["full_kv"], args["full_kv"][:, :, :0]),
    )
    pool = SimpleNamespace(
        fsa_runtime_state=state, decode_host_backing=backing, start_layer=0, layer_num=1
    )
    coordinator = NpuHiSparseCoordinator.__new__(NpuHiSparseCoordinator)
    coordinator.mem_pool_device = pool
    coordinator.req_to_token_pool = state.pool
    coordinator.model_runner = SimpleNamespace(prefill_offload_runtime=None)
    coordinator.device = "cpu"
    state.pool.req_to_token = torch.full((5, 256), 128, dtype=torch.int64)
    req = SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=3, kv_committed_len=6), finished=lambda: False
    )
    return coordinator, state, args, req


def test_reachable_coordinator_requires_request_specific_publication(monkeypatch):
    coordinator, state, args, req = coordinator_fixture(monkeypatch)
    calls = fill_double(monkeypatch)
    # A generic ready event does not establish this request's coverage.
    assert coordinator.warmup_committed_requests([req]) == 0
    coordinator.record_warmup_publication(req, 6)
    assert coordinator.warmup_committed_requests([req]) == 1
    assert coordinator.warmup_committed_requests([req]) == 0
    assert len(calls) == 1
    # Allocated/device lengths are absent and cannot be used as committed truth.
    req.kv.kv_committed_len = 7
    assert coordinator.warmup_committed_requests([req]) == 0


def test_local_prefill_publication_preserves_exact_coverage(monkeypatch):
    coordinator, state, args, req = coordinator_fixture(monkeypatch)
    calls = fill_double(monkeypatch)
    coordinator.model_runner.prefill_offload_runtime = SimpleNamespace(
        warmup_publication=((3,), (3,), (6,), 1, args["publication_event"])
    )
    coordinator.record_local_warmup_publication([req])
    assert coordinator._warmup_publications[3, :3].tolist() == [3, 1, 6]
    assert coordinator.warmup_committed_requests([req]) == 1
    assert len(calls) == 1






@pytest.mark.parametrize("page", [2**32, 2**32 + 1, 2**31, -2])
def test_wide_host_page_rejected_before_narrowing(monkeypatch, page):
    state, args = warm_fixture(monkeypatch)
    calls = fill_double(monkeypatch)
    assert not warm_committed_prefix(
        state,
        0,
        **(args | dict(block_table=torch.tensor([[page, 1]], dtype=torch.int64)))
    )
    assert not calls


def test_coordinator_wide_host_mapping_does_not_alias_page1(monkeypatch):
    coordinator, state, args, req = coordinator_fixture(monkeypatch)
    calls = fill_double(monkeypatch)
    args["binding"].device_to_host = torch.tensor([0, 2**32 + 1], dtype=torch.int64)
    coordinator.record_warmup_publication(req, 6)
    assert coordinator.warmup_committed_requests([req]) == 0
    assert not calls


def test_failed_final_guard_fence_revokes_success_records(monkeypatch):
    state, args = warm_fixture(monkeypatch)
    calls = fill_double(monkeypatch)

    def failure():
        raise RuntimeError("injected final guard fence failure")

    monkeypatch.setattr(state.layers[0].guard, "release", failure)
    with pytest.raises(RuntimeError, match="final guard fence"):
        warm_committed_prefix(state, 0, **args)
    warm = state.layers[0].warmup
    assert (warm.marker[3] == -1).all() and (warm.stamp[3] == -1).all()
    assert not state.layers[0].guard.active


def test_failed_marker_revocation_still_drains_and_quarantines(monkeypatch):
    import sglang.srt.hardware_backend.npu.hisparse.shared_execution as shared

    state, args = warm_fixture(monkeypatch)
    fill_double(monkeypatch)
    ws = state.layers[0]
    warm = ws.warmup
    actions = []
    original_fill = torch.Tensor.fill_

    def fill(tensor, value):
        if value == -1 and tensor.ndim == 1 and tensor.numel() in (5, 6):
            raise RuntimeError("injected marker revoke launch failure")
        return original_fill(tensor, value)

    monkeypatch.setattr(torch.Tensor, "fill_", fill)
    monkeypatch.setattr(
        ws.guard,
        "release",
        lambda: (_ for _ in ()).throw(RuntimeError("final fence fault")),
    )
    monkeypatch.setattr(
        shared, "drain_shared", lambda ws, binding: actions.append("drain")
    )
    original_cancel = state.external_resident.cancel_execution

    def cancel(*a, **k):
        actions.append("cancel")
        return original_cancel(*a, **k)

    monkeypatch.setattr(state.external_resident, "cancel_execution", cancel)
    with pytest.raises(RuntimeError, match="revocation failed"):
        warm_committed_prefix(state, 0, **args)
    assert actions == ["drain", "drain"]
    assert state.external_resident.layers[0]["in_flight"].sum() == 0
    assert ws.guard.quarantine is not None and ws.guard.retained


def test_unfinished_warmup_lease_cancels_only_after_drain(monkeypatch):
    import sglang.srt.hardware_backend.npu.hisparse.shared_execution as shared
    import sglang.srt.hardware_backend.npu.fsa_native as native

    state, args = warm_fixture(monkeypatch)
    actions = []

    def fail(*args):
        raise RuntimeError("injected native submit exception")

    monkeypatch.setattr(
        native, "native_ops", lambda: SimpleNamespace(external_plan_fill_only=fail)
    )
    monkeypatch.setattr(
        shared, "drain_shared", lambda ws, binding: actions.append("drain")
    )
    original_cancel = state.external_resident.cancel_execution

    def cancel(*a, **k):
        actions.append("cancel")
        return original_cancel(*a, **k)

    monkeypatch.setattr(state.external_resident, "cancel_execution", cancel)
    with pytest.raises(RuntimeError, match="submit exception"):
        warm_committed_prefix(state, 0, **args)
    assert actions == ["drain", "cancel", "drain"]
    assert state.external_resident.layers[0]["in_flight"].sum() == 0


def test_disabled_pd_warmup_never_reads_committed_authority(monkeypatch):
    coordinator, state, args, req = coordinator_fixture(monkeypatch)

    class UnreadableRequest:
        @property
        def kv(self):
            raise AssertionError("disabled warmup read Req.kv")

    coordinator.mem_pool_device.fsa_runtime_state = SimpleNamespace(
        execution_abi="shared_v3", warmup_threshold=0
    )
    assert coordinator.warmup_publication_committed_length(UnreadableRequest()) is None
    coordinator.mem_pool_device.fsa_runtime_state = SimpleNamespace(
        execution_abi="row_v0", warmup_threshold=128
    )
    assert coordinator.warmup_publication_committed_length(UnreadableRequest()) is None
    coordinator.mem_pool_device.fsa_runtime_state = state
    assert coordinator.warmup_publication_committed_length(req) == 6
    assert coordinator.warmup_publication_committed_length(SimpleNamespace()) is None
    assert (
        coordinator.warmup_publication_committed_length(
            SimpleNamespace(kv=SimpleNamespace(kv_committed_len=-1))
        )
        is None
    )


@pytest.mark.parametrize("missing", [None, "external_plan_fill_only", "external_plan_warmup_capabilities", "wrong_abi"])
def test_independent_warmup_capability_requires_only_its_helper(monkeypatch, missing):
    import sglang.srt.hardware_backend.npu.fsa_native as native

    expected = [1, 1, 512, 0, 128, 16384, 32767, 1024, 9, 1]
    fields = dict(external_plan_fill_only=lambda *args: None, external_plan_warmup_capabilities=lambda: expected)
    if missing == "wrong_abi":
        fields["external_plan_warmup_capabilities"] = lambda: [2, *expected[1:]]
    elif missing:
        fields.pop(missing)
    monkeypatch.setattr(native, "native_ops", lambda: SimpleNamespace(**fields))
    if missing:
        with pytest.raises(RuntimeError, match="warmup|fill-only"):
            native.require_external_plan_warmup_capability()
    else:
        assert native.require_external_plan_warmup_capability() == expected
