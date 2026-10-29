"""CPU contracts for real offload bindings, Host pools and HostPoolGroup.

Only physical MemFabric storage and NPU events/streams are simulated. Invalid
restore tests require atomic rejection, even while the binding is being fixed.
This file does not qualify NPU execution, TP visibility or storage integration.
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest
import test_hicache_memfabric_pool as host_tests
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import npu_module_name, run_pytest

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture
def runtime(monkeypatch):
    yield from host_tests.runtime.__wrapped__(monkeypatch)


@pytest.fixture
def device(runtime, monkeypatch):
    stream = SimpleNamespace(waits=[], records=[])
    stream.wait_event = stream.waits.append
    events = []

    class Event:
        def __init__(self):
            self.complete = False
            self.dependencies = ()
            events.append(self)

        def record(self, current_stream=None):
            current_stream = stream if current_stream is None else current_stream
            self.dependencies = tuple(current_stream.waits)
            current_stream.records.append(self)

        def query(self):
            return self.complete and all(event.query() for event in self.dependencies)

        def synchronize(self):
            for event in self.dependencies:
                event.synchronize()
            self.complete = True

    def synchronize():
        for event in events:
            event.complete = True

    api = SimpleNamespace(
        Event=Event,
        current_stream=lambda *args: stream,
        synchronize=synchronize,
        stream=stream,
        events=events,
    )
    monkeypatch.setattr(torch, "npu", api, raising=False)
    return api


@pytest.fixture
def bound(runtime, device):
    backing, device_pool = runtime.make()
    device_pool.size = 32
    pool = backing.bind_hicache(device_pool)
    return SimpleNamespace(pool=pool, binding=pool.binding, device=device)


def rows(*pages, page_size=4):
    return (
        torch.tensor(pages, dtype=torch.int64)[:, None] * page_size
        + torch.arange(page_size)
    ).reshape(-1)


def test_pd_transfer_pin_keeps_host_page_live_after_request_release(bound):
    binding = bound.binding
    assert binding.ensure_pages(torch.tensor([1]))
    host_pages = binding.pin_transfer_pages([1])
    binding.release_device_pages(torch.tensor([1]))
    assert binding.references[host_pages].tolist() == [1]
    assert not binding.cache_references.any()
    binding.unpin_transfer_pages(host_pages)
    assert binding.references[host_pages].tolist() == [0]
    assert len(binding.retired) == 1


def test_pd_transfer_cannot_pin_unmapped_pages(bound):
    before = bound.binding.references.clone()
    with pytest.raises(RuntimeError, match="allocated Host pages"):
        bound.binding.pin_transfer_pages([1])
    assert torch.equal(before, bound.binding.references)


def test_pd_publication_runs_once_for_packed_target_and_draft(bound):
    from sglang.srt.hardware_backend.npu.hisparse.hicache_offload import (
        publish_received_host_kv,
    )

    calls = []
    shared = SimpleNamespace(publish_write=lambda: calls.append("publish"))
    backing = SimpleNamespace(
        page_binding=bound.binding, pool=SimpleNamespace(shared_group=shared)
    )
    publish_received_host_kv(SimpleNamespace(host_backings=(backing, backing)))
    assert calls == []
    assert len(bound.device.events) == 1
    assert bound.binding.ready_event is bound.device.events[-1]


@pytest.mark.parametrize("waiting", ["transfer", "restore"])
@pytest.mark.parametrize("warmup", ["no-coordinator", "disabled", "enabled", "unknown"])
def test_pd_publication_waits_for_tp_receive_and_restore_readiness(
    bound, waiting, warmup
):
    from test_hisparse_pd_dispatch import production_definitions

    polls = production_definitions(
        host_tests.SRT / "disaggregation/base/conn.py",
        {"KVPoll": None},
    ).KVPoll
    restore = production_definitions(
        host_tests.SRT / "disaggregation/decode_hicache_mixin.py",
        {"HiCacheRestoreResult": None, "HiCacheRestoreGatedKVReceiver": None},
        bases={"HiCacheRestoreResult": Enum},
        KVPoll=polls,
    )
    peer = SimpleNamespace(
        poll=polls.Transferring if waiting == "transfer" else polls.Success,
    )
    reductions = []

    def all_reduce(tensor, op, group):
        reductions.append(tensor.tolist())
        tensor.clamp_(max=peer.poll)

    polling = production_definitions(
        host_tests.SRT / "disaggregation/utils.py",
        {},
        functions={
            "poll_and_all_reduce",
            "_poll_with_failure_injection",
            "_apply_metadata_gate",
            "_all_reduce_polls",
        },
        KVPoll=polls,
        torch=torch,
        dist=SimpleNamespace(
            get_world_size=lambda group: 2,
            all_reduce=all_reduce,
            ReduceOp=SimpleNamespace(MIN="min"),
        ),
        envs=SimpleNamespace(
            SGLANG_TEST_DISAGG_FAILURE_PROB=SimpleNamespace(get=lambda: 0)
        ),
        _is_fake_transfer=lambda req: False,
    )
    module = production_definitions(
        host_tests.SRT / "disaggregation/decode.py",
        {
            "DecodeTransferQueue": {
                "pop_transferred",
                "_publish_pending_layer_metadata",
                "_pollers",
                "_poll_with_metadata_gate",
            }
        },
        KVPoll=polls,
        HiCacheRestoreResult=restore.HiCacheRestoreResult,
        HiCacheRestoreGatedKVReceiver=restore.HiCacheRestoreGatedKVReceiver,
        poll_and_all_reduce=polling.poll_and_all_reduce,
        FINISH_ABORT=type("FinishAbort", (), {}),
        torch=torch,
    )
    queue = module.DecodeTransferQueue()
    calls = []
    shared = SimpleNamespace(publish_write=lambda: calls.append("broadcast"))
    backing = SimpleNamespace(
        page_binding=bound.binding, pool=SimpleNamespace(shared_group=shared)
    )
    request = SimpleNamespace(rid="request", finished_reason=None, bootstrap_room=5)
    entry = SimpleNamespace(
        req=request,
        kv_receiver=SimpleNamespace(
            poll=lambda: polls.Success,
            kv_mgr=SimpleNamespace(
                kv_args=SimpleNamespace(
                    npu_layer_receive_pool=SimpleNamespace(host_backings=(backing,))
                )
            ),
        ),
        hicache_restore_status=(
            restore.HiCacheRestoreResult.PENDING
            if waiting == "restore"
            else restore.HiCacheRestoreResult.READY
        ),
        host_staged=False,
        npu_layer_metadata=None,
        metadata_buffer_index=0,
    )
    queue.queue = [entry]
    queue.scheduler = SimpleNamespace(enable_decode_hicache=waiting == "restore")
    if warmup != "no-coordinator":
        from sglang.srt.hardware_backend.npu.hisparse.runtime import (
            NpuHiSparseCoordinator,
        )

        coordinator = NpuHiSparseCoordinator.__new__(NpuHiSparseCoordinator)
        coordinator.mem_pool_device = SimpleNamespace(
            fsa_runtime_state=SimpleNamespace(
                execution_abi="shared_v3",
                warmup_threshold=0 if warmup == "disabled" else 128,
            )
        )
        coordinator.record_warmup_publication = lambda req, committed: calls.append(
            ("warmup", committed)
        )
        queue.scheduler.hisparse_coordinator = coordinator
        if warmup == "enabled":
            request.kv = SimpleNamespace(kv_committed_len=6)
            request.finished = lambda: False
    queue._process_hicache_local_restores = lambda entries: None
    queue.enable_staging = False
    queue.enable_host_receive = False
    queue.enable_deferred_kv_release = False
    queue.gloo_group = object()

    def commit(entry):
        calls.append("commit")
        if warmup == "enabled":
            entry.req.kv.kv_committed_len = 7

    queue._commit_transfer_to_req = commit
    queue.metadata_buffers = SimpleNamespace(bootstrap_room=torch.tensor([[5]]))
    queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
        free=lambda index: calls.append("free metadata")
    )
    assert queue.pop_transferred() == []
    assert reductions == [
        [polls.Success if waiting == "transfer" else polls.Transferring]
    ]
    assert queue.queue == [entry]
    assert not bound.device.events
    assert not calls
    peer.poll = polls.Success
    entry.hicache_restore_status = restore.HiCacheRestoreResult.READY
    assert queue.pop_transferred() == [request]
    assert calls == (
        ["commit", ("warmup", 6), "free metadata"]
        if warmup == "enabled"
        else ["commit", "free metadata"]
    )
    assert reductions[-1] == [polls.Success]
    assert queue.queue == []
    assert queue.metadata_buffers.bootstrap_room[0, 0].item() == 0
    assert len(bound.device.events) == 1
    assert bound.binding.ready_event is bound.device.events[-1]


def snapshot(pool):
    binding = pool.binding
    return (
        binding.references.clone(),
        binding.device_to_host_cpu.clone(),
        binding.device_to_host.clone(),
        pool.slot_used.clone(),
        pool.free_slots.clone(),
        pool.available_size(),
        [(event, pages.clone()) for event, pages in binding.retired],
    )


def assert_unchanged(pool, before):
    after = snapshot(pool)
    for current, previous in zip(after[:5], before[:5]):
        assert torch.equal(current, previous)
    assert after[5] == before[5]
    assert len(after[6]) == len(before[6])
    for (event, pages), (old_event, old_pages) in zip(after[6], before[6]):
        assert event is old_event
        assert torch.equal(pages, old_pages)


def retain_prefix(bound):
    assert bound.binding.ensure_pages(torch.tensor([1]))
    cached = bound.pool.acquire_device_pages(rows(1))
    bound.binding.release_device_pages(torch.tensor([1]))
    assert bound.binding.references[1] == 1
    return cached


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_host_table_out_preserves_address_and_clears_padding(bound, dtype):
    binding = bound.binding
    assert binding.ensure_pages(torch.tensor([1, 2]))
    source = torch.tensor([[1, -1, 2], [0, 999, 1]], dtype=dtype)
    storage = torch.full((4, 10), 777, dtype=dtype)
    output = storage[:, ::2]
    pointer = output.data_ptr()
    binding.fill_host_table(source, output)
    expected = torch.zeros_like(output)
    expected[:2, :3] = binding.host_table(source)
    torch.testing.assert_close(output, expected)
    assert output.data_ptr() == pointer
    assert torch.all(storage[:, 1::2] == 777)
    binding._update(torch.tensor([1]), torch.tensor([2]))
    binding.fill_host_table(source[:1, :1], output)
    expected.zero_()
    expected[0, 0] = 2
    torch.testing.assert_close(output, expected)


def test_host_table_out_rejects_truncation_without_writing(bound):
    output = torch.full((1, 1), 123, dtype=torch.int32)
    with pytest.raises(ValueError, match="cover the input"):
        bound.binding.fill_host_table(torch.ones(2, 2, dtype=torch.int32), output)
    assert output.item() == 123


def test_draft_host_table_scope_shares_capture_but_refreshes_replay(bound, monkeypatch):
    module = sys.modules[type(bound.binding).__module__]
    scope = module.DraftHostTableScope(3)
    binding = bound.binding
    assert binding.ensure_pages(torch.tensor([1, 2]))
    source = torch.tensor([[1, 2, 0]], dtype=torch.int32)
    calls = []
    fill = binding.fill_host_table

    def record(source, output):
        calls.append(output.data_ptr())
        return fill(source, output)

    monkeypatch.setattr(binding, "fill_host_table", record)
    tables = [scope.host_table(binding, source) for _ in range(4)]
    assert len(calls) == 1
    assert all(table is tables[0] for table in tables)
    binding._update(torch.tensor([1]), torch.tensor([2]))
    replay = module.DraftHostTableScope(3)
    for table in tables:
        replay.host_table(binding, source, table)
    assert len(calls) == 2
    torch.testing.assert_close(tables[0], binding.host_table(source))
    independent_capture = torch.empty_like(source)
    replay.host_table(binding, source, independent_capture)
    assert len(calls) == 3
    torch.testing.assert_close(independent_capture, tables[0])


def test_draft_host_mapping_scope_does_not_reuse_after_remap(bound):
    module = sys.modules[type(bound.binding).__module__]
    binding = bound.binding
    assert binding.ensure_pages(torch.tensor([1]))
    source = torch.tensor([[1]], dtype=torch.int32)
    scope = module.DraftHostTableScope(3)
    output = scope.host_table(binding, source)
    previous = output.item()
    binding._update(torch.tensor([1]), torch.tensor([previous + 1]))
    assert scope.host_table(binding, source, output) is output
    assert output.item() == previous + 1


@pytest.mark.parametrize("cache_first", [True, False])
def test_device_and_cache_references_retire_only_after_both_release(bound, cache_first):
    pool, binding = bound.pool, bound.binding
    assert binding.ensure_pages(torch.tensor([1]))
    cached = pool.acquire_device_pages(rows(1))
    assert binding.references[1] == 2
    if cache_first:
        assert pool.free(cached) == 4
    else:
        binding.release_device_pages(torch.tensor([1]))
    assert binding.references[1] == 1
    assert not binding.retired
    assert pool.slot_used[cached].all()
    if cache_first:
        binding.release_device_pages(torch.tensor([1]))
    else:
        assert pool.free(cached) == 4
    assert binding.references[1] == 0
    assert len(binding.retired) == 1
    assert binding.retired[0][1].tolist() == [1]
    binding.reap()
    assert pool.available_size() == pool.usable_size - 4
    bound.device.events[-1].complete = True
    binding.reap()
    assert pool.available_size() == pool.usable_size
    assert not pool.slot_used[cached].any()
    assert pool.slot_used[:4].all()
    binding.reap()
    assert pool.available_size() == pool.usable_size


@pytest.mark.parametrize("release_together", [True, False])
def test_repeated_host_mapping_counts_each_device_page_once(bound, release_together):
    pool, binding = bound.pool, bound.binding
    cached = retain_prefix(bound)
    binding.restore_mapping(torch.cat((cached, cached)), rows(2, 3))
    assert binding.references[1] == 3
    binding.restore_mapping(torch.cat((cached, cached)), rows(2, 3))
    assert binding.references[1] == 3
    assert binding.host_rows(rows(2, 3)).tolist() == cached.repeat(2).tolist()
    pool.free(cached)
    assert binding.references[1] == 2
    if release_together:
        binding.release_device_pages(torch.tensor([2, 3, 2]))
    else:
        binding.release_device_pages(torch.tensor([2]))
        assert binding.references[1] == 1
        assert not binding.retired
        binding.release_device_pages(torch.tensor([3]))
    assert binding.references[1] == 0
    assert len(binding.retired) == 1
    assert binding.retired[0][1].tolist() == [1]
    binding.reap()
    assert pool.slot_used[cached].all()
    bound.device.events[-1].complete = True
    binding.reap()
    assert pool.available_size() == pool.usable_size
    assert not binding.device_to_host_cpu.any()


def test_cached_aliases_retain_one_reference_per_page_occurrence(bound):
    pool, binding = bound.pool, bound.binding
    cached = retain_prefix(bound)
    binding.restore_mapping(cached.repeat(2), rows(2, 3))
    alias_rows = pool.acquire_device_pages(rows(2, 3))
    assert binding.references[1] == 5
    pool.free(alias_rows)
    assert binding.references[1] == 3
    pool.free(cached)
    binding.release_device_pages(torch.tensor([2, 3]))
    bound.device.events[-1].complete = True
    binding.reap()
    assert pool.available_size() == pool.usable_size


def test_retirement_waits_for_backup_ready_event_before_reuse(bound):
    pool, binding, device = bound.pool, bound.binding, bound.device
    assert binding.ensure_pages(torch.tensor([1]))
    ready = device.Event()
    ready.record(device.stream)
    binding.ready_event = ready
    binding.release_device_pages(torch.tensor([1]))
    retired_event = binding.retired[0][0]
    assert device.stream.waits == [ready]
    retired_event.complete = True
    binding.reap()
    assert pool.available_size() == pool.usable_size - 4
    ready.complete = True
    binding.reap()
    assert pool.available_size() == pool.usable_size


def test_capacity_failure_is_atomic_and_pressure_waits_for_retired_readers(bound):
    pool, binding = bound.pool, bound.binding
    before = snapshot(pool)
    assert not binding.ensure_pages(torch.tensor([1, 2, 3, 4, 5]))
    assert_unchanged(pool, before)
    assert binding.ensure_pages(torch.tensor([1, 2, 3, 4]))
    assert pool.available_size() == 0
    before = snapshot(pool)
    assert binding.ensure_pages(torch.tensor([1, 1]))
    assert_unchanged(pool, before)
    assert not binding.ensure_pages(torch.tensor([1, 5]))
    assert_unchanged(pool, before)
    binding.release_device_pages(torch.tensor([2]))
    retired_event = bound.device.events[-1]
    assert not retired_event.query()
    assert binding.ensure_pages(torch.tensor([5]))
    assert retired_event.query()
    assert binding.device_to_host_cpu[5] == 2
    assert pool.available_size() == 0


def test_eviction_callback_cannot_make_active_device_references_available(bound):
    pool, binding = bound.pool, bound.binding
    assert binding.ensure_pages(torch.tensor([1, 2, 3, 4]))
    cached = pool.acquire_device_pages(rows(1))
    requests = []

    def evict(needed):
        requests.append(needed)
        pool.free(cached)

    binding.evict_host = evict
    assert not binding.ensure_pages(torch.tensor([5]))
    assert requests == [pool.page_size]
    assert binding.references[1] == 1
    assert pool.slot_used[cached].all()
    assert not binding.retired


@pytest.mark.parametrize("pages", [[0], [-1], [9], [1, 9]])
def test_invalid_device_admission_leaves_existing_ownership_unchanged(bound, pages):
    assert bound.binding.ensure_pages(torch.tensor([1]))
    before = snapshot(bound.pool)
    with pytest.raises(ValueError):
        bound.binding.ensure_pages(torch.tensor(pages))
    assert_unchanged(bound.pool, before)


@pytest.mark.parametrize(
    "host_values,device_values",
    [
        ([4, 5, 6, 7], [12, 13, 14]),
        ([4, 5, 6], [12, 13, 14]),
        ([4, 5, 6, 7], [13, 14, 15, 16]),
        ([0, 1, 2, 3], [12, 13, 14, 15]),
        ([-4, -3, -2, -1], [12, 13, 14, 15]),
        ([20, 21, 22, 23], [12, 13, 14, 15]),
        ([12, 13, 14, 15], [12, 13, 14, 15]),
        ([4, 5, 6, 7], [0, 1, 2, 3]),
        ([4, 5, 6, 7], [36, 37, 38, 39]),
        ([4, 5, 6, 11], [12, 13, 14, 19]),
        ([5, 6, 7, 8], [13, 14, 15, 16]),
        ([4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 12, 13, 14, 15]),
    ],
)
def test_failed_restore_rejects_invalid_pages_atomically(
    bound, host_values, device_values
):
    binding = bound.binding
    assert binding.ensure_pages(torch.tensor([1, 2, 3]))
    bound.pool.acquire_device_pages(rows(1, 2))
    binding.release_device_pages(torch.tensor([1, 2, 3]))
    bound.device.events[-1].complete = True
    binding.reap()
    binding.restore_mapping(rows(1), rows(3))
    before = snapshot(bound.pool)
    with pytest.raises((ValueError, RuntimeError)):
        binding.restore_mapping(torch.tensor(host_values), torch.tensor(device_values))
    assert_unchanged(bound.pool, before)


def test_retain_unmapped_device_pages_fails_without_incrementing_live_refs(bound):
    assert bound.binding.ensure_pages(torch.tensor([1]))
    before = snapshot(bound.pool)
    with pytest.raises(RuntimeError):
        bound.pool.acquire_device_pages(rows(1, 2))
    assert_unchanged(bound.pool, before)


@pytest.fixture
def host_group(runtime, monkeypatch):
    environ = ModuleType("sglang.srt.environ")
    environ.envs = SimpleNamespace()
    monkeypatch.setitem(sys.modules, npu_module_name(environ.__name__), environ)
    memory_pool = sys.modules["sglang.srt.mem_cache.memory_pool"]
    monkeypatch.setattr(
        memory_pool,
        "MHATokenToKVPool",
        type("MHATokenToKVPool", (), {}),
        raising=False,
    )
    storage = host_tests.load_module(
        monkeypatch,
        "sglang.srt.mem_cache.hicache_storage",
        host_tests.SRT / "mem_cache/hicache_storage.py",
    )
    group = host_tests.load_module(
        monkeypatch,
        "sglang.srt.mem_cache.pool_host.group",
        host_tests.SRT / "mem_cache/pool_host/group.py",
    )
    return SimpleNamespace(storage=storage, group=group)


@pytest.mark.parametrize("device_value", ["cpu", torch.device("cpu")])
def test_host_pool_group_contract_with_string_or_torch_device(
    runtime, device, host_group, device_value
):
    backing, device_pool = runtime.make()
    device_pool.device = device_value
    pool = backing.bind_hicache(device_pool)
    entry = host_group.group.PoolEntry(
        name=host_group.storage.PoolName.KV,
        host_pool=pool,
        device_pool=device_pool,
        layer_mapper=lambda layer: layer,
        is_primary_index_anchor=True,
    )
    group = host_group.group.HostPoolGroup([entry])
    assert group.layout == "layer_first"
    assert group.page_size == pool.page_size
    assert torch.device(group.device) == torch.device("cpu")
    assert pool.binding.device == torch.device("cpu")
    assert group.size == group.logical_size == pool.size
    assert group.size_per_token == pool.get_size_per_token()
    assert not group.can_use_write_back_jit
    assert not group.supports_per_pool_backup_indices
    assert group.get_pool(host_group.storage.PoolName.KV) is pool
    assert pool.binding.ensure_pages(torch.tensor([1]))
    cached = group.anchor_entry.host_pool.acquire_device_pages(rows(1))
    ready = device.Event()
    pool.binding.ready_event = ready
    pool.backup_from_device_all_layer(device_pool, cached, rows(1), "kernel")
    assert device.stream.waits == [ready]
    assert group.free(cached) == 4
    assert group.available_size() == pool.usable_size - 4
    group.clear()
    assert group.available_size() == pool.usable_size
    assert pool.slot_used[:4].all()
    assert not pool.binding.references.any()
    assert not pool.binding.device_to_host_cpu.any()
    before = list(runtime.events)
    group.destroy()
    assert runtime.events == before
    assert not backing.pool._closed


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_page_io_accepts_scalar_tensor_indices_from_controller(runtime, dtype):
    backing, device_pool = runtime.make()
    pool = runtime.pool_type(backing, device_pool)
    index = pool.alloc(pool.page_size)[0].to(dtype)
    page = pool.get_dummy_flat_data_page().fill_(13)
    pool.set_from_flat_data_page(index, page)
    assert torch.equal(pool.get_data_page(index), page)
    assert torch.equal(pool.get_data_page(index.item()), page)


@pytest.mark.parametrize(
    "index",
    [
        torch.tensor(True),
        torch.tensor(4.0),
        torch.tensor(4 + 0j),
        torch.tensor([4]),
        torch.tensor([[4]]),
        torch.tensor(0),
        torch.tensor(-4),
        torch.tensor(5),
        torch.tensor(20),
        torch.empty((), dtype=torch.int64, device="meta"),
    ],
)
def test_page_io_rejects_invalid_tensor_indices_without_mutation(runtime, index):
    backing, device_pool = runtime.make()
    pool = runtime.pool_type(backing, device_pool)
    slots = pool.alloc(pool.page_size)
    page = pool.get_dummy_flat_data_page().fill_(3)
    pool.set_from_flat_data_page(slots[0], page)
    before = pool.get_data_page(slots[0])
    with pytest.raises(ValueError):
        pool.get_data_page(index)
    with pytest.raises(ValueError):
        pool.set_from_flat_data_page(index, page + 1)
    assert torch.equal(pool.get_data_page(slots[0]), before)


def test_live_mapping_permutation_must_not_retire_still_referenced_pages(bound):
    pool, binding = bound.pool, bound.binding
    assert binding.ensure_pages(torch.tensor([1, 2]))
    binding.restore_mapping(rows(2, 1), rows(1, 2))
    assert binding.references[1:3].tolist() == [1, 1]
    assert binding.device_to_host_cpu[1:3].tolist() == [2, 1]
    assert not binding.retired
    bound.device.synchronize()
    binding.reap()
    assert pool.slot_used[rows(1, 2)].all()
    assert pool.available_size() == pool.usable_size - 8


def test_restore_rejects_retired_page_even_while_physical_slots_are_used(bound):
    pool, binding = bound.pool, bound.binding
    assert binding.ensure_pages(torch.tensor([1]))
    binding.release_device_pages(torch.tensor([1]))
    assert pool.slot_used[rows(1)].all()
    assert binding.references[1] == 0
    before = snapshot(pool)
    with pytest.raises((ValueError, RuntimeError)):
        binding.restore_mapping(rows(1), rows(2))
    assert_unchanged(pool, before)


def test_repeated_cache_release_cannot_consume_device_reference(bound):
    pool, binding = bound.pool, bound.binding
    assert binding.ensure_pages(torch.tensor([1]))
    cached = pool.acquire_device_pages(rows(1))
    pool.free(cached)
    assert binding.references[1] == 1
    before = snapshot(pool)
    with pytest.raises((ValueError, RuntimeError)):
        pool.free(cached)
    assert_unchanged(pool, before)


def test_split_cached_aliases_preserve_both_live_device_references(bound):
    pool, binding = bound.pool, bound.binding
    cached = retain_prefix(bound)
    binding.restore_mapping(cached.repeat(2), rows(2, 3))
    split_cache = pool.acquire_device_pages(rows(2, 3))
    pool.free(cached)
    pool.free(split_cache[:4])
    pool.free(split_cache[4:])
    assert binding.references[1] == 2
    assert not binding.retired
    binding.release_device_pages(torch.tensor([2, 3]))
    bound.device.synchronize()
    binding.reap()
    assert pool.available_size() == pool.usable_size


def test_retirement_includes_recorded_reader_completion(bound):
    pool, binding, device = bound.pool, bound.binding, bound.device
    assert binding.ensure_pages(torch.tensor([1]))
    reader = device.Event()
    reader.record(device.stream)
    binding.record_reader(reader)
    binding.release_device_pages(torch.tensor([1]))
    retirement = binding.retired[0][0]
    assert reader in retirement.dependencies
    retirement.complete = True
    binding.reap()
    assert pool.slot_used[rows(1)].all()
    reader.complete = True
    binding.reap()
    assert pool.available_size() == pool.usable_size


def make_tp_pools(runtime, barrier):
    pools = []
    for rank in range(2):
        backing, device_pool = runtime.make()
        device_pool.size = 32
        pool = backing.bind_hicache(device_pool)
        pool.binding.shared_group = SimpleNamespace(
            group=SimpleNamespace(
                world_size=2,
                rank_in_group=rank,
                barrier=lambda rank=rank: barrier(rank),
            )
        )
        pools.append(pool)
    return pools


def test_tp_local_completion_does_not_change_capacity_without_pressure(runtime, device):
    def unexpected_barrier(rank):
        raise AssertionError("No pressure: a CPU barrier must not be submitted")

    pools = make_tp_pools(runtime, unexpected_barrier)
    for pool in pools:
        assert pool.binding.ensure_pages(torch.tensor([1, 2]))
        pool.binding.release_device_pages(torch.tensor([1]))
    pools[0].binding.retired[0][0].complete = True
    assert not pools[1].binding.retired[0][0].query()
    for pool in pools:
        pool.binding.reap()
        assert pool.available_size() == 8
        assert pool.slot_used[rows(1)].all()
        assert pool.binding.ensure_pages(torch.tensor([3]))
        assert pool.binding.device_to_host_cpu[3] == 3
        assert len(pool.binding.retired) == 1
    assert torch.equal(pools[0].free_slots, pools[1].free_slots)
    assert torch.equal(pools[0].binding.references, pools[1].binding.references)


def test_tp_pressure_waits_for_slowest_rank_and_reuses_identical_fifo_pages(
    runtime, device, monkeypatch
):
    rendezvous = threading.Barrier(2, timeout=5)
    owner_at_barrier = threading.Event()
    peer_waiting = threading.Event()
    release_peer = threading.Event()
    barrier_calls = [0, 0]
    pools = []

    def barrier(rank):
        barrier_calls[rank] += 1
        assert all(event.query() for event, _ in pools[rank].binding.retired)
        if rank == 0:
            owner_at_barrier.set()
        rendezvous.wait()

    pools.extend(make_tp_pools(runtime, barrier))
    for pool in pools:
        assert pool.binding.ensure_pages(torch.tensor([1, 2, 3, 4]))
        pool.binding.release_device_pages(torch.tensor([2]))
        pool.binding.release_device_pages(torch.tensor([4]))
        for event, _ in pool.binding.retired:
            event.complete = True
    peer_event = pools[1].binding.retired[-1][0]
    peer_event.complete = False

    def await_peer_reader():
        peer_waiting.set()
        if not release_peer.wait(timeout=5):
            raise AssertionError("Timed out waiting for simulated peer reader")
        peer_event.complete = True

    monkeypatch.setattr(peer_event, "synchronize", await_peer_reader)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(pool.binding.ensure_pages, torch.tensor([5, 6]))
            for pool in pools
        ]
        try:
            assert owner_at_barrier.wait(timeout=5)
            assert peer_waiting.wait(timeout=5)
            assert not futures[0].done()
            for pool in pools:
                assert pool.slot_used[rows(2, 4)].all()
                assert pool.available_size() == 0
        finally:
            release_peer.set()
        assert all(future.result(timeout=5) for future in futures)
    assert barrier_calls == [1, 1]
    for pool in pools:
        assert pool.binding.device_to_host_cpu[[5, 6]].tolist() == [2, 4]
        assert not pool.binding.retired
        assert pool.available_size() == 0
        assert pool.slot_used[:4].all()
    assert torch.equal(pools[0].slot_used, pools[1].slot_used)
    assert torch.equal(pools[0].binding.references, pools[1].binding.references)
    assert torch.equal(
        pools[0].binding.device_to_host_cpu, pools[1].binding.device_to_host_cpu
    )


def test_prefill_plans_use_independent_host_pages_and_deduplicate_shared_prefix(
    bound, monkeypatch
):
    host_tests.load_module(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.prefill_dma",
        host_tests.SRT / "hardware_backend/npu/prefill_dma.py",
    )
    assert bound.binding.ensure_pages(torch.tensor([2, 5, 7]))
    prefix = torch.cat((torch.tensor([0]), rows(5, 2, 5)))
    restore, save = bound.pool.backing.plan_prefill_rows(prefix, rows(7))
    assert restore.tolist() == [[4, 8, 4], [8, 20, 4]]
    assert save.tolist() == [[28, 12, 4]]
    for invalid in (-1, 36):
        with pytest.raises(ValueError, match="outside the device"):
            bound.pool.backing.plan_prefill_rows(torch.tensor([invalid]), rows(7))
    with pytest.raises(RuntimeError, match="unmapped"):
        bound.pool.backing.plan_prefill_rows(rows(4), rows(7))


def test_prefill_plans_reject_duplicate_writes_and_accept_empty_rows(
    bound, monkeypatch
):
    host_tests.load_module(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.prefill_dma",
        host_tests.SRT / "hardware_backend/npu/prefill_dma.py",
    )
    assert bound.binding.ensure_pages(torch.tensor([1]))
    empty = torch.empty(0, dtype=torch.int64)
    with pytest.raises(ValueError, match="duplicate destination"):
        bound.pool.backing.plan_prefill_rows(empty, rows(1, 1))
    restore, save = bound.pool.backing.plan_prefill_rows(empty, empty)
    assert restore.shape == save.shape == (0, 3)


def test_l2_host_pin_separates_transfer_request_and_cache_ownership(bound):
    binding = bound.binding
    assert binding.ensure_pages(torch.tensor([1]))
    cached = bound.pool.acquire_device_pages(rows(1))
    pinned = binding.pin_host_rows(cached)
    binding.release_device_pages(torch.tensor([1]))
    bound.pool.free(cached)
    assert binding.references[pinned].tolist() == [1]
    event = bound.device.Event()
    event.record()
    binding.record_reader(event)
    binding.unpin_transfer_pages(pinned)
    binding.reap()
    assert bound.pool.available_size() == bound.pool.usable_size - 4
    bound.device.synchronize()
    binding.reap()
    assert bound.pool.available_size() == bound.pool.usable_size


def test_metadata_only_backup_rejects_new_empty_host_pages(bound):
    binding = bound.binding
    assert binding.ensure_pages(torch.tensor([1]))
    cached = bound.pool.acquire_device_pages(rows(1))
    empty = bound.pool.alloc(4)
    with pytest.raises(RuntimeError, match="adopt existing Host pages"):
        bound.pool.backup_from_device_all_layer(
            bound.pool.device_pool, empty, rows(1), "kernel_ascend"
        )
    assert cached.tolist() != empty.tolist()


if __name__ == "__main__":
    run_pytest(__file__)
