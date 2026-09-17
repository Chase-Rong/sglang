"""Collective attachment, writer ownership and shared-page release contracts."""

import ast
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import test_registered_host_kv_adapter as pool_tests
import torch

SOURCE = (
    Path(__file__).parents[4]
    / "python/sglang/srt/hardware_backend/npu/shared_host_kv.py"
)
SPEC = importlib.util.spec_from_file_location("shared_host_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Group:
    world_size = 2

    def __init__(self, rank=0, *, payload=None, replies=()):
        self.rank_in_group = rank
        self.payload = payload
        self.replies = iter(replies)
        self.events = []

    def all_gather_object(self, value):
        self.events.append(("gather", value))
        return next(self.replies, [value, value])

    def broadcast_object(self, value, src=0):
        self.events.append(("broadcast_object", value))
        return self.payload if self.rank_in_group != src else value

    def broadcast(self, value, src=0):
        self.events.append(("publish", src))


class Region:
    allocated = []
    fail_attach = False

    def __init__(self):
        self.tensor = torch.zeros((4, 32))
        self.closed = False

    @classmethod
    def allocate(cls, *args, **kwargs):
        region = cls()
        cls.allocated.append(region)
        return region

    @classmethod
    def attach(cls, *args, **kwargs):
        if cls.fail_attach:
            raise RuntimeError("HAL attach failed")
        return cls.allocate()

    def export_handle(self):
        return "handle"

    def close(self, *, drained=False):
        assert drained
        self.closed = True


@pytest.fixture(autouse=True)
def reset_regions():
    Region.allocated = []
    Region.fail_attach = False


def shared_group(group):
    shared = MODULE.SharedHostGroup.__new__(MODULE.SharedHostGroup)
    shared.group = group
    shared.device_id = group.rank_in_group
    shared.writer_rank = 0
    shared.sequence = 0
    shared.failed = False
    shared.fence = torch.zeros(1, dtype=torch.int32)
    return shared


def test_owner_retains_region_until_all_peers_acknowledge_attach():
    group = Group()
    shared = shared_group(group)
    region = shared.allocate(Region, "latent", (4, 32), torch.float32, 2)
    assert not region.closed
    assert [event[0] for event in group.events] == [
        "gather",
        "broadcast_object",
        "gather",
    ]
    assert group.events[1][1] == ("handle", None)


def test_failed_peer_attach_releases_owner_and_latches_group_failure():
    geometry = (0, "latent", (4, 32), "torch.float32", 2)
    shared = shared_group(
        Group(replies=([geometry, geometry], [None, "HAL attach failed"]))
    )
    with pytest.raises(RuntimeError, match="attach failed"):
        shared.allocate(Region, "latent", (4, 32), torch.float32, 2)
    assert Region.allocated[0].closed and shared.failed
    with pytest.raises(RuntimeError, match="failed"):
        shared.publish_write()


def test_consumer_reports_attach_failure_in_same_collective_sequence():
    Region.fail_attach = True
    group = Group(1, payload=("handle", None))
    shared = shared_group(group)
    with pytest.raises(RuntimeError, match="HAL attach failed"):
        shared.allocate(Region, "latent", (4, 32), torch.float32, 2)
    assert group.events[-1] == ("gather", "HAL attach failed")
    assert not Region.allocated


def test_page_identity_mismatch_prevents_shared_writes():
    shared = shared_group(Group(replies=([(5, (1, 2)), (5, (1, 3))],)))
    with pytest.raises(ValueError, match="page identity"):
        shared.validate_identity((5, (1, 2)))
    assert shared.failed


@pytest.mark.parametrize("rank", [0, 1])
def test_only_owner_copies_but_every_consumer_publishes_on_its_stream(rank):
    events = []
    shared = shared_group(Group(rank))
    shared.publish_write = lambda: events.append("publish")
    pool = pool_tests.NpuRegisteredHostKvPool(rank, shared_group=shared)
    region = Region()
    pool._components["latent"] = SimpleNamespace(region=region)
    pool._kernel_module = SimpleNamespace(
        copy_rows_to_registered_host=lambda *args: events.append("copy")
    )
    result = pool.copy_to_host("latent", None, None, None, None)
    assert result is region.tensor
    assert events == (["copy", "publish"] if rank == 0 else ["publish"])


def test_shared_abort_does_not_recycle_pages_until_all_ranks_drain():
    group = Group(replies=([{5: True, 6: True}, {5: False, 6: True}],))
    assert MODULE.shared_abort_drain_readiness(group, {5: True, 6: True}) == {
        5: False,
        6: True,
    }
    missing = Group(replies=([{5: True}, {}],))
    assert MODULE.shared_abort_drain_readiness(missing, {5: True}) == {5: False}


def test_foreign_node_is_rejected_before_npu_allocation(monkeypatch):
    monkeypatch.setattr(
        MODULE, "local_shared_identity", lambda rank: ("host-a", 1, 2, rank)
    )
    group = Group(replies=([("host-a", 1, 2, 0), ("host-b", 1, 2, 0)],))
    with pytest.raises(ValueError, match="host and IPC"):
        MODULE.SharedHostGroup(group, 0)


@pytest.mark.parametrize("shared", [True, False, None])
def test_decode_release_waits_for_shared_writer_even_after_deadline(
    monkeypatch, shared
):
    source = SOURCE.parents[2] / "disaggregation/decode.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "resolve_deferred_releases"
    )
    group = Group(replies=([{5: True}, {5: False}], [{5: True}, {5: True}]))
    parallel = ModuleType("sglang.srt.distributed.parallel_state")
    parallel.get_attn_tp_group = lambda: group
    monkeypatch.setitem(sys.modules, parallel.__name__, parallel)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.hardware_backend.npu.shared_host_kv", MODULE
    )
    namespace = dict(
        time=time,
        envs=SimpleNamespace(
            SGLANG_NPU_SHARED_HOST_KV=SimpleNamespace(get=lambda: True)
        ),
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    manager = SimpleNamespace(
        is_abort_release_safe=lambda *args: True,
        requires_strict_abort_drain=lambda room: True,
    )
    request = SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=5),
        kv_receiver=SimpleNamespace(kv_mgr=manager),
    )
    released = []
    queue = SimpleNamespace(
        scheduler=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(
                    decode_host_backing=(
                        SimpleNamespace(
                            pool=SimpleNamespace(
                                shared_group=SimpleNamespace(group=group)
                                if shared
                                else None
                            )
                        )
                        if shared is not None
                        else None
                    )
                )
            )
        ),
        _deferred_releases=[(request, 0, 3, 1)],
        _do_release=lambda *args: released.append(args),
    )
    resolve = namespace["resolve_deferred_releases"]
    resolve(queue)
    if not shared:
        assert released == [(request, 3)] and not queue._deferred_releases
        resolve(queue)
        assert not group.events
        return
    assert not released and len(queue._deferred_releases) == 1
    resolve(queue)
    assert released == [(request, 3)] and not queue._deferred_releases
