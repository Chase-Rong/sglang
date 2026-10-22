"""Aligned shared reserves and private side pools use one Host allowance."""

from types import SimpleNamespace

import pytest
import test_hicache_memfabric_pool as host_tests
import torch
from test_hicache_indexer_pool import device_and_anchor
from test_hicache_memfabric_pool import SRT, load_module

from sglang.srt.hardware_backend.npu.hisparse.decode_host_kv import (
    _prepare_host_memory_account,
)
from sglang.srt.mem_cache.pool_host import base
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1 << 30


@pytest.fixture
def runtime(monkeypatch):
    yield from host_tests.runtime.__wrapped__(monkeypatch)


@pytest.mark.parametrize("rank", range(4))
def test_cgroup_os_reserve_and_tp_share_bound_aligned_physical_allocation(
    monkeypatch, rank
):
    monkeypatch.setattr(base, "available_host_memory_bytes", lambda **kwargs: 14 * GIB)
    monkeypatch.setattr(base, "ranks_per_host", lambda: 4)
    shared = SimpleNamespace(
        group=SimpleNamespace(
            rank_in_group=rank,
            world_size=4,
            all_gather_object=lambda value: [value] * 4,
        )
    )
    fabric = SimpleNamespace(_active_pool=None)
    account, _, _ = _prepare_host_memory_account(fabric, 3 * GIB + 1, shared)
    assert account.budget_bytes == GIB
    assert account.remaining_bytes == 0
    with base.ensure_host_memory_budget_scope(account):
        assert base.host_memory_budget_bytes(1) == 0


def test_aligned_reserve_is_rejected_before_memfabric_initialization(
    runtime, monkeypatch
):
    monkeypatch.setattr(
        runtime.base_module, "host_memory_budget_bytes", lambda *args, **kwargs: GIB
    )
    before = list(runtime.events)
    with pytest.raises(MemoryError, match="reserve rejected"):
        runtime.make(host_budget=GIB + 1)
    assert runtime.events == before


def test_target_draft_borrowing_and_real_indexer_sidepools_charge_once(
    runtime, monkeypatch
):
    budget_function = runtime.base_module.host_memory_budget_bytes
    monkeypatch.setattr(
        runtime.base_module, "host_memory_budget_bytes", lambda *args, **kwargs: 3 * GIB
    )
    target, target_device = runtime.make(page_size=8, host_budget=GIB + 1)
    draft, draft_device = runtime.make(page_size=8, layers=1, host_budget=GIB + 1)
    account = target.host_memory_account
    assert account is draft.host_memory_account
    assert (
        runtime.backing_module.borrowed_host_physical_bytes((target, draft)) == 2 * GIB
    )
    assert account.remaining_bytes == GIB
    host = target.bind_hicache(target_device)
    host.attach_mtp_drafts((draft_device,))
    assert account.remaining_bytes == GIB
    budget_api = runtime.base_module
    monkeypatch.setattr(budget_api, "host_memory_budget_bytes", budget_function)
    module = load_module(
        monkeypatch,
        "sglang.srt.hardware_backend.npu.hicache_indexer_pool",
        SRT / "hardware_backend/npu/hicache_indexer_pool.py",
    )
    device, anchor = device_and_anchor(with_scale=True)
    assert module.host_memory_budget_bytes is budget_function
    with budget_api.ensure_host_memory_budget_scope(account):
        first = module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
        second = module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
        expected = first.kv_buffer.nbytes + second.kv_buffer.nbytes
        assert first.allocated_bytes + second.allocated_bytes == expected
        assert budget_api.host_memory_budget_bytes() == GIB - expected
    assert account.remaining_bytes == GIB - expected


def test_nested_account_scopes_keep_all_private_pool_charges():
    account = base.HostMemoryAllocationAccount(1000)
    account.charge("memfabric", 400)
    account.charge("memfabric", 400)
    with base.ensure_host_memory_budget_scope(account):
        assert base.host_memory_budget_bytes(100) == 600
        with base.ensure_host_memory_budget_scope(account):
            assert base.host_memory_budget_bytes(200) == 500
        assert base.host_memory_budget_bytes() == 300
    assert account.remaining_bytes == 300
    with base.ensure_host_memory_budget_scope(account):
        assert base.host_memory_budget_bytes(300) == 300
    assert account.remaining_bytes == 0


def test_explicit_mamba_host_capacity_is_not_multiplied_by_generic_ratio(monkeypatch):
    from sglang.srt.mem_cache.device_pool_info import (
        DevicePoolInfo,
        MambaStateBufferInfo,
    )
    from sglang.srt.mem_cache.hicache_storage import PoolName
    from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
    from sglang.srt.mem_cache.pool_host import mamba

    monkeypatch.setattr(assembler, "is_npu", lambda: False)
    monkeypatch.setattr(
        assembler, "get_memory", lambda: SimpleNamespace(hicache_ratio=99)
    )
    monkeypatch.setattr(assembler, "_get_allocator_type", lambda: "default")
    monkeypatch.setitem(
        mamba.ALLOC_MEMORY_FUNCS,
        "cpu",
        lambda dims, **kwargs: torch.empty(dims, dtype=kwargs["dtype"]),
    )
    info = DevicePoolInfo(
        pool_name=PoolName.MAMBA,
        indices_from_pool=PoolName.MAMBA,
        layer_ids=(0,),
        buffer_info=MambaStateBufferInfo(
            temporal=torch.zeros(1, 4, 2),
            conv=(torch.zeros(1, 4, 2),),
            extra_checkpoint_buffers=(torch.zeros(4, 3, dtype=torch.int32),),
        ),
    )
    account = base.HostMemoryAllocationAccount(1000)
    with base.ensure_host_memory_budget_scope(account):
        host = assembler.build_mamba_host_pool(
            SimpleNamespace(size=3, host_capacity_tokens=5),
            info,
            host_size=1,
            layout="page_first_direct",
            preserve_host_capacity=True,
        )
    assert host.size == 6
    assert account.remaining_bytes == 1000 - host.size * host.size_per_token


if __name__ == "__main__":
    run_pytest(__file__)
