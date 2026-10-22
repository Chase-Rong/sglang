import abc
import ast
import sys
from collections import Counter
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def production_class(path, name):
    return next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


@pytest.fixture
def allocator_factory(monkeypatch):
    namespace = dict(
        abc=abc,
        torch=torch,
        get_bool_env_var=lambda _: True,
        current_platform=SimpleNamespace(
            capabilities=SimpleNamespace(supports_triton=False)
        ),
        _is_hip=False,
        next_power_of_2=lambda value: 1 << (max(1, value) - 1).bit_length(),
        get_num_new_pages=lambda **values: int(
            (values["seq_lens"] % values["page_size"] == 1).sum()
        ),
    )
    classes = [
        production_class(
            SRT / "mem_cache/allocator/base.py", "BaseTokenToKVPoolAllocator"
        ),
        production_class(
            SRT / "mem_cache/allocator/paged.py", "PagedTokenToKVPoolAllocator"
        ),
        production_class(
            SRT / "hardware_backend/npu/allocator_npu.py",
            "NPUPagedTokenToKVPoolAllocator",
        ),
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    paged_tree = ast.parse(
        (SRT / "mem_cache/allocator/paged.py").read_text(encoding="utf-8")
    )
    naive = next(
        node
        for node in paged_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "alloc_extend_naive"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, naive, *classes], type_ignores=[])
    )
    exec(compile(module, "production_allocator", "exec"), namespace)

    budget_definition = production_class(
        SRT / "mem_cache/prefill_budget.py", "PrefillBudget"
    )
    budget_namespace = {}
    exec(
        compile(
            ast.Module(body=[budget_definition], type_ignores=[]),
            "production_prefill_budget",
            "exec",
        ),
        budget_namespace,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.prefill_budget",
        SimpleNamespace(PrefillBudget=budget_namespace["PrefillBudget"]),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.unified_cache.component_type",
        SimpleNamespace(ComponentType=SimpleNamespace(FULL=0)),
    )

    class ExtendKernel:
        def __getitem__(self, grid):
            def launch(prefix, lengths, last, pages, output, batch, page_size, maximum):
                namespace["alloc_extend_naive"](
                    prefix, lengths, last, pages, output, page_size, "cpu"
                )

            return launch

    kernel_module = SimpleNamespace(alloc_extend_kernel=ExtendKernel())
    monkeypatch.setitem(
        sys.modules, "sgl_kernel_npu.mem_cache.allocator", kernel_module
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.base_prefix_cache",
        SimpleNamespace(EvictParams=SimpleNamespace),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.common",
        SimpleNamespace(_evict_until_allocatable=lambda *args: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.hardware_backend.npu.hisparse.hicache_offload",
        SimpleNamespace(
            get_offload_backing=lambda pool: getattr(pool, "decode_host_backing", None)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.hardware_backend.npu.hicache_indexer_mapping",
        SimpleNamespace(get_indexer_mapping=lambda _: None),
    )

    def create(need_sort=False, enabled=True, size=512, pool_page_size=128):
        reserved = []
        released = []
        binding = SimpleNamespace(
            lock=RLock(),
            retired=[],
            references=torch.zeros(size // 128 + 1, dtype=torch.int64),
            cache_references=torch.zeros(size // 128 + 1, dtype=torch.int64),
            ensure_pages=lambda pages: reserved.extend(pages.tolist()) or True,
            release_device_pages=lambda pages: released.extend(pages.tolist()),
            restore_mapping=lambda host, device: None,
            reclaim_for_capacity=lambda count: None,
        )
        backing = SimpleNamespace(
            page_binding=binding,
            bind_hicache=lambda _: None,
            hicache_pool=SimpleNamespace(
                clear=lambda: None, available_size=lambda: size
            ),
        )
        pool = SimpleNamespace(
            decode_host_backing=backing if enabled else None,
            page_size=pool_page_size,
            size=size,
        )
        allocator = namespace["NPUPagedTokenToKVPoolAllocator"](
            size, 128, torch.bfloat16, "cpu", pool, need_sort
        )
        return SimpleNamespace(
            allocator=allocator,
            binding=binding,
            reserved=reserved,
            released=released,
            kernel=kernel_module,
        )

    return create


@pytest.mark.parametrize("need_sort", [False, True])
def test_mixed_group_free_releases_each_owned_page_once(allocator_factory, need_sort):
    state = allocator_factory(need_sort)
    allocator = state.allocator
    rows = allocator.alloc(384)
    assert state.reserved == [1, 2, 3]
    allocator.free_group_begin()
    allocator.free(rows[:64])
    allocator.free(rows[64:128])
    allocator.free_segment(rows[128:256], start_pos=0)
    page_ids = rows[256::128] // 128
    allocator.free_page_ids(page_ids)
    page_ids.fill_(4)
    assert not state.released
    allocator.free_group_end()
    assert Counter(state.released) == Counter([1, 2, 3])
    assert allocator.available_size() == 512
    assert allocator.get_all_free_pages().unique().numel() == 4


def test_host_pressure_does_not_consume_device_pages(allocator_factory):
    state = allocator_factory()
    state.binding.ensure_pages = lambda _: False
    assert state.allocator.alloc(128) is None
    assert state.allocator.available_size() == 512
    assert not state.released


def test_failed_host_restore_rolls_back_only_the_unbound_device_allocation(
    allocator_factory,
):
    state = allocator_factory()

    def reject(host, device):
        raise ValueError("unowned Host pages")

    state.binding.restore_mapping = reject
    with pytest.raises(ValueError, match="unowned"):
        state.allocator.alloc_for_host_restore(torch.arange(128, 256))
    assert state.allocator.available_size() == 512
    assert not state.reserved
    assert not state.released


def test_decode_reuses_last_page_when_no_free_page_remains(allocator_factory):
    state = allocator_factory()
    state.allocator.alloc(512)
    result = state.allocator.alloc_decode(
        torch.tensor([2]), torch.tensor([2]), torch.tensor([128])
    )
    assert result.tolist() == [129]
    assert state.reserved == [1, 2, 3, 4]


def test_closed_offload_keeps_ordinary_allocator_capacity(allocator_factory):
    state = allocator_factory(enabled=False)
    rows = state.allocator.alloc(256)
    state.allocator.free_page_ids(rows[::128] // 128)
    assert state.allocator.available_size() == 512
    assert not state.reserved
    assert not state.released


def test_offload_allocator_rejects_radix_group_as_physical_page_size(allocator_factory):
    with pytest.raises(ValueError, match="physical"):
        allocator_factory(pool_page_size=512)


def test_request_page_demand_counts_physical_pages_inside_one_radix_group(
    allocator_factory,
):
    state = allocator_factory()
    allocator = state.allocator
    allocator.alloc(128)
    prefixes = torch.tensor([127, 0])
    lengths = torch.tensor([129, 129])
    demand = allocator.get_extend_allocation_demand(
        prefixes,
        lengths,
        conservative_num_tokens=1024,
        shard_size=1,
    )
    assert demand == 3 * 128
    rows = allocator.alloc_extend(
        prefixes, prefixes, lengths, lengths, torch.tensor([254, -1]), 131
    )
    assert rows.tolist() == [255, 256, *range(384, 513)]
    assert state.reserved == [1, 2, 3, 4]
    assert allocator.available_size() == 0


def test_incorrect_group_page_count_cannot_underallocate_extend(allocator_factory):
    state = allocator_factory()
    with pytest.raises(ValueError, match="physical KV pages"):
        state.allocator.alloc_extend(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([257]),
            torch.tensor([257]),
            torch.tensor([-1]),
            257,
            num_new_pages=1,
        )
    assert state.allocator.available_size() == 512
    assert not state.reserved


def test_extend_kernel_failure_rolls_back_host_reservation_without_freeing_device_twice(
    allocator_factory,
):
    state = allocator_factory()

    class FailedKernel:
        def __getitem__(self, grid):
            def launch(*args):
                raise RuntimeError("allocation kernel failed")

            return launch

    state.kernel.alloc_extend_kernel = FailedKernel()
    with pytest.raises(RuntimeError, match="kernel failed"):
        state.allocator.alloc_extend(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([129]),
            torch.tensor([129]),
            torch.tensor([-1]),
            129,
        )
    assert state.reserved == state.released == [1, 2]
    assert state.allocator.available_size() == 512
    assert state.allocator.get_all_free_pages().unique().numel() == 4


@pytest.mark.parametrize("need_sort", [False, True])
def test_overlapping_token_segment_and_page_group_releases_once(
    allocator_factory, need_sort
):
    state = allocator_factory(need_sort)
    allocator = state.allocator
    rows = allocator.alloc(256)
    allocator.free_group_begin()
    allocator.free(rows[:128])
    allocator.free_segment(rows[:128], start_pos=0)
    allocator.free_page_ids(rows[::128] // 128)
    allocator.free_group_end()
    assert Counter(state.released) == Counter([1, 2])
    assert allocator.available_size() == 512


def test_host_shortfall_retracts_even_with_free_hbm_pages(allocator_factory):
    state = allocator_factory()
    state.allocator.offload_backing.hicache_pool.available_size = lambda: 0
    assert not state.allocator.check_decode_capacity(num_tokens=128, tree_cache=None)
    assert state.allocator.check_decode_capacity(num_tokens=0, tree_cache=None)
    assert not state.reserved


def test_free_segment_releases_one_partial_physical_tail_page(allocator_factory):
    state = allocator_factory()
    rows = state.allocator.alloc(384)
    state.allocator.free_segment(rows[128:257], start_pos=128)
    assert state.released == [2, 3]
    assert state.allocator.available_size() == 384
    state.allocator.free_segment(rows[:128], start_pos=0)
    assert state.allocator.available_size() == 512


def test_extend_reuses_owned_tail_when_no_free_physical_page_remains(allocator_factory):
    state = allocator_factory()
    state.allocator.alloc(512)
    rows = state.allocator.alloc_extend(
        torch.tensor([126]),
        torch.tensor([126]),
        torch.tensor([128]),
        torch.tensor([128]),
        torch.tensor([253]),
        2,
    )
    assert rows.tolist() == [254, 255]
    assert state.reserved == [1, 2, 3, 4]
    assert state.allocator.available_size() == 0


def test_host_restore_requires_complete_physical_pages_without_new_host_claims(
    allocator_factory,
):
    state = allocator_factory()
    restored = state.allocator.alloc_for_host_restore(torch.arange(128, 256))
    assert restored.tolist() == list(range(128, 256))
    assert not state.reserved
    state.allocator.free_page_ids(restored[::128] // 128)
    assert state.released == [1]
    assert state.allocator.available_size() == 512


@pytest.mark.parametrize("mode", ["prefill", "decode", "hybrid"])
def test_configurator_selects_physical_npu_allocator_for_offload_pools(
    monkeypatch, mode
):
    path = SRT / "mem_cache/kv_cache_configurator.py"
    definition = production_class(path, "KVCacheConfigurator")
    method = next(
        node
        for node in definition.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_token_to_kv_pool_allocator"
    )
    definition.body = [method]
    definition.decorator_list = []
    definition.bases = []
    namespace = dict(
        _is_npu=True,
        get_disagg=lambda: SimpleNamespace(disaggregation_mode="decode"),
        get_schedule=lambda: SimpleNamespace(page_size=128),
        get_parallel=lambda: SimpleNamespace(attn_dcp_size=1),
        get_exec=lambda: pytest.fail(
            "offload selection must not depend on the generic backend"
        ),
        uses_native_hisparse=lambda: False,
    )
    calls = []

    def create(size, **values):
        calls.append((size, values))
        return object()

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.hardware_backend.npu.allocator_npu",
        SimpleNamespace(NPUPagedTokenToKVPoolAllocator=create),
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, definition], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    configurator = namespace["KVCacheConfigurator"]()
    configurator.is_hybrid_swa = False
    configurator.kv_cache_dtype = torch.bfloat16
    configurator.device = "npu"
    pool = SimpleNamespace(
        decode_host_backing=object() if mode != "prefill" else None,
        prefill_layer_slots=2 if mode != "decode" else None,
    )
    result = configurator._build_token_to_kv_pool_allocator(
        sizes=SimpleNamespace(max_total_num_tokens=1024),
        token_to_kv_pool=pool,
        is_dsv4_model=False,
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=None,
    )
    assert result is not None
    assert calls[0][0] == 1024
    assert calls[0][1]["page_size"] == 128
    assert calls[0][1]["kvcache"] is pool


def prefill_tree(*, evictable=0, root=None):
    return SimpleNamespace(
        supports_mamba=lambda: False,
        evictable_size=lambda: evictable,
        tree_core=SimpleNamespace(root_node=root),
    )


def host_cache_node(page=None, *, locked=False, children=()):
    return SimpleNamespace(
        children={index: child for index, child in enumerate(children)},
        component_data=[
            SimpleNamespace(
                value=None,
                host_value=None
                if page is None
                else torch.arange(page * 128, (page + 1) * 128),
                lock_ref=0,
                host_lock_ref=int(locked),
            )
        ],
        write_through_pending_id=None,
        load_back_pending_id=None,
    )


def test_prefill_admission_and_pending_reservations_follow_host_capacity(
    allocator_factory,
):
    state = allocator_factory()
    state.allocator.offload_backing.hicache_pool.available_size = lambda: 256
    budget = state.allocator.create_prefill_budget(
        prefill_tree(), num_mixed_decode_tokens=1
    )
    assert state.allocator.available_size() == 512
    assert budget.remaining_current == 255
    assert not budget.check_prefill(
        extend_input_len=256,
        total_tokens=384,
        max_new_tokens=0,
        input_tokens=256,
        swa_host_hit_length=0,
        chunk_limit=None,
    )[0]
    budget.reserve(1, 0)
    assert budget.remaining_current == -1
    assert not budget.has_capacity()
    assert not state.reserved and not state.released


def test_prefill_counts_host_leaf_chain_once_and_rechecks_locks(allocator_factory):
    state = allocator_factory()
    state.allocator.offload_backing.hicache_pool.available_size = lambda: 0
    state.binding.references[1:3] = 1
    state.binding.cache_references[1:3] = 1
    leaf = host_cache_node(2)
    parent = host_cache_node(1, children=[leaf])
    root = host_cache_node(children=[parent])
    budget = state.allocator.create_prefill_budget(prefill_tree(root=root))
    assert budget.remaining_current == 256
    leaf.component_data[0].host_lock_ref = 1
    assert budget.remaining_current == 0
    leaf.component_data[0].host_lock_ref = 0
    state.binding.references[2] += 1
    assert budget.remaining_current == 128
    assert state.binding.references.tolist() == [0, 1, 2, 0, 0]
    assert not state.released


def test_prefill_retired_host_pages_count_as_waitable_capacity(allocator_factory):
    state = allocator_factory()
    state.allocator.offload_backing.hicache_pool.available_size = lambda: 0
    state.binding.retired = [(object(), torch.tensor([1, 2]))]
    budget = state.allocator.create_prefill_budget(prefill_tree())
    assert budget.remaining_current == 256
    assert len(state.binding.retired) == 1


def test_closed_prefill_keeps_main_capacity_without_host_constraints(allocator_factory):
    state = allocator_factory(enabled=False)
    budget = state.allocator.create_prefill_budget(prefill_tree(evictable=128))
    assert budget.remaining_current == 640


if __name__ == "__main__":
    run_pytest(__file__)
