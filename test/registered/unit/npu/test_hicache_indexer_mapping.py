"""CPU mapping and sidecar contracts; no NPU/controller/Graph qualification."""

import ast
import importlib.util
import math
import sys
from array import array
from dataclasses import replace
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from test_hicache_indexer_pool import definitions, device_and_anchor
from test_hicache_indexer_pool import pool_module as pool_module
from test_hicache_indexer_pool import token_rows

MODULE_PATH = (
    Path(__file__).parents[4]
    / "python/sglang/srt/hardware_backend/npu/hicache_indexer_mapping.py"
)
SPEC = importlib.util.spec_from_file_location("hicache_indexer_mapping", MODULE_PATH)
MAPPING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAPPING)
ROOT = MODULE_PATH.parents[2]


def make_mapping(page_size=8, index_kpool=2, device_pages=32):
    return MAPPING.IndexerPageMapping(page_size, index_kpool, device_pages)


def source_method(relative_path, class_name, method_name, **namespace):
    path = ROOT / relative_path
    source = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in source.body if getattr(node, "name", None) == class_name
    )
    method = next(
        node for node in cls.body if getattr(node, "name", None) == method_name
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[method_name]


def compressed_allocator(mapping=None, page_size=128, index_kpool=4):
    pool = SimpleNamespace(
        _kpool_use_compress=True, page_size=page_size, index_kpool=index_kpool
    )
    return SimpleNamespace(
        page_size=page_size,
        offload_backing=SimpleNamespace(
            page_binding=SimpleNamespace(indexer_mapping=mapping),
            hicache_pool=SimpleNamespace(device_pool=pool),
        ),
    )


@pytest.mark.parametrize("page_size,index_kpool", [(8, 2), (16, 4), (128, 8)])
def test_locations_follow_actual_kpool_token_address_formula(page_size, index_kpool):
    mapping = make_mapping(page_size, index_kpool)
    pages = torch.tensor([9, 2, 7, 4, 11, 6, 5, 8, 3])
    mapping.update_request_pages(pages)
    locations = mapping.indexer_page_locations(pages)
    for logical_page in range(pages.numel()):
        for compressed_slot in range(page_size // index_kpool):
            token = logical_page * page_size + (compressed_slot + 1) * index_kpool - 1
            pool_id = token // index_kpool
            token_page_row = pool_id // page_size * index_kpool
            expected = pages[token_page_row] * page_size + pool_id % page_size
            assert locations[logical_page] + compressed_slot == expected


def test_shared_prefix_is_idempotent_but_live_relocation_is_atomic():
    mapping = make_mapping()
    mapping.update_request_pages(torch.tensor([3, 5]))
    mapping.update_request_pages(torch.tensor([3, 5, 8, 9]))
    mapping.update_request_pages(torch.tensor([3, 5]))
    with pytest.raises(RuntimeError, match="relocate"):
        mapping.update_request_pages(torch.tensor([1, 5]))
    assert mapping.indexer_page_locations(torch.tensor([3, 5])).tolist() == [24, 28]
    with pytest.raises(RuntimeError, match="No logical"):
        mapping.indexer_page_locations(torch.tensor([1]))


def test_group_interior_branch_collision_is_rejected_without_partial_registration():
    mapping = make_mapping()
    mapping.update_request_pages(torch.tensor([3, 5]))
    with pytest.raises(RuntimeError, match="branches alias"):
        mapping.update_request_pages(torch.tensor([3, 6]))
    with pytest.raises(RuntimeError, match="No logical"):
        mapping.indexer_page_locations(torch.tensor([6]))


def test_group_aligned_branching_uses_distinct_anchors():
    mapping = make_mapping()
    mapping.update_request_pages(torch.tensor([3, 5, 1, 6]))
    mapping.update_request_pages(torch.tensor([3, 5, 2, 7]))
    assert mapping.indexer_page_locations(torch.tensor([1, 6, 2, 7])).tolist() == [
        8,
        12,
        16,
        20,
    ]


def test_release_anchor_requires_all_dependents_and_allows_reuse():
    mapping = make_mapping()
    mapping.update_request_pages(torch.tensor([3, 5]))
    with pytest.raises(RuntimeError, match="live dependent"):
        mapping.release_device_pages(torch.tensor([3]))
    assert mapping.indexer_page_locations(torch.tensor([5])).item() == 28
    mapping.release_device_pages(torch.tensor([5, 3]))
    with pytest.raises(RuntimeError, match="No logical"):
        mapping.indexer_page_locations(torch.tensor([5]))
    mapping.update_request_pages(torch.tensor([5, 3]))
    assert mapping.indexer_page_locations(torch.tensor([5, 3])).tolist() == [40, 44]
    mapping.clear()
    with pytest.raises(RuntimeError, match="No logical"):
        mapping.indexer_page_locations(torch.tensor([3]))


def test_restore_suffix_uses_resident_group_anchor_and_absolute_logical_position():
    mapping = make_mapping()
    mapping.update_request_pages(torch.tensor([3]))
    result = mapping.prepare_restore(token_rows([3]), token_rows([6, 1, 5]))
    assert result.tolist() == [28, 8, 12]
    assert mapping.indexer_page_locations(torch.tensor([3, 6, 1, 5])).tolist() == [
        24,
        28,
        8,
        12,
    ]


@pytest.mark.parametrize(
    "prefix,suffix",
    [
        (token_rows([3])[:-1], token_rows([5])),
        (token_rows([3]), token_rows([5])[:-1]),
        (token_rows([3]), token_rows([5]).flip(0)),
        (token_rows([3]), token_rows([3])),
    ],
)
def test_restore_rejects_partial_or_malformed_logical_rows(prefix, suffix):
    mapping = make_mapping()
    with pytest.raises(ValueError):
        mapping.prepare_restore(prefix, suffix)
    with pytest.raises(RuntimeError, match="No logical"):
        mapping.indexer_page_locations(torch.tensor([3]))


@pytest.mark.parametrize("pages", [[0], [-1], [32], [3, 3]])
def test_invalid_request_page_table(pages):
    with pytest.raises(ValueError):
        make_mapping().update_request_pages(torch.tensor(pages))


def test_empty_request_and_restore():
    mapping = make_mapping()
    empty = torch.empty(0, dtype=torch.int64)
    assert mapping.update_request_pages(empty).numel() == 0
    assert mapping.prepare_restore(empty, empty).numel() == 0
    assert mapping.indexer_page_locations(empty).dtype == torch.int64


def test_admission_and_decode_read_only_valid_logical_page_starts():
    mapping = make_mapping()
    table = torch.full((3, 32), -999, dtype=torch.int32)
    table[2, :16] = token_rows([3, 5]).int()
    table[0, :8] = token_rows([1]).int()
    mapping.update_requests(table, torch.tensor([2, 0]), torch.tensor([16, 5]))
    assert mapping.indexer_page_locations(torch.tensor([3, 5, 1])).tolist() == [
        24,
        28,
        8,
    ]
    table[2, 16] = 6 * 8
    mapping.update_requests(
        table,
        torch.tensor([2, 0]),
        torch.tensor([17, 6]),
        start_lens_cpu=torch.tensor([16, 5]),
    )
    assert mapping.indexer_page_locations(torch.tensor([6])).item() == 48
    table.fill_(-999)
    mapping.update_requests(
        table,
        torch.tensor([2, 0]),
        torch.tensor([18, 7]),
        start_lens_cpu=torch.tensor([17, 6]),
    )


def test_decode_new_nonanchor_page_uses_group_start_from_request_table():
    mapping = make_mapping()
    table = token_rows([7, 2]).reshape(1, -1)
    mapping.update_requests(table, torch.tensor([0]), torch.tensor([8]))
    mapping.update_requests(
        table,
        torch.tensor([0]),
        torch.tensor([9]),
        start_lens_cpu=torch.tensor([8]),
    )
    assert mapping.indexer_page_locations(torch.tensor([2])).item() == 60


def test_incremental_update_requires_preexisting_group_anchor():
    mapping = make_mapping()
    with pytest.raises(RuntimeError, match="Missing live"):
        mapping.update_requests(
            token_rows([7, 2]).reshape(1, -1),
            torch.tensor([0]),
            torch.tensor([9]),
            start_lens_cpu=torch.tensor([8]),
        )
    with pytest.raises(RuntimeError, match="No logical"):
        mapping.indexer_page_locations(torch.tensor([2]))


@pytest.mark.parametrize("wrapped", [False, True])
def test_scheduler_hook_resolves_full_allocator_and_updates_actual_table(wrapped):
    mapping = make_mapping()
    backing = SimpleNamespace(
        page_binding=SimpleNamespace(indexer_mapping=mapping),
        hicache_pool=SimpleNamespace(
            device_pool=SimpleNamespace(_kpool_use_compress=True)
        ),
    )
    allocator = SimpleNamespace(offload_backing=backing)
    if wrapped:
        allocator = SimpleNamespace(full_attn_allocator=allocator)
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(token_to_kv_pool_allocator=allocator),
        req_to_token_pool=SimpleNamespace(
            req_to_token=token_rows([3, 5]).reshape(1, -1)
        ),
    )
    MAPPING.maybe_update_indexer_mapping(batch, torch.tensor([0]), torch.tensor([9]))
    assert mapping.indexer_page_locations(torch.tensor([5])).item() == 28
    backing.page_binding.indexer_mapping = None
    with pytest.raises(RuntimeError, match="binding.indexer_mapping"):
        MAPPING.maybe_update_indexer_mapping(
            batch, torch.tensor([0]), torch.tensor([9])
        )


def test_other_allocators_are_untouched():
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace())
    )
    MAPPING.maybe_update_indexer_mapping(batch, None, None)


@pytest.mark.parametrize("with_scale", [False, True])
def test_real_sidecar_roundtrip_with_logically_derived_mapping(pool_module, with_scale):
    device, anchor = device_and_anchor(with_scale)
    mapping = make_mapping(device_pages=9)
    anchor.binding = mapping
    mapping.update_request_pages(torch.tensor([3, 5]))
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    saved_keys = device.index_k_buffer[:, 3].clone()
    saved_scale = device.index_scale_buffer[:, 3].clone() if with_scale else None
    host.backup_from_device_all_layer(
        device, token_rows([4, 2]), token_rows([5, 3]), "direct"
    )
    mapping.release_device_pages(torch.tensor([3, 5]))
    mapping.prepare_restore(token_rows([]), token_rows([1, 6]))
    device.index_k_buffer.fill_(-7)
    if with_scale:
        device.index_scale_buffer.fill_(-9)
    for layer in range(2):
        host.load_to_device_per_layer(
            device, token_rows([4, 2]), token_rows([6, 1]), layer, "direct"
        )
    assert torch.equal(device.index_k_buffer[:, 1], saved_keys)
    assert torch.all(device.index_k_buffer[:, 3] == -7)
    if with_scale:
        assert torch.equal(device.index_scale_buffer[:, 1], saved_scale)


def test_radix_groups_preserve_physical_pages():
    allocator = compressed_allocator()
    assert MAPPING.indexer_tree_page_size(allocator, 128) == 512
    assert MAPPING.indexer_tree_page_size(allocator, 512) == 512
    assert MAPPING.indexer_tree_page_size(allocator, 384) == 1536
    assert allocator.page_size == 128
    assert allocator.offload_backing.hicache_pool.device_pool.page_size == 128
    assert MAPPING.indexer_tree_page_size(SimpleNamespace(), 128) == 128


def test_real_radix_match_keeps_partial_group_private(monkeypatch):
    radix = definitions(
        monkeypatch,
        "indexer_mapping_radix",
        ROOT / "mem_cache/radix_cache.py",
        {"RadixKey"},
        array=array,
    )
    original = array("q", range(1024))
    branch = array("q", original)
    branch[700] = -1
    first = radix.RadixKey(original)
    second = radix.RadixKey(branch)
    group_size = MAPPING.indexer_tree_page_size(compressed_allocator(), 128)
    assert first.match(second, page_size=128) == 640
    assert first.match(second, page_size=group_size) == 512
    assert first[512:].child_key(group_size) != second[512:].child_key(group_size)
    assert len(radix.RadixKey(original[:700]).page_aligned(group_size)) == 512
    mapping = make_mapping(page_size=128, index_kpool=4)
    mapping.update_request_pages(torch.tensor([3, 5, 2, 7, 1, 6, 8, 9]))
    mapping.update_request_pages(torch.tensor([3, 5, 2, 7, 10, 11, 12, 13]))
    mapping.release_device_pages(torch.tensor([1, 6, 8, 9]))
    assert mapping.indexer_page_locations(torch.tensor([10, 13])).tolist() == [
        1280,
        1376,
    ]


def test_constructor_copies_tree_params_before_components_and_kda_grid(monkeypatch):
    import dataclasses
    import logging

    params_module = definitions(
        monkeypatch,
        "indexer_mapping_params",
        ROOT / "mem_cache/cache_init_params.py",
        {"CacheInitParams"},
        dataclasses=dataclasses,
    )
    runtime = definitions(
        monkeypatch,
        "indexer_mapping_grid",
        ROOT / "runtime_context.py",
        {"mamba_checkpoint_grid", "mamba_track_grid"},
        math=math,
        mamba_cache_chunk_size=lambda: 128,
        get_exec=lambda: SimpleNamespace(
            mamba=SimpleNamespace(mamba_track_interval=768)
        ),
    )

    class ComponentType(IntEnum):
        FULL = 0
        MAMBA = 1
        SWA = 2

    components = {}

    def make_component(cache, params):
        component = SimpleNamespace(
            grid=runtime.mamba_checkpoint_grid(params.page_size)
        )
        components[id(component)] = component
        return component

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.hardware_backend.npu.hicache_indexer_mapping",
        MAPPING,
    )
    construct = source_method(
        "mem_cache/unified_radix_cache.py",
        "UnifiedRadixCache",
        "__init__",
        replace=replace,
        ComponentType=ComponentType,
        COMPONENT_REGISTRY={
            ComponentType.FULL: make_component,
            ComponentType.MAMBA: make_component,
        },
        envs=SimpleNamespace(
            SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=SimpleNamespace(get=lambda: "python")
        ),
        create_tree_core=lambda **kwargs: SimpleNamespace(
            page_size=kwargs["params"].page_size
        ),
        UnifiedSessionRefTracker=Mock(),
        StreamingSession=Mock(),
        StorageExistenceCache=Mock(),
        logger=logging.getLogger(__name__),
    )
    allocator = compressed_allocator()
    params = params_module.CacheInitParams(
        disable=False,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=allocator,
        page_size=128,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
    )
    cache = SimpleNamespace(reset=Mock())
    construct(cache, params)
    assert params.page_size == allocator.page_size == 128
    assert cache.tree_core.page_size == 512
    assert cache._indexer_grouped_pages
    assert {component.grid for component in components.values()} == {512}
    assert runtime.mamba_track_grid(cache.tree_core.page_size) == 1536


def test_restore_plan_uses_radix_ancestors_and_registers_before_enqueue():
    mapping = make_mapping()
    mapping.update_request_pages(torch.tensor([3, 5]))
    allocator = compressed_allocator(mapping, page_size=8, index_kpool=2)
    allocator.alloc_for_host_restore = Mock(return_value=token_rows([1, 6]))
    allocator.free = Mock(
        side_effect=lambda rows: mapping.release_device_pages(rows[::8] // 8)
    )
    nodes = {12: SimpleNamespace(parent=SimpleNamespace(id=11))}
    collect = Mock(return_value=token_rows([3, 5]))
    cache = SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        page_size=16,
        tree_core=SimpleNamespace(
            node_by_id=nodes.__getitem__, collect_full_device_indices=collect
        ),
        root_node=SimpleNamespace(id=0),
    )
    build_plan = source_method(
        "mem_cache/unified_radix_cache.py", "UnifiedRadixCache", "_build_loadback_plan"
    )
    transfer = SimpleNamespace(host_indices=token_rows([2, 4]), nodes_to_load=[12])
    prepare = build_plan(cache, transfer)
    collect.assert_called_once_with(11, 0)
    with pytest.raises(RuntimeError, match="No logical"):
        mapping.indexer_page_locations(torch.tensor([1]))

    def resolve(*args, **kwargs):
        assert mapping.indexer_page_locations(torch.tensor([1, 6])).tolist() == [8, 12]
        assert not controller.load_queue
        return []

    controller = SimpleNamespace(
        mem_pool_device_allocator=allocator,
        device="cpu",
        load_queue=[],
        mem_pool_host=SimpleNamespace(acquire_device_pages=lambda: None),
        _resolve_device_transfers=resolve,
    )
    rollback = source_method(
        "mem_cache/hybrid_cache/hybrid_cache_controller.py",
        "HybridCacheController",
        "_rollback_load",
    )
    controller._rollback_load = lambda *args: rollback(controller, *args)
    load = source_method(
        "mem_cache/hybrid_cache/hybrid_cache_controller.py",
        "HybridCacheController",
        "load",
        torch=torch,
        CacheOperation=lambda *args, **kwargs: (args, kwargs),
    )
    result = load(controller, transfer.host_indices, prepare_device_indices=prepare)
    assert torch.equal(result, token_rows([1, 6]))
    assert len(controller.load_queue) == 1
    allocator.free.assert_not_called()
    mapping.release_device_pages(torch.tensor([1, 6]))
    controller.load_queue.clear()

    def fail_resolution(*args, **kwargs):
        raise RuntimeError("auxiliary allocation failed")

    controller._resolve_device_transfers = fail_resolution
    with pytest.raises(RuntimeError, match="auxiliary allocation"):
        load(controller, transfer.host_indices, prepare_device_indices=prepare)
    assert not controller.load_queue
    allocator.free.assert_called_once()
    assert mapping.indexer_page_locations(torch.tensor([3, 5])).tolist() == [24, 28]
    with pytest.raises(RuntimeError, match="No logical"):
        mapping.indexer_page_locations(torch.tensor([1]))


def test_runtime_l3_attach_is_rejected_before_backend_changes():
    attach = source_method(
        "mem_cache/unified_radix_cache.py",
        "UnifiedRadixCache",
        "attach_storage_backend",
    )
    attachment = Mock()
    success, message = attach(
        SimpleNamespace(_indexer_grouped_pages=True, _storage_attachment=attachment),
        "unused",
    )
    assert not success and "physical-page" in message
    attachment.attach.assert_not_called()


def test_streaming_session_private_tail_uses_physical_page_lifetime():
    namespace = {"ceil_align": lambda value, multiple: -(-value // multiple) * multiple}
    trim = source_method(
        "session/streaming_session.py",
        "StreamingSession",
        "_free_kv_aligned",
        **namespace,
    )
    record = SimpleNamespace(
        holds_kv=True,
        req_pool_idx=0,
        kv_allocated_len=900,
        cache_protected_len=512,
        swa_evicted_seqlen=512,
    )
    session = SimpleNamespace(
        page_size=512,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=128),
        free_kv_row=Mock(),
        slots={"request": SimpleNamespace(kv=record)},
    )
    trim(session, record, 600, 900)
    session.free_kv_row.assert_called_once_with(record, [(640, 900)])
    record.kv_allocated_len = 700
    for method_name in ("session_held_tokens", "session_held_swa_tokens"):
        count = source_method(
            "session/streaming_session.py", "StreamingSession", method_name, **namespace
        )
        assert count(session) == 768 - 512


def test_npu_allocation_dispatch_uses_physical_page_size():
    path = ROOT / "mem_cache/allocation.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node
        for node in source.body
        if getattr(node, "name", None) == "_alloc_page_size"
    )
    namespace = {"_is_npu": True, "ScheduleBatch": object}
    tree = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    batch = SimpleNamespace(
        tree_cache=SimpleNamespace(
            page_size=512, token_to_kv_pool_allocator=SimpleNamespace(page_size=128)
        )
    )
    assert namespace["_alloc_page_size"](batch) == 128
