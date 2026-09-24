"""CPU contract tests; runtime imports/controller and NPU streams are not exercised."""

import abc
import ast
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).parents[4] / "python/sglang/srt"


def definitions(monkeypatch, name, path, selected, **namespace):
    module = ModuleType(name)
    module.__dict__.update(namespace)
    monkeypatch.setitem(sys.modules, name, module)
    source = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in source.body if getattr(node, "name", None) in selected]
    assert {node.name for node in nodes} == set(selected)
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


@pytest.fixture
def pool_module(monkeypatch):
    base = definitions(
        monkeypatch,
        "sglang.srt.mem_cache.pool_host.base",
        ROOT / "mem_cache/pool_host/base.py",
        {"HostKVCache", "synchronized"},
        abc=abc,
        torch=torch,
        wraps=wraps,
    )
    base.host_memory_budget_bytes = lambda: 1 << 40
    name = "sglang.srt.hardware_backend.npu.hicache_indexer_pool"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "hardware_backend/npu/hicache_indexer_pool.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def device_and_anchor(with_scale=False):
    device = SimpleNamespace(
        device="npu:0",
        layer_num=2,
        start_layer=0,
        end_layer=2,
        page_size=8,
        index_kpool=2,
        _kpool_use_compress=True,
        tail_extra_slots=0,
        integrated_offload=True,
        index_k_buffer=torch.arange(2 * 9 * 8 * 4)
        .reshape(2, 9, 8, 1, 4)
        .to(torch.int8 if with_scale else torch.bfloat16),
    )
    if with_scale:
        device.index_scale_buffer = (
            torch.arange(2 * 9 * 8).reshape(2, 9, 8, 1, 1).float() / 7
        )
    locations = {3: 24, 5: 28, 1: 8, 6: 12}
    binding = SimpleNamespace(
        indexer_page_locations=lambda pages: torch.tensor(
            [locations[int(page)] for page in pages], dtype=torch.int64
        )
    )
    anchor = SimpleNamespace(
        page_size=8,
        page_num=7,
        size=56,
        logical_size=56,
        device="cpu",
        layout="layer_first",
        layer_num=2,
        target_layer_num=2,
        device_pool=device,
        can_use_write_back_jit=False,
        binding=binding,
        available_size=Mock(return_value=48),
        free=Mock(),
        clear=Mock(),
        destroy=Mock(),
    )
    device.decode_host_backing = SimpleNamespace(hicache_pool=anchor)
    return device, anchor


def token_rows(pages):
    return (
        torch.tensor(pages, dtype=torch.int64)[:, None] * 8 + torch.arange(8)
    ).flatten()


@pytest.mark.parametrize("index_kpool", [2, 4, 8])
def test_resolver_offsets_match_existing_glm_bf16_write_locations(
    pool_module, monkeypatch, index_kpool
):
    reference = definitions(
        monkeypatch,
        "hicache_indexer_location_reference",
        ROOT / "layers/attention/dsa/kpool_bf16_index.py",
        {"compute_pooled_write_locs"},
        torch=torch,
        BLOCK_SIZE_K=128,
    )
    page_size = 128
    table = torch.tensor([9, 3, 15, 1, 6, 20, 8, 7, 18, 13])
    actual = pool_module.indexer_locations_from_page_table(
        table, page_size, index_kpool
    )
    pooled_ids = torch.arange(table.numel()) * (page_size // index_kpool)
    expected = reference.compute_pooled_write_locs(table, pooled_ids, index_kpool)
    assert torch.equal(actual, expected)
    host_page_table = table + 100
    assert not torch.equal(
        actual,
        pool_module.indexer_locations_from_page_table(
            host_page_table, page_size, index_kpool
        ),
    )


def test_layer_transfers_reject_unstaged_device_metadata(pool_module):
    device, anchor = device_and_anchor()
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    with pytest.raises(ValueError, match="Stage HiCache"):
        host.backup_from_device_all_layer(
            device,
            token_rows([1]),
            torch.empty(8, dtype=torch.int64, device="meta"),
            "kernel_ascend",
        )
    anchor.binding.indexer_page_locations = lambda pages: torch.empty(
        1, dtype=torch.int64, device="meta"
    )
    with pytest.raises(ValueError, match="CPU integer"):
        host.backup_from_device_all_layer(
            device, token_rows([1]), token_rows([3]), "kernel_ascend"
        )


def test_prepared_load_snapshots_indices_and_resolves_only_once(
    pool_module, monkeypatch
):
    device, anchor = device_and_anchor()
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    original = device.index_k_buffer[:, 3].clone()
    host.backup_from_device_all_layer(
        device, token_rows([4, 2]), token_rows([5, 3]), "kernel_ascend"
    )
    resolver = Mock(wraps=anchor.binding.indexer_page_locations)
    anchor.binding.indexer_page_locations = resolver
    host_indices, device_indices = token_rows([4, 2]), token_rows([6, 1])
    load_layer = host.prepare_load_to_device(
        device, host_indices, device_indices, "kernel_ascend"
    )
    resolver.assert_called_once()
    host_indices.zero_()
    device_indices.zero_()
    device.index_k_buffer.fill_(-1)
    monkeypatch.setattr(
        host, "_transfer_rows", Mock(side_effect=AssertionError("metadata repeated"))
    )
    for layer in range(2):
        load_layer(layer)
    assert torch.equal(device.index_k_buffer[:, 1], original)
    resolver.assert_called_once()


@pytest.mark.parametrize("with_scale", [False, True])
@pytest.mark.parametrize("io_backend", ["kernel", "direct", "kernel_ascend"])
def test_fragmented_compressed_slices_roundtrip(pool_module, with_scale, io_backend):
    device, anchor = device_and_anchor(with_scale)
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    saved_keys = device.index_k_buffer.clone()
    saved_scales = getattr(device, "index_scale_buffer", None)
    if saved_scales is not None:
        saved_scales = saved_scales.clone()
    host.backup_from_device_all_layer(
        device, token_rows([4, 2]), token_rows([5, 3]), io_backend
    )
    device.index_k_buffer.fill_(-7)
    if with_scale:
        device.index_scale_buffer.fill_(-9)
    for layer in range(2):
        host.load_to_device_per_layer(
            device, token_rows([4, 2]), token_rows([6, 1]), layer, io_backend
        )
        expected = torch.full_like(device.index_k_buffer[layer], -7)
        expected[1, :4] = saved_keys[layer, 3, :4]
        expected[1, 4:] = saved_keys[layer, 3, 4:]
        assert torch.equal(device.index_k_buffer[layer], expected)
        if with_scale:
            expected_scale = torch.full_like(device.index_scale_buffer[layer], -9)
            expected_scale[1] = saved_scales[layer, 3]
            assert torch.equal(device.index_scale_buffer[layer], expected_scale)
        if layer == 0:
            assert torch.all(device.index_k_buffer[1] == -7)
    assert host.allocated_bytes == host.kv_buffer.nbytes
    assert host.logical_size == anchor.size
    assert host.get_size_per_token() * host.size == host.allocated_bytes


@pytest.mark.parametrize("with_scale", [False, True])
def test_expanded_host_indexer_restores_high_pages_to_small_device(
    pool_module, with_scale
):
    device, anchor = device_and_anchor(with_scale)
    anchor.page_num = 33
    anchor.size = anchor.logical_size = 264
    original_shape = device.index_k_buffer.shape
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    saved = device.index_k_buffer[:, 3].clone()
    scales = device.index_scale_buffer[:, 3].clone() if with_scale else None
    host.backup_from_device_all_layer(
        device, token_rows([31, 32]), token_rows([3, 5]), "kernel_ascend"
    )
    device.index_k_buffer.fill_(-1)
    if with_scale:
        device.index_scale_buffer.fill_(-1)
    for layer in range(2):
        host.load_to_device_per_layer(
            device, token_rows([31, 32]), token_rows([1, 6]), layer, "kernel_ascend"
        )
    assert torch.equal(device.index_k_buffer[:, 1], saved)
    if with_scale:
        assert torch.equal(device.index_scale_buffer[:, 1], scales)
    assert device.index_k_buffer.shape == original_shape
    assert host.page_num == 33


def test_split_node_backup_keeps_each_kv_pages_own_compressed_slice(pool_module):
    device, anchor = device_and_anchor()
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    first_slice = device.index_k_buffer[:, 3, :4].clone()
    host.backup_from_device_all_layer(
        device, token_rows([1]), token_rows([3]), "direct"
    )
    device.index_k_buffer[:, 3, 4:].fill_(123)
    host.backup_from_device_all_layer(
        device, token_rows([2]), token_rows([5]), "direct"
    )
    device.index_k_buffer.fill_(-1)
    for layer in range(2):
        host.load_to_device_per_layer(
            device, token_rows([2]), token_rows([6]), layer, "direct"
        )
        host.load_to_device_per_layer(
            device, token_rows([1]), token_rows([1]), layer, "direct"
        )
        assert torch.equal(device.index_k_buffer[layer, 1, :4], first_slice[layer])
        assert torch.all(device.index_k_buffer[layer, 1, 4:] == 123)


@pytest.mark.parametrize("with_scale", [False, True])
def test_transfers_copy_direct_final_host_slices_without_gathers(
    pool_module, monkeypatch, with_scale
):
    device, anchor = device_and_anchor(with_scale)
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    host_storage = host.kv_buffer.untyped_storage().data_ptr()
    device_storage = {
        buffers[0].untyped_storage().data_ptr() for buffers, _, _, _ in host._components
    }
    calls = []
    original_copy = torch.Tensor.copy_

    def checked_copy(destination, source, non_blocking=False):
        assert non_blocking
        pointers = {
            destination.untyped_storage().data_ptr(),
            source.untyped_storage().data_ptr(),
        }
        assert host_storage in pointers
        assert pointers.intersection(device_storage)
        assert destination.is_contiguous() and source.is_contiguous()
        calls.append((destination.numel(), source.numel()))
        return original_copy(destination, source, non_blocking=non_blocking)

    def reject_gather(*args, **kwargs):
        pytest.fail(
            "Payload copies must not allocate index_select/index_copy temporaries"
        )

    monkeypatch.setattr(torch.Tensor, "copy_", checked_copy)
    monkeypatch.setattr(torch.Tensor, "index_select", reject_gather)
    monkeypatch.setattr(torch.Tensor, "index_copy_", reject_gather)
    host.backup_from_device_all_layer(
        device, token_rows([4, 2]), token_rows([5, 3]), "kernel_ascend"
    )
    for layer in range(2):
        host.load_to_device_per_layer(
            device, token_rows([4, 2]), token_rows([6, 1]), layer, "kernel_ascend"
        )
    assert len(calls) == 8 * len(host._components)


@pytest.mark.parametrize("layer_num", [1, 2])
def test_runs_coalesce_only_when_both_physical_slices_are_contiguous(
    pool_module, layer_num
):
    device, anchor = device_and_anchor()
    device.layer_num = device.end_layer = anchor.layer_num = layer_num
    device.index_k_buffer = device.index_k_buffer[:layer_num]
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    byte_count = host._components[0][2]
    assert host._copy_runs(
        torch.tensor([1, 2]), torch.tensor([24, 28]), byte_count
    ) == ((1, 24, 2),)
    assert host._copy_runs(
        torch.tensor([1, 3]), torch.tensor([24, 28]), byte_count
    ) == ((1, 24, 1), (3, 28, 1))
    host.backup_from_device_all_layer(
        device, token_rows([1, 2]), token_rows([3, 5]), "kernel_ascend"
    )
    original = device.index_k_buffer[:, 3].clone()
    load_layer = host.prepare_load_to_device(
        device, token_rows([1, 2]), token_rows([1, 6]), "kernel_ascend"
    )
    for layer in range(layer_num):
        load_layer(layer)
    assert torch.equal(device.index_k_buffer[:, 1], original)


def test_large_coalescing_plan_accounts_for_hbm_group_gaps(pool_module):
    device, anchor = device_and_anchor()
    device.index_kpool = 4
    anchor.page_num = 8193
    anchor.size = anchor.page_num * device.page_size
    device.index_k_buffer = torch.empty((2, 8193, 8, 1, 4), dtype=torch.bfloat16)
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    pages = torch.arange(1, 8193)
    locations = pool_module.indexer_locations_from_page_table(pages, 8, 4)
    byte_count = host._components[0][2]
    runs = host._copy_runs(pages, locations, byte_count)
    assert len(runs) == 2048
    assert all(count == 4 for _, _, count in runs)
    dense_locations = 8 + torch.arange(8192) * host.slots_per_page
    assert host._copy_runs(pages, dense_locations, byte_count) == ((1, 8, 8192),)


def test_storage_roundtrip_and_zero_copy_metadata(pool_module):
    device, anchor = device_and_anchor(True)
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    host.backup_from_device_all_layer(
        device, token_rows([4]), token_rows([5]), "direct"
    )
    serialized = host.get_data_page(32).clone()
    host.set_from_flat_data_page(16, serialized)
    assert torch.equal(host.get_data_page(16), serialized)
    assert not host.get_data_page(16, flat=False).is_contiguous()
    pointers, lengths = host.get_page_buffer_meta(token_rows([4, 2]))
    assert pointers == [
        host.kv_buffer[layer, page].data_ptr() for page in (4, 2) for layer in range(2)
    ]
    assert lengths == [host.bytes_per_layer] * 4
    assert sum(lengths) == 2 * serialized.numel()
    assert host.get_dummy_flat_data_page().shape == serialized.shape
    with pytest.raises(ValueError, match="dtype or size"):
        host.set_from_flat_data_page(0, serialized[:-1])


@pytest.mark.parametrize(
    "indices",
    [torch.arange(7), torch.arange(1, 9), token_rows([1, 1]), token_rows([9])],
)
def test_invalid_pages_fail_before_copy(pool_module, indices):
    device, anchor = device_and_anchor()
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    with pytest.raises(ValueError):
        host.backup_from_device_all_layer(device, indices, indices, "direct")
    assert not torch.any(host.kv_buffer)


def test_missing_mapping_fails_and_empty_transfer_is_safe(pool_module):
    device, anchor = device_and_anchor()
    anchor.binding = None
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    empty = torch.empty(0, dtype=torch.int64)
    host.backup_from_device_all_layer(device, empty, empty, "direct")
    with pytest.raises(RuntimeError, match="indexer_page_locations"):
        host.backup_from_device_all_layer(
            device, token_rows([1]), token_rows([3]), "direct"
        )


@pytest.mark.parametrize("locations", [[0, 8], [24, 24], [25, 32], [-4, 8], [72, 8]])
def test_invalid_resolver_results_fail_before_copy(pool_module, locations):
    device, anchor = device_and_anchor()
    anchor.binding.indexer_page_locations = lambda pages: torch.tensor(locations)
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    with pytest.raises(ValueError, match="locations"):
        host.backup_from_device_all_layer(
            device, token_rows([1, 2]), token_rows([3, 5]), "direct"
        )
    assert not torch.any(host.kv_buffer)


def test_budget_and_invalid_geometry_fail_before_allocation(pool_module, monkeypatch):
    device, anchor = device_and_anchor()
    monkeypatch.setattr(pool_module, "host_memory_budget_bytes", lambda: 1)
    with pytest.raises(ValueError, match="additional host bytes"):
        pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    device.index_kpool = 3
    with pytest.raises(ValueError, match="divide"):
        pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)


@pytest.mark.parametrize("with_scale", [False, True])
@pytest.mark.parametrize("draft_count", [0, 1, 2])
def test_mtp_complete_groups_roundtrip_target_and_packed_drafts(
    pool_module, with_scale, draft_count
):
    device, anchor = device_and_anchor(with_scale)
    device.tail_extra_slots = 3
    drafts = []
    for depth in range(draft_count):
        draft, _ = device_and_anchor(with_scale)
        draft.layer_num = draft.end_layer = 1
        draft.index_k_buffer = draft.index_k_buffer[:1].clone().fill_(40 + depth)
        if with_scale:
            draft.index_scale_buffer = (
                draft.index_scale_buffer[:1].clone().fill_(3 + depth)
            )
        drafts.append(draft)
    anchor.mtp_draft_device_pools = tuple(drafts)
    anchor.layer_num += draft_count
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    assert host.target_layer_num == 2
    assert host.layer_num == 2 + draft_count
    pools = [device, *drafts]
    names = ["index_k_buffer"] + (["index_scale_buffer"] if with_scale else [])
    originals = [
        [getattr(pool, name)[:, 3].clone() for name in names] for pool in pools
    ]
    host.backup_from_device_all_layer(
        device, token_rows([4, 2]), token_rows([5, 3]), "direct"
    )
    serialized = host.get_data_page(32).clone()
    assert serialized.numel() == host.bytes_per_layer * (2 + draft_count)
    for pool in pools:
        for name in names:
            getattr(pool, name).fill_(-1)
    load_target = host.prepare_load_to_device(
        device, token_rows([4, 2]), token_rows([6, 1]), "direct"
    )
    with pytest.raises(IndexError):
        load_target(2)
    for layer in range(2):
        load_target(layer)
    for draft in drafts:
        assert torch.all(draft.index_k_buffer == -1)
    for depth, draft in enumerate(drafts):
        load_draft = host.prepare_load_to_device(
            draft, token_rows([4, 2]), token_rows([6, 1]), "direct", is_draft=True
        )
        with pytest.raises(IndexError):
            load_draft(0)
        with pytest.raises(IndexError):
            load_draft(3 + depth)
        load_draft(2 + depth)
    for pool, expected in zip(pools, originals):
        for name, value in zip(names, expected):
            assert torch.equal(getattr(pool, name)[:, 1], value)
    with pytest.raises(ValueError, match="device pool"):
        host.prepare_load_to_device(
            device_and_anchor(with_scale)[0],
            token_rows([4, 2]),
            token_rows([6, 1]),
            "direct",
            is_draft=True,
        )


@pytest.mark.parametrize(
    "indices", [torch.arange(7), token_rows([3]), token_rows([3, 3])]
)
def test_mtp_rejects_partial_or_duplicate_compression_groups(pool_module, indices):
    device, anchor = device_and_anchor()
    device.tail_extra_slots = 3
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    with pytest.raises(ValueError):
        host.backup_from_device_all_layer(
            device, token_rows([1, 2])[: indices.numel()], indices, "direct"
        )
    assert not torch.any(host.kv_buffer)
    with pytest.raises(ValueError):
        host.prepare_load_to_device(
            device, token_rows([1, 2])[: indices.numel()], indices, "direct"
        )


def test_mtp_rejects_slices_from_two_incomplete_groups(pool_module):
    device, anchor = device_and_anchor()
    device.tail_extra_slots = 3
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    with pytest.raises(ValueError, match="complete compression"):
        host.backup_from_device_all_layer(
            device, token_rows([1, 2]), token_rows([3, 1]), "direct"
        )
    assert not torch.any(host.kv_buffer)


def test_mtp_backup_and_restore_never_access_temporary_tail(pool_module):
    class NoTailAccess(SimpleNamespace):
        def __getattribute__(self, name):
            if name in (
                "_compress_tail_k",
                "_compress_tail_score",
                "get_compress_tail_buffers",
            ):
                pytest.fail("HiCache must not access speculative request tail rings")
            return super().__getattribute__(name)

    device, anchor = device_and_anchor()
    device = NoTailAccess(**vars(device))
    device.tail_extra_slots = 3
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    expected = device.index_k_buffer[:, 3].clone()
    host.backup_from_device_all_layer(
        device, token_rows([1, 2]), token_rows([3, 5]), "direct"
    )
    device.index_k_buffer.fill_(-1)
    for layer in range(2):
        host.load_to_device_per_layer(
            device, token_rows([1, 2]), token_rows([1, 6]), layer, "direct"
        )
    assert torch.equal(device.index_k_buffer[:, 1], expected)


@pytest.mark.parametrize(
    "invalid",
    ["layers", "page_size", "index_kpool", "capacity", "dtype", "scale", "cp"],
)
def test_mtp_rejects_incompatible_draft_before_allocation(pool_module, invalid):
    device, anchor = device_and_anchor()
    draft, _ = device_and_anchor()
    draft.layer_num = draft.end_layer = 1
    draft.index_k_buffer = draft.index_k_buffer[:1]
    if invalid == "layers":
        draft.layer_num = 2
    elif invalid == "page_size":
        draft.page_size = 4
    elif invalid == "index_kpool":
        draft.index_kpool = 4
    elif invalid == "capacity":
        draft.index_k_buffer = draft.index_k_buffer[:, :8]
    elif invalid == "dtype":
        draft.index_k_buffer = draft.index_k_buffer.float()
    elif invalid == "scale":
        draft.index_scale_buffer = torch.zeros((1, 9, 8, 1, 1))
    else:
        draft.layer_shard_enabled = True
    anchor.mtp_draft_device_pools = (draft,)
    anchor.layer_num = 3
    with pytest.raises((ValueError, NotImplementedError)):
        pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)


def test_sidecar_never_owns_anchor_lifetime(pool_module):
    device, anchor = device_and_anchor()
    host = pool_module.NPUIndexerPoolHost(device, anchor, pin_memory=False)
    with pytest.raises(RuntimeError, match="SidecarPoolSpec"):
        host.alloc(8)
    with pytest.raises(RuntimeError, match="Only the KV anchor"):
        host.free(token_rows([1]))
    assert host.available_size() == 48
    host.clear()
    host.destroy()
    host.destroy()
    anchor.free.assert_not_called()
    anchor.clear.assert_not_called()
    anchor.destroy.assert_not_called()
    with pytest.raises(RuntimeError, match="destroyed"):
        host.get_data_page(0)


@pytest.mark.parametrize("tail_extra_slots", [0, 3])
def test_aligned_prefix_needs_no_old_tail_in_existing_bf16_reference(
    monkeypatch, tail_extra_slots
):
    reference = definitions(
        monkeypatch,
        "hicache_indexer_tail_reference",
        ROOT / "layers/attention/dsa/kpool_bf16_index.py",
        {"kpool_decode_update_and_maybe_write_cache_bf16"},
        torch=torch,
        INDEX_HEAD_DIM=128,
    )
    pool = SimpleNamespace(
        index_kpool=4, tail_extra_slots=tail_extra_slots, page_size=128
    )
    cache = torch.full((6, 128, 128), -1, dtype=torch.bfloat16)
    tail_keys = torch.full(
        (1, 4 + tail_extra_slots, 128), float("nan"), dtype=torch.bfloat16
    )
    tail_scores = torch.full_like(tail_keys, float("nan"))
    for step in range(4):
        reference.kpool_decode_update_and_maybe_write_cache_bf16(
            pool=pool,
            buf=cache,
            tail_k=tail_keys,
            tail_score=tail_scores,
            key=torch.full((1, 128), step + 1, dtype=torch.bfloat16),
            slot_score=torch.zeros((1, 128), dtype=torch.bfloat16),
            ape=torch.zeros((4, 128)),
            block_tables=torch.tensor([[3, 5]]),
            req_pool_indices=torch.tensor([0]),
            positions=torch.tensor([128 + step]),
            seq_lens=torch.tensor([129 + step]),
            out_cache_loc=torch.tensor([640 + step]),
        )
        if step < 3:
            assert torch.all(cache == -1)
    expected = torch.full_like(cache, -1)
    expected[3, 32] = 2.5
    assert torch.equal(cache, expected)
    written = torch.arange(128, 132) % (4 + tail_extra_slots)
    assert not torch.isnan(tail_keys[:, written]).any()
    assert not torch.isnan(tail_scores[:, written]).any()


@pytest.fixture
def assembler(monkeypatch, pool_module):
    storage = definitions(
        monkeypatch,
        "sglang.srt.mem_cache.hicache_storage",
        ROOT / "mem_cache/hicache_storage.py",
        {"PoolName", "PoolHitPolicy", "PoolTransfer", "SidecarPoolSpec"},
        Enum=Enum,
        dataclass=dataclass,
    )
    group = definitions(
        monkeypatch,
        "hicache_indexer_test_group",
        ROOT / "mem_cache/pool_host/group.py",
        {"PoolEntry", "HostPoolGroup"},
        dataclass=dataclass,
        PoolName=storage.PoolName,
    )
    component = definitions(
        monkeypatch,
        "hicache_indexer_test_component",
        ROOT / "mem_cache/unified_cache/component_type.py",
        {"ComponentType"},
        Enum=Enum,
    )
    memory_pool = ModuleType("sglang.srt.mem_cache.memory_pool")
    memory_pool.HybridLinearKVPool = type("HybridLinearKVPool", (), {})
    monkeypatch.setitem(sys.modules, memory_pool.__name__, memory_pool)
    mamba_module = ModuleType("sglang.srt.hardware_backend.npu.hicache_mamba_pool")
    mamba_module.NpuMambaPoolHost = Mock(
        return_value=SimpleNamespace(can_use_write_back_jit=False)
    )
    monkeypatch.setitem(sys.modules, mamba_module.__name__, mamba_module)
    real_pool = pool_module.NPUIndexerPoolHost
    monkeypatch.setattr(
        pool_module,
        "NPUIndexerPoolHost",
        lambda device, anchor: real_pool(device, anchor, pin_memory=False),
    )
    memory = SimpleNamespace(
        hicache_ratio=2,
        hicache_size=0,
        hicache_mem_layout="page_first",
        hicache_write_policy="write_through",
        hicache_io_backend="direct",
        hicache_host_memory_mode="default",
    )
    parallel = SimpleNamespace(attn_cp_size=1, attn_dcp_size=1, dcp_enabled=False)
    module = definitions(
        monkeypatch,
        "hicache_indexer_test_assembler",
        ROOT / "mem_cache/hybrid_cache/hybrid_pool_assembler.py",
        {
            "build_kv_host_pool",
            "_has_npu_compressed_indexer",
            "_validate_npu_hicache_parallel",
            "_make_layer_mapper",
            "build_pool_entry",
            "build_hybrid_mamba_stack",
            "StackBuildResult",
            "StackStrategy",
            "_MambaStrategy",
            "_apply_stack_result",
        },
        dataclass=dataclass,
        field=field,
        PoolName=storage.PoolName,
        SidecarPoolSpec=storage.SidecarPoolSpec,
        ComponentType=component.ComponentType,
        PoolEntry=group.PoolEntry,
        HostPoolGroup=group.HostPoolGroup,
        HybridCacheController=Mock(return_value=SimpleNamespace()),
        MambaPoolHost=Mock(return_value=SimpleNamespace(can_use_write_back_jit=False)),
        get_memory=lambda: memory,
        get_parallel=lambda: parallel,
        _get_allocator_type=lambda: "default",
        logger=logging.getLogger(__name__),
    )
    module.parallel = parallel
    module.memory = memory
    module.NpuMambaPoolHost = mamba_module.NpuMambaPoolHost
    return module


def test_integrated_anchor_is_returned_by_identity(assembler):
    device, anchor = device_and_anchor()
    assert (
        assembler.build_kv_host_pool(kv_pool=device, page_size=8, use_mla=True)
        is anchor
    )
    assert (
        assembler.build_kv_host_pool(kv_pool=device, page_size=16, use_mla=True)
        is anchor
    )
    with pytest.raises(ValueError, match="does not match"):
        assembler.build_kv_host_pool(kv_pool=device, page_size=12, use_mla=True)
    device.decode_host_backing.hicache_pool = None
    with pytest.raises(RuntimeError, match="second KV backing"):
        assembler.build_kv_host_pool(kv_pool=device, page_size=8, use_mla=True)


@pytest.mark.parametrize("unsupported", ["cp", "dcp"])
def test_integrated_rejects_unsupported_combinations(assembler, unsupported):
    device, anchor = device_and_anchor()
    if unsupported == "cp":
        assembler.parallel.attn_cp_size = 2
    else:
        assembler.parallel.dcp_enabled = True
    with pytest.raises(NotImplementedError):
        assembler.build_kv_host_pool(kv_pool=device, page_size=8, use_mla=True)


def test_integrated_allows_mtp_aligned_prefixes(assembler):
    device, anchor = device_and_anchor()
    device.tail_extra_slots = 3
    assert (
        assembler.build_kv_host_pool(kv_pool=device, page_size=16, use_mla=True)
        is anchor
    )


@pytest.mark.parametrize("integrated", [False, True])
def test_full_mamba_registers_indexer_with_full_layer_mapping(assembler, integrated):
    device, anchor = device_and_anchor()
    device.integrated_offload = integrated
    assembler.MLATokenToKVPoolHost = Mock(return_value=anchor)
    assembler.memory.hicache_mem_layout = (
        "page_first_kv_split" if integrated else "page_first"
    )
    params = SimpleNamespace(
        page_size=8,
        mtp_draft_device_pools=(),
        token_to_kv_pool_allocator=object(),
        req_to_token_pool=SimpleNamespace(
            mamba_map={1: 0, 3: 1},
            mamba_pool=object(),
            mamba_allocator=SimpleNamespace(alloc=Mock(), free=Mock()),
        ),
        tp_cache_group=None,
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        pp_cache_group=None,
    )
    result = assembler._MambaStrategy().build(
        cache=SimpleNamespace(),
        kvcache=SimpleNamespace(
            full_kv_pool=device,
            use_mla=True,
            full_attention_layer_id_mapping={0: 0, 2: 1},
        ),
        params=params,
        server_args=None,
        load_cache_event=None,
    )
    if integrated:
        indexer = result.host_pool_group.get_entry(assembler.PoolName.INDEXER)
        assert [indexer.layer_mapper(layer) for layer in range(5)] == [
            0,
            None,
            1,
            None,
            None,
        ]
        assert indexer.device_pool is device
        assert indexer.host_pool.anchor_host is anchor
        assert result.sidecars == [
            assembler.SidecarPoolSpec(
                pool_name=assembler.PoolName.INDEXER,
                indices_from_pool=assembler.PoolName.KV,
            )
        ]
    else:
        assert assembler.PoolName.INDEXER not in result.host_pool_group.entry_map
        assert not result.sidecars
    assert result.transfer_layer_num == 4
    assert result.register_req_to_token_counter
    selected = assembler.NpuMambaPoolHost if integrated else assembler.MambaPoolHost
    unselected = assembler.MambaPoolHost if integrated else assembler.NpuMambaPoolHost
    assert selected.call_args.args[0] is params.req_to_token_pool.mamba_pool
    assert selected.call_args.kwargs["layout"] == "page_first"
    unselected.assert_not_called()
    if not integrated:
        return
    params.req_to_token_pool.mamba_ckpt_pool = object()
    with pytest.raises(NotImplementedError, match="INT8"):
        assembler.build_hybrid_mamba_stack(
            params=params,
            kv_pool=device,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping={0: 0, 2: 1},
            mamba_layer_mapping={1: 0, 3: 1},
            load_cache_event=None,
            storage_backend=None,
            use_mla=True,
        )
    params.req_to_token_pool.mamba_ckpt_pool = None
    assembler.memory.hicache_size = 1
    with pytest.raises(ValueError, match="separate budgets"):
        assembler.build_hybrid_mamba_stack(
            params=params,
            kv_pool=device,
            mamba_pool=params.req_to_token_pool.mamba_pool,
            full_layer_mapping={0: 0, 2: 1},
            mamba_layer_mapping={1: 0, 3: 1},
            load_cache_event=None,
            storage_backend=None,
            use_mla=True,
        )


def test_apply_result_wires_existing_host_eviction(assembler):
    device, anchor = device_and_anchor()
    cache = SimpleNamespace(evict_host=Mock())
    controller = SimpleNamespace(layer_done_counter=object())
    result = SimpleNamespace(
        host_pool_group=SimpleNamespace(anchor_entry=SimpleNamespace(host_pool=anchor)),
        cache_controller=controller,
        component_host_pools={},
        sidecars=[],
        register_req_to_token_counter=False,
        pools_desc="test",
        transfer_layer_num=2,
    )
    device.register_layer_transfer_counter = Mock()
    assembler._apply_stack_result(
        cache, device, SimpleNamespace(mtp_draft_device_pools=()), result
    )
    assert anchor.binding.evict_host is cache.evict_host
    device.register_layer_transfer_counter.assert_called_once_with(
        controller.layer_done_counter
    )
