"""CPU contracts for speculative allocation with real compressed-page mapping."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch
from test_hicache_indexer_pool import ROOT, definitions


@pytest.fixture
def allocation_case(monkeypatch):
    name = "sglang.srt.hardware_backend.npu.hicache_indexer_mapping"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "hardware_backend/npu/hicache_indexer_mapping.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    mapping = module.IndexerPageMapping(8, 2, 32)
    table = torch.full((3, 48), -999, dtype=torch.int64)
    table[2, :8] = torch.arange(24, 32)
    mapping.update_request_pages(torch.tensor([3]))
    allocated = set()
    pending = []

    def allocate(*args, **kwargs):
        pages = torch.tensor(pending, dtype=torch.int64)
        allocated.update(pending)
        return (pages[:, None] * 8 + torch.arange(8)).flatten()

    def free(rows):
        pages = torch.unique(rows // 8)
        mapping.release_device_pages(pages)
        allocated.difference_update(pages.tolist())

    def assign(indices, target, starts, ends, rows, batch_size):
        offset = 0
        for slot, start, end in zip(indices.tolist(), starts.tolist(), ends.tolist()):
            target[slot, start:end] = rows[offset : offset + end - start]
            offset += end - start

    allocator = SimpleNamespace(
        page_size=8,
        free=free,
        offload_backing=SimpleNamespace(
            page_binding=SimpleNamespace(indexer_mapping=mapping),
            hicache_pool=SimpleNamespace(
                device_pool=SimpleNamespace(_kpool_use_compress=True)
            ),
        ),
    )
    req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=8))
    tree_cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    pool = SimpleNamespace(req_to_token=table)
    indices = torch.tensor([2], dtype=torch.int64)
    batch = SimpleNamespace(
        device="cpu",
        tree_cache=tree_cache,
        req_to_token_pool=pool,
        req_pool_indices_cpu=indices,
    )
    production = definitions(
        monkeypatch,
        "mtp_allocation_under_test",
        ROOT / "mem_cache/allocation.py",
        {"alloc_for_spec_decode"},
        torch=torch,
        _is_npu=True,
        get_last_loc=lambda target, slots, lengths: target[slots, lengths - 1],
        ALLOC_EXTEND_FUNCS={"cpu": allocate},
        assign_req_to_token_pool_func=assign,
    )

    def run(start=8, end=16, pages=(5,)):
        pending[:] = pages
        production.alloc_for_spec_decode(
            tree_cache,
            pool,
            reqs=[req],
            req_pool_indices=indices,
            cur_kv_lens=torch.tensor([start]),
            cur_kv_lens_cpu=torch.tensor([start]),
            nxt_kv_lens=torch.tensor([end]),
            nxt_kv_lens_cpu=torch.tensor([end]),
            num_needed_tokens=(end - start),
            batch=batch,
        )

    return SimpleNamespace(
        run=run,
        mapping=mapping,
        table=table,
        req=req,
        allocated=allocated,
        allocator=allocator,
    )


def test_speculative_new_pages_register_after_table_write(allocation_case):
    case = allocation_case
    case.run(end=32, pages=(5, 7, 2))
    assert case.mapping.indexer_page_locations(torch.tensor([3, 5, 7, 2])).tolist() == [
        24,
        28,
        56,
        60,
    ]
    assert case.table[2, 8:16].tolist() == list(range(40, 48))
    assert case.req.kv.kv_allocated_len == 32


def test_speculative_existing_reserve_needs_no_mapping_read(allocation_case):
    case = allocation_case
    case.table.fill_(-999)
    case.run(start=8, end=8, pages=())
    assert case.mapping.indexer_page_locations(torch.tensor([3])).item() == 24
    assert not case.allocated
    assert case.req.kv.kv_allocated_len == 8


def test_failed_speculative_mapping_returns_new_pages(allocation_case):
    case = allocation_case
    # Page 6 already owns the second compressed slice under anchor page 3.
    case.mapping.update_request_pages(torch.tensor([3, 6]))
    with pytest.raises(RuntimeError, match="branches alias"):
        case.run()
    assert not case.allocated
    assert case.req.kv.kv_allocated_len == 8
    assert case.mapping.indexer_page_locations(torch.tensor([3, 6])).tolist() == [
        24,
        28,
    ]
    with pytest.raises(RuntimeError, match="No logical"):
        case.mapping.indexer_page_locations(torch.tensor([5]))
