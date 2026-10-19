"""Native A5 Indexer qualification adapter, separate from serving dispatch."""

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def native_indexer_ops():
    path = os.environ.get("SGLANG_LIM_NATIVE_EXTENSION")
    if not path:
        raise RuntimeError(
            "SGLANG_LIM_NATIVE_EXTENSION must select the isolated A5 LIM library"
        )
    torch.ops.load_library(path)
    return torch.ops.sgl_lim_native


@triton.jit
def _decode_scores(
    source,
    lengths,
    output,
    POOLS: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    pools = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    visible = tl.load(lengths + row) // 4
    active = (pools < POOLS) & (pools < visible)
    codes = tl.load(source + row * STRIDE + pools, mask=active, other=0).to(tl.uint32)
    bits = tl.where((codes & 0x80000000) != 0, codes ^ 0x80000000, ~codes)
    scores = bits.to(tl.float32, bitcast=True)
    tl.store(
        output + row * POOLS + pools,
        tl.where(active, scores, -float("inf")),
        pools < POOLS,
    )


@triton.jit
def _prepare_lengths(lengths, pooled, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(lengths + rows, mask=rows < ROWS, other=0)
    tl.store(pooled + rows, tl.maximum(length, 0) // 4, mask=rows < ROWS)


@triton.jit
def _expand(pool_ids, lengths, output, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(lengths + row)
    pool = tl.load(pool_ids + row * 512 + columns // 4, mask=columns < 2048, other=-1)
    pooled_end = length // 4 * 4
    token = tl.where(
        columns < 2048, pool * 4 + columns % 4, pooled_end + columns - 2048
    )
    valid = (columns >= 2048) | ((pool >= 0) & (pool < length // 4))
    token = tl.where(length <= 2048, columns, tl.where(valid, token, -1))
    token = tl.where((token >= 0) & (token < length), token, -1)
    tl.store(output + row * 2051 + columns, token, mask=columns < 2051)


class NativePooledIndexer:
    """Preallocated native Top512 plus pool expansion and score decoding.

    This stage still launches separate adaptation kernels. It is not the
    completed single-launch Indexer/residency-manager kernel.
    """

    def __init__(self, spec, row_capacity, device):
        if row_capacity < 1:
            raise ValueError("Indexer row capacity must be positive")
        self.spec = spec
        self.ops = native_indexer_ops()
        self.rows = row_capacity
        self.raw_columns = (spec.source_capacity // 4 + 127) // 128 * 128
        hardware_cores = torch.npu.get_device_properties(device).cube_core_num
        if not 0 < hardware_cores <= 64:
            raise ValueError("LIM requires an A5 device with 1..64 Cube cores")
        self.cores = min(hardware_cores, row_capacity * (self.raw_columns // 128))
        self.ends = torch.arange(1, row_capacity + 1, device=device, dtype=torch.int32)
        self.lengths = torch.empty_like(self.ends)
        self.indices = torch.empty(
            (row_capacity, 512), device=device, dtype=torch.int32
        )
        self.raw_scores = torch.empty(
            (row_capacity, self.raw_columns), device=device, dtype=torch.int32
        )
        self.scores = torch.empty(
            (row_capacity, spec.source_capacity // 4),
            device=device,
            dtype=torch.float32,
        )
        self.topk = torch.empty((row_capacity, 2051), device=device, dtype=torch.int32)
        self.workspace = torch.empty(
            self.cores * 4 * self.raw_columns * 4, device=device, dtype=torch.uint8
        )

    def select(self, query, key, weights, row_lengths, pool_table):
        rows = query.shape[0]
        if (
            not 0 < rows <= self.rows
            or row_lengths.shape != (rows,)
            or row_lengths.dtype != torch.int32
            or row_lengths.device != self.ends.device
            or not row_lengths.is_contiguous()
        ):
            raise ValueError(
                "Indexer query rows and per-query lengths must match the reservation"
            )
        _prepare_lengths[(triton.cdiv(rows, 256),)](
            row_lengths, self.lengths, rows, 256
        )
        self.ops.indexer_out(
            query,
            key,
            weights,
            self.ends[:rows],
            self.lengths[:rows],
            pool_table,
            self.indices[:rows],
            self.raw_scores[:rows],
            self.workspace,
            min(self.cores, rows * (self.raw_columns // 128)),
        )
        _decode_scores[(rows, triton.cdiv(self.scores.shape[1], 256))](
            self.raw_scores,
            row_lengths,
            self.scores,
            self.scores.shape[1],
            self.raw_columns,
            256,
        )
        _expand[(rows, triton.cdiv(2051, 256))](
            self.indices, row_lengths, self.topk, 256
        )
        return self.topk[:rows], self.scores[:rows]
