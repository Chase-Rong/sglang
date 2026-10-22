"""Private per-rank HiCache payload for resident NPU MLA and packed MTP.

The ordinary HiCache controller owns allocation, eviction and transfer events.
Compressed indexer payload is a separate NPUIndexerPoolHost using the same
logical page ownership; it must not be copied at uncompressed KV addresses.
"""

from collections import deque

import torch

from sglang.srt.hardware_backend.npu.hicache_indexer_mapping import (
    IndexerPageMapping,
    prepare_npu_hicache_indices,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost


class ResidentIndexerBinding(IndexerPageMapping):
    """Keep private resident page metadata live until its L2 readers finish."""

    def __init__(self, page_size, index_kpool, device_pages):
        super().__init__(page_size, index_kpool, device_pages)
        self.reader_events = deque()

    def record_reader(self, event):
        with self._lock:
            while self.reader_events and self.reader_events[0].query():
                self.reader_events.popleft()
            self.reader_events.append(event)

    def drain(self):
        with self._lock:
            while self.reader_events:
                event = self.reader_events[0]
                if not event.query():
                    event.synchronize()
                self.reader_events.popleft()

    def release_device_pages(self, device_pages):
        self.drain()
        super().release_device_pages(device_pages)

    def clear(self):
        self.drain()
        super().clear()


class NpuResidentHiCachePool(MLATokenToKVPoolHost):
    prepare_transfer_indices = staticmethod(prepare_npu_hicache_indices)

    def __init__(
        self,
        device_pool,
        host_to_device_ratio,
        host_size,
        page_size,
        layout="layer_first",
        pin_memory=True,
        device="cpu",
        allocator_type="default",
        *,
        mtp_draft_device_pools=(),
        pool_label="kv",
    ):
        if getattr(device_pool, "integrated_offload", False):
            raise ValueError("Resident HiCache must not allocate an offload backing")
        # Packed MLA storage gives contiguous component rows only without rope.
        # Nonzero rope needs separate pinned K/V storage and serialization.
        if device_pool.qk_rope_head_dim != 0:
            raise NotImplementedError(
                "Resident NPU HiCache currently requires qk_rope_head_dim=0"
            )
        if torch.device(device).type != "cpu" or (
            str(device_pool.device) != "cpu" and not pin_memory
        ):
            raise ValueError("Resident NPU HiCache requires pinned CPU host memory")
        if page_size <= 0 or page_size % device_pool.page_size:
            raise ValueError("Radix pages must contain complete physical KV pages")
        self.mtp_draft_device_pools = tuple(mtp_draft_device_pools)
        pools = (device_pool, *self.mtp_draft_device_pools)
        if len({id(pool) for pool in pools}) != len(pools):
            raise ValueError("Resident target and draft pools must be distinct")
        for pool in pools:
            if getattr(pool, "layer_shard_enabled", False):
                raise NotImplementedError(
                    "Resident NPU HiCache does not support layer sharding"
                )
            for name in (
                "page_size",
                "kv_lora_rank",
                "qk_rope_head_dim",
                "store_dtype",
                "size",
            ):
                if getattr(pool, name) != getattr(device_pool, name):
                    raise ValueError(f"Resident MTP {name} geometry differs")
            if pool is not device_pool and pool.layer_num != 1:
                raise ValueError("Resident HiCache supports single-layer MTP drafts")
        self.override_kv_cache_dim = None
        self.can_use_jit = False
        self._destroyed = False
        # Reuse host allocation/accounting and serialization, but not CUDA
        # device pointer packing or the packed Ascend dim-exchange path.
        (pool_info,) = device_pool.get_device_pool_infos()
        pool_info.buffer_info.validate(layer_ids=pool_info.layer_ids)
        super().__init__(
            pool_info.buffer_info,
            host_to_device_ratio,
            host_size,
            device_pool.page_size,
            "layer_first",
            pin_memory,
            device,
            allocator_type,
            device_capacity_tokens=device_pool.size,
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            target_layer_num=device_pool.layer_num,
            draft_buffer_infos=tuple(
                pool.get_device_pool_infos()[0].buffer_info
                for pool in self.mtp_draft_device_pools
            ),
            pool_label=pool_label,
        )
        self.device_pool = device_pool
        mapping = getattr(device_pool, "hicache_indexer_mapping", None)
        if mapping is None:
            mapping = ResidentIndexerBinding(
                self.page_size,
                device_pool.index_kpool,
                device_pool.size // self.page_size + 1,
            )
            device_pool.hicache_indexer_mapping = mapping
        self.binding = mapping
        self._device_layers = tuple(
            (pool, layer) for pool in pools for layer in range(pool.layer_num)
        )

    def clear(self):
        binding = getattr(self, "binding", None)
        if binding is not None:
            binding.drain()
        super().clear()

    def destroy(self):
        self.binding.drain()
        super().destroy()

    def _copy_plan(self, host_indices, device_indices, io_backend):
        if self._destroyed:
            raise RuntimeError("Resident HiCache pool is destroyed")
        if io_backend != "kernel_ascend":
            raise ValueError("Resident NPU HiCache requires kernel_ascend")
        pages = []
        for indices, capacity in (
            (host_indices, self.size),
            (device_indices, self.device_pool.size + self.page_size),
        ):
            if (
                indices.device.type != "cpu"
                or indices.ndim != 1
                or indices.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(
                    "Stage whole pages of integer indices on CPU before transfer"
                )
            if indices.numel() % self.page_size:
                raise ValueError("Resident HiCache transfers require whole pages")
            blocks = indices.reshape(-1, self.page_size)
            starts = blocks[:, 0]
            if (
                (starts < 0).any()
                or (starts + self.page_size > capacity).any()
                or (starts % self.page_size).any()
                or starts.unique().numel() != starts.numel()
                or not torch.equal(
                    blocks, starts[:, None] + torch.arange(self.page_size)
                )
            ):
                raise ValueError(
                    "Resident HiCache requires distinct aligned whole pages"
                )
            pages.append(starts.tolist())
        if len(pages[0]) != len(pages[1]):
            raise ValueError("Resident Host/device transfer lengths differ")
        runs = []
        for host_start, device_start in zip(*pages, strict=True):
            if (
                runs
                and host_start == runs[-1][0] + runs[-1][2]
                and device_start == runs[-1][1] + runs[-1][2]
            ):
                runs[-1] = (*runs[-1][:2], runs[-1][2] + self.page_size)
            else:
                runs.append((host_start, device_start, self.page_size))
        return tuple(runs)

    def _copy_layer(self, host_layer, runs, *, to_host):
        pool, local_layer = self._device_layers[host_layer]
        for buffer, offset, width in (
            (pool.k_buffer, 0, self.kv_lora_rank),
            (pool.v_buffer, self.kv_lora_rank, self.qk_rope_head_dim),
        ):
            if width == 0:
                continue
            device_rows = buffer[local_layer].view(-1, width)
            for host_start, device_start, count in runs:
                host_value = self.kv_buffer[
                    host_layer,
                    host_start : host_start + count,
                    0,
                    offset : offset + width,
                ]
                device_value = device_rows[device_start : device_start + count]
                destination, source = (
                    (host_value, device_value)
                    if to_host
                    else (device_value, host_value)
                )
                destination.copy_(source, non_blocking=True)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if device_pool is not self.device_pool:
            raise ValueError("Resident backup requires its target pool")
        runs = self._copy_plan(host_indices, device_indices, io_backend)
        for host_layer in range(self.layer_num):
            self._copy_layer(host_layer, runs, to_host=True)

    def prepare_load_to_device(
        self, device_pool, host_indices, device_indices, io_backend, *, is_draft=False
    ):
        if is_draft:
            depth = next(
                (
                    i
                    for i, pool in enumerate(self.mtp_draft_device_pools)
                    if pool is device_pool
                ),
                None,
            )
            if depth is None:
                raise ValueError("Unknown resident MTP draft pool")
            first, last = (
                self.target_layer_num + depth,
                self.target_layer_num + depth + 1,
            )
        else:
            if device_pool is not self.device_pool:
                raise ValueError("Unknown resident target pool")
            first, last = 0, self.target_layer_num
        runs = self._copy_plan(host_indices, device_indices, io_backend)

        def load_layer(layer_id):
            if self._destroyed:
                raise RuntimeError("Resident HiCache pool is destroyed")
            if not first <= layer_id < last:
                raise IndexError("Resident HiCache layer is outside the selected pool")
            self._copy_layer(layer_id, runs, to_host=False)

        return load_layer

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        self.prepare_load_to_device(
            device_pool,
            host_indices,
            device_indices,
            io_backend,
            is_draft=is_draft,
        )(layer_id)
