"""Page-granular HiCache sidecar for the NPU compressed DSA indexer.

The anchor owns allocation and page lifetime. Its binding must implement
``indexer_page_locations(device_pages) -> int64 Tensor[num_pages]``: each result
is the first flattened index-buffer row for that KV page's compressed keys.
For logical KV page q, this is table[q // kpool * kpool] * page_size +
(q % kpool) * (page_size // kpool). Physical page IDs alone cannot recover q.
The binding must return CPU metadata, retain this mapping until transfer
completion, establish it before load, and serialize writes to overlapping
compressed slices. CacheController.move_indices stages kernel_ascend indices
on CPU once per operation. Layer copies never stage device metadata or payload
on the host. L2 completion protects the final pinned Host slots in both directions.

Only complete KV pages are accepted. A page boundary divisible by kpool has no
live compression tail: kpool_index_npu.py compresses at slot == POOL_SIZE - 1;
dsa_indexer_kpool.py builds subsequent pools from the new chunk. Old request
tail rings are overwritten before reuse. Unaligned prefixes and speculative
tail rollback require request state and are deliberately unsupported here.
The current NPU BF16 path has no scale; a separate index_scale_buffer, when
present, is transferred with identical row mapping and its original dtype.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache, host_memory_budget_bytes

logger = logging.getLogger(__name__)


def indexer_locations_from_page_table(device_page_table, page_size, index_kpool):
    """Derive CPU resolver entries from one request's live, unconverted KV table.

    The table starts at logical page zero and excludes padding. Its values
    remain HBM device page IDs, as in AscendDSAIndexerMetadata.get_page_table_64.
    Bind each returned row to the corresponding input physical page ID before
    backup/load; retain the group-leading device page while its slices are live.
    This helper does not discover request ownership or mutate an offload binding.
    """
    if (
        device_page_table.device.type != "cpu"
        or device_page_table.ndim != 1
        or device_page_table.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Indexer page table must be a CPU integer vector")
    if index_kpool <= 1 or page_size <= 0 or page_size % index_kpool:
        raise ValueError("Indexer page_size must be divisible by index_kpool")
    if torch.any(device_page_table <= 0) or (
        device_page_table.unique().numel() != device_page_table.numel()
    ):
        raise ValueError("Indexer page table must contain live distinct device pages")
    logical_pages = torch.arange(device_page_table.numel(), dtype=torch.int64)
    group_leaders = logical_pages // index_kpool * index_kpool
    return device_page_table.to(torch.int64)[group_leaders] * page_size + (
        logical_pages % index_kpool
    ) * (page_size // index_kpool)


class NPUIndexerPoolHost(HostKVCache):
    """Store compressed slices under the anchor's uncompressed KV page IDs."""

    def __init__(self, device_pool, anchor_host, *, pin_memory=True):
        self.device_pool = device_pool
        self.anchor_host = anchor_host
        self.page_size = anchor_host.page_size
        self.index_kpool = device_pool.index_kpool
        if not getattr(device_pool, "_kpool_use_compress", False):
            raise ValueError("NPU HiCache indexer requires compressed KPool")
        if (
            self.index_kpool <= 1
            or self.page_size <= 0
            or self.page_size != device_pool.page_size
            or self.page_size % self.index_kpool
        ):
            raise ValueError(
                "KV page_size must match the device and divide by index_kpool"
            )
        if getattr(anchor_host, "dcp_size", 1) != 1 or getattr(
            device_pool, "layer_shard_enabled", False
        ):
            raise NotImplementedError(
                "NPU HiCache indexer CP/layer sharding is unsupported"
            )
        if getattr(device_pool, "tail_extra_slots", 0) or getattr(
            anchor_host, "mtp_draft_device_pools", ()
        ):
            raise NotImplementedError(
                "NPU HiCache indexer MTP tail restore is unsupported"
            )
        self.slots_per_page = self.page_size // self.index_kpool
        self.size = anchor_host.size
        self.page_num = anchor_host.page_num
        if self.size != self.page_num * self.page_size:
            raise ValueError("Anchor capacity is not a whole number of KV pages")
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = self.target_layer_num = device_pool.layer_num
        self.mtp_draft_device_pools = ()
        self.layout = "layer_first"
        self.device = "cpu"
        self.dtype = torch.uint8
        self.pin_memory = pin_memory
        self.can_use_write_back_jit = False
        self.allocator = None
        self._destroyed = False
        self._components = []
        self.bytes_per_layer = 0
        for name in ("index_k_buffer", "index_scale_buffer"):
            buffers = getattr(device_pool, name, None)
            if buffers is None:
                if name == "index_k_buffer":
                    raise ValueError("NPU index_k_buffer is missing")
                continue
            if len(buffers) != self.layer_num or not self.layer_num:
                raise ValueError(f"{name} must contain every logical FULL layer")
            reference = buffers[0]
            if reference.device.type != "cpu" and not pin_memory:
                raise ValueError(
                    "NPU indexer asynchronous copies require pinned Host memory"
                )
            if reference.ndim < 2 or reference.shape[1] != self.page_size:
                raise ValueError(f"{name} must have [pages, page_size, ...] layout")
            if not reference.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            views = []
            for buffer in buffers:
                if (
                    buffer.shape != reference.shape
                    or buffer.dtype != reference.dtype
                    or not buffer.is_contiguous()
                    or buffer.device != reference.device
                ):
                    raise ValueError(f"{name} layer layouts differ")
                views.append(buffer.view(buffer.shape[0] * self.page_size, -1))
            if self._components and views[0].shape[0] != self.device_rows:
                raise ValueError("Indexer key and scale page capacities differ")
            self.device_rows = views[0].shape[0]
            width = views[0].shape[1]
            byte_count = self.slots_per_page * width * reference.element_size()
            self._components.append((views, self.bytes_per_layer, byte_count, width))
            self.bytes_per_layer += (byte_count + 7) // 8 * 8
        if (
            not self._components[0][0][0].is_floating_point()
            and len(self._components) == 1
        ):
            raise ValueError("Quantized NPU indexer keys require index_scale_buffer")
        self.bytes_per_page = self.layer_num * self.bytes_per_layer
        self.size_per_token = self.get_size_per_token()
        self.allocated_bytes = self.page_num * self.bytes_per_page
        available_bytes = host_memory_budget_bytes()
        if self.allocated_bytes > available_bytes:
            raise ValueError(
                f"NPU HiCache indexer needs {self.allocated_bytes} additional host bytes; "
                f"rank budget has {available_bytes}. KV backing and Mamba are separate."
            )
        self.kv_buffer = self.init_kv_buffer()
        logger.info(
            "NPU HiCache indexer allocates %d host bytes (%d compressed slots/KV page); "
            "anchor KV backing is reused, Mamba budget is separate.",
            self.allocated_bytes,
            self.slots_per_page,
        )

    def get_size_per_token(self):
        return self.bytes_per_page / self.page_size

    def init_kv_buffer(self):
        return torch.zeros(
            (self.layer_num, self.page_num, self.bytes_per_layer),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def _require_live(self):
        if self._destroyed:
            raise RuntimeError("NPU HiCache indexer pool is destroyed")

    def _pages(self, indices, capacity):
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("HiCache indices must be a one-dimensional integer tensor")
        if indices.device.type != "cpu":
            raise ValueError("Stage HiCache indices on CPU once before layer transfers")
        rows = indices.to(device="cpu", dtype=torch.int64)
        if rows.numel() % self.page_size:
            raise ValueError(
                "NPU indexer cannot restore a partial KV page or live tail"
            )
        rows = rows.reshape(-1, self.page_size)
        starts = rows[:, 0]
        if (
            torch.any(starts < 0)
            or torch.any(starts + self.page_size > capacity)
            or torch.any(starts % self.page_size)
            or not torch.equal(rows, starts[:, None] + torch.arange(self.page_size))
        ):
            raise ValueError(
                "HiCache indices must contain ordered, aligned whole pages"
            )
        pages = starts // self.page_size
        if pages.unique().numel() != pages.numel():
            raise ValueError("Duplicate KV pages in an indexer transfer")
        return pages

    def _transfer_rows(self, device_pool, host_indices, device_indices, io_backend):
        self._require_live()
        if device_pool is not self.device_pool:
            raise ValueError("NPU indexer sidecar belongs to a different device pool")
        if io_backend not in ("kernel", "direct", "kernel_ascend"):
            raise ValueError(f"Unsupported NPU indexer IO backend: {io_backend}")
        host_pages = self._pages(host_indices, self.size)
        device_pages = self._pages(device_indices, self.device_rows)
        if host_pages.numel() != device_pages.numel():
            raise ValueError("Host and device page counts differ")
        if not device_pages.numel():
            return host_pages, torch.empty(0, dtype=torch.int64)
        binding = getattr(self.anchor_host, "binding", None)
        resolver = getattr(binding, "indexer_page_locations", None)
        if not callable(resolver):
            raise RuntimeError(
                "HiCache binding must implement indexer_page_locations(device_pages); "
                "compressed KPool addresses cannot be inferred from physical KV pages"
            )
        locations = resolver(device_pages)
        if (
            not isinstance(locations, torch.Tensor)
            or locations.shape != device_pages.shape
            or locations.dtype not in (torch.int32, torch.int64)
            or locations.device.type != "cpu"
        ):
            raise ValueError(
                "indexer_page_locations must return one CPU integer row per page"
            )
        locations = locations.to(device="cpu", dtype=torch.int64)
        if (
            torch.any(locations < self.page_size)
            or torch.any(locations + self.slots_per_page > self.device_rows)
            or torch.any(locations % self.slots_per_page)
            or locations.unique().numel() != locations.numel()
        ):
            raise ValueError("Invalid, overlapping, or reserved indexer page locations")
        return host_pages, locations

    def _copy_runs(self, host_pages, locations, byte_count):
        runs = []
        contiguous_host_pages = byte_count == self.bytes_per_layer
        for host_page, location in zip(
            host_pages.tolist(), locations.tolist(), strict=True
        ):
            if (
                runs
                and contiguous_host_pages
                and host_page == runs[-1][0] + runs[-1][2]
                and location == runs[-1][1] + runs[-1][2] * self.slots_per_page
            ):
                start_page, start_row, count = runs[-1]
                runs[-1] = (start_page, start_row, count + 1)
            else:
                runs.append((host_page, location, 1))
        return tuple(runs)

    def _copy_layer(self, layer_id, plans, *, to_host):
        for (buffers, offset, byte_count, width), runs in zip(
            self._components, plans, strict=True
        ):
            buffer = buffers[layer_id]
            for host_page, device_row, page_count in runs:
                host_bytes = self.kv_buffer[
                    layer_id,
                    host_page : host_page + page_count,
                    offset : offset + byte_count,
                ]
                if not host_bytes.is_contiguous():
                    raise RuntimeError(
                        "Indexer copy run must use contiguous final Host storage"
                    )
                host_value = host_bytes.view(buffer.dtype).view(-1, width)
                device_value = buffer[
                    device_row : device_row + page_count * self.slots_per_page
                ]
                destination, source = (
                    (host_value, device_value)
                    if to_host
                    else (device_value, host_value)
                )
                destination.copy_(source, non_blocking=True)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host_pages, locations = self._transfer_rows(
            device_pool, host_indices, device_indices, io_backend
        )
        plans = [
            self._copy_runs(host_pages, locations, component[2])
            for component in self._components
        ]
        for layer in range(self.layer_num):
            self._copy_layer(layer, plans, to_host=True)

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
            device_pool, host_indices, device_indices, io_backend, is_draft=is_draft
        )(layer_id)

    def prepare_load_to_device(
        self, device_pool, host_indices, device_indices, io_backend, *, is_draft=False
    ):
        """Snapshot CPU metadata once before L2 enters its per-layer copy loop."""
        if is_draft:
            raise NotImplementedError("NPU HiCache indexer MTP restore is unsupported")
        host_pages, locations = self._transfer_rows(
            device_pool, host_indices, device_indices, io_backend
        )
        plans = tuple(
            self._copy_runs(host_pages, locations, component[2])
            for component in self._components
        )

        def load_layer(layer_id):
            self._require_live()
            if not 0 <= layer_id < self.layer_num:
                raise IndexError("NPU indexer layer is outside this pool")
            self._copy_layer(layer_id, plans, to_host=False)

        return load_layer

    def _page_index(self, index):
        self._require_live()
        index = int(index)
        if index < 0 or index >= self.size or index % self.page_size:
            raise ValueError(
                "Host data page must start at an aligned in-range KV index"
            )
        return index // self.page_size

    def get_data_page(self, index, flat=True):
        page = self.kv_buffer[:, self._page_index(index)]
        return page.contiguous().flatten() if flat else page

    def get_dummy_flat_data_page(self):
        self._require_live()
        return torch.zeros(self.bytes_per_page, dtype=self.dtype)

    def set_from_flat_data_page(self, index, data_page):
        page = self.get_data_page(index, flat=False)
        if data_page.dtype != self.dtype or data_page.numel() != self.bytes_per_page:
            raise ValueError("Serialized NPU indexer page has the wrong dtype or size")
        page.copy_(data_page.reshape_as(page))

    def get_hybrid_pool_buffer(self):
        self._require_live()
        return [self.kv_buffer]

    def get_page_buffer_meta(self, indices):
        """Return page-major lists of per-layer ranges, never one fictitious span."""
        self._require_live()
        pages = self._pages(indices, self.size)
        pointers = [
            self.kv_buffer[layer, int(page)].data_ptr()
            for page in pages
            for layer in range(self.layer_num)
        ]
        return pointers, [self.bytes_per_layer] * len(pointers)

    def is_stride_page_aligned(self, page_size_bytes=4096):
        self._require_live()
        return (
            self.bytes_per_layer % page_size_bytes == 0
            and self.kv_buffer.data_ptr() % page_size_bytes == 0
        )

    def alloc(self, need_size):
        raise RuntimeError("INDEXER uses SidecarPoolSpec indices from the KV anchor")

    def free(self, indices):
        raise RuntimeError("Only the KV anchor may release shared INDEXER page indices")

    def available_size(self):
        self._require_live()
        return self.anchor_host.available_size()

    def clear(self):
        self._require_live()

    def destroy(self):
        self._destroyed = True
        self.kv_buffer = None
