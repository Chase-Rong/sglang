"""HiCache page ownership over an existing MemFabric Decode backing.

Bind once, before admitting requests. The caller coordinates page ownership,
TP decisions and reader/writer drain before clear/reset or backing shutdown.
This adapter never allocates or closes physical regions. Layer views on TP
peers are device-address metadata, not CPU-readable storage.
"""

from threading import RLock

import torch
from sglang.srt.hardware_backend.npu.memfabric_host_kv import MemFabricHostKvPool
from sglang.srt.mem_cache.pool_host.base import HostKVCache, synchronized


class MemFabricHiCachePool(HostKVCache):
    """Borrow layer-contiguous latent/RoPE regions and reuse HiCache alloc/free.

    ``size`` includes the reserved first page; ``usable_size`` excludes it.
    Unlike HostKVCache.__init__, construction uses the exact existing capacity
    without rounding, checking free RAM again, or allocating another payload.
    The backing owner must outlive this adapter and every exported view.
    """

    def __init__(self, backing, device_pool, *, pool_label="kv"):
        if not isinstance(backing.pool, MemFabricHostKvPool):
            raise TypeError("HiCache requires a MemFabric DecodeHostKvBacking")
        if backing.pool._closed:
            raise RuntimeError("MemFabric backing is closed")
        if backing.page_size <= 0 or backing.pages < 2 or not backing.regions:
            raise ValueError("Host backing needs page0 and at least one usable page")
        if getattr(device_pool, "layer_shard_enabled", False):
            raise NotImplementedError(
                "MemFabric HiCache layer sharding is not connected"
            )
        if getattr(device_pool, "dcp_size", 1) != 1:
            raise NotImplementedError("MemFabric HiCache DCP is not connected")
        self.device_pool = device_pool
        self.backing = backing
        self.binding = None
        self.pool_label = pool_label
        self.page_size = backing.page_size
        self.page_num = backing.pages
        self.size = self.page_num * self.page_size
        self.usable_size = self.size - self.page_size
        self.dtype = backing.dtype
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.layer_num = self.target_layer_num = len(backing.regions)
        if (
            device_pool.store_dtype != self.dtype
            or device_pool.page_size != self.page_size
            or device_pool.layer_num != self.layer_num
        ):
            raise ValueError("Device pool and Host backing geometry differ")
        self.kv_lora_rank = device_pool.kv_lora_rank
        self.qk_rope_head_dim = device_pool.qk_rope_head_dim
        if self.kv_lora_rank <= 0 or self.qk_rope_head_dim != backing.rope_dim:
            raise ValueError("Device pool and Host backing component widths differ")
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        self.regions = tuple(backing.regions)
        for regions in self.regions:
            for region, width in zip(
                regions, (self.kv_lora_rank, self.qk_rope_head_dim), strict=True
            ):
                if width == 0 and region is None:
                    continue
                if region is None:
                    raise ValueError("Host backing is missing a component")
                region._require_live()
                tensor = region.tensor
                if (
                    tuple(tensor.shape) != (self.size, width)
                    or tensor.dtype != self.dtype
                    or tensor.device.type != "cpu"
                    or not tensor.is_contiguous()
                ):
                    raise ValueError("Host backing component geometry differs")
        self.layout = "layer_first"
        self.device = "cpu"
        self.pin_memory = False
        self.allocator = backing.pool
        self.fd = None
        self.dcp_size = 1
        self.dcp_rank = 0
        self.can_use_jit = False
        self.can_use_write_back_jit = False
        self.size_per_token = self.get_size_per_token()
        self._destroyed = False
        self.lock = RLock()
        self.kv_buffer = self.init_kv_buffer()
        self.k_buffer = tuple(views[0] for views in self.kv_buffer)
        self.v_buffer = tuple(views[1] for views in self.kv_buffer)
        self.clear()

    def _require_live(self):
        if self._destroyed or self.backing.pool._closed:
            raise RuntimeError("MemFabric Host pool or backing is closed")

    def _require_cpu_access(self):
        self._require_live()
        if not self.backing.pool._writer:
            raise RuntimeError("TP peer shared GVA must not be dereferenced on CPU")
        for regions in self.regions:
            for region in regions:
                if region is not None:
                    region._require_live()

    def get_size_per_token(self):
        return self.layer_num * self.kv_cache_dim * self.dtype.itemsize

    def init_kv_buffer(self):
        self._require_live()
        return tuple(self.backing.layer_views(layer) for layer in range(self.layer_num))

    def layer_views(self, layer_id):
        """Return borrowed FSA views using a backing-local layer index."""
        self._require_live()
        if not 0 <= layer_id < self.layer_num:
            raise IndexError("Host layer index is out of range")
        for region in self.regions[layer_id]:
            if region is not None:
                region._require_live()
        return self.kv_buffer[layer_id]

    def clear(self):
        """Reset logical ownership after callers have drained all users."""
        self._require_live()
        if self.binding is not None:
            self.binding.clear()
        with self.lock:
            super().clear()
            super().alloc(self.page_size)

    def reset(self):
        self.clear()

    @synchronized
    def available_size(self):
        self._require_live()
        return super().available_size()

    @synchronized
    def alloc(self, need_size):
        self._require_live()
        if type(need_size) is not int or need_size < 0 or need_size % self.page_size:
            raise ValueError("Allocation size must be a nonnegative whole-page count")
        return super().alloc(need_size)

    def free(self, indices):
        self._require_live()
        if self.binding is not None:
            return self.binding.release_cached_rows(indices)
        return self.free_unreferenced(indices)

    def acquire_device_pages(self, device_indices):
        self._require_binding()
        return self.binding.retain_device_rows(device_indices)

    @synchronized
    def free_unreferenced(self, indices):
        """Recycle pages after the binding releases their last reference."""
        self._require_live()
        if (
            not isinstance(indices, torch.Tensor)
            or indices.ndim != 1
            or indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Free requires a one-dimensional integer slot tensor")
        slots = indices.to(device="cpu", dtype=torch.int64).clone()
        if slots.numel() == 0:
            return 0
        if slots.numel() % self.page_size:
            raise ValueError("Free must return complete pages")
        if torch.any(slots < self.page_size) or torch.any(slots >= self.size):
            raise ValueError("Free contains reserved page0 or out-of-range slots")
        pages = slots.view(-1, self.page_size)
        starts = pages[:, 0]
        if (
            torch.any(starts % self.page_size)
            or not torch.equal(pages, starts[:, None] + torch.arange(self.page_size))
            or torch.unique(starts).numel() != starts.numel()
        ):
            raise ValueError("Free requires distinct, aligned, complete pages")
        if not self.slot_used[slots].all():
            raise ValueError("Double-free or unallocated Host slots")
        return super().free(slots)

    def _validate_page(self, index):
        if isinstance(index, torch.Tensor):
            if (
                index.device.type != "cpu"
                or index.ndim != 0
                or index.dtype
                not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            ):
                raise ValueError("Page index tensor must be a scalar CPU integer")
            index = index.item()
        if (
            type(index) is not int
            or index < self.page_size
            or index % self.page_size
            or index + self.page_size > self.size
        ):
            raise ValueError("Page index must be aligned, in range and outside page0")
        if not self.slot_used[index : index + self.page_size].all():
            raise ValueError("Host page is not allocated")
        return index

    def get_data_page(self, index, flat=True):
        """Pack one page as [layer, token, head=1, latent+rope], copying data."""
        self._require_cpu_access()
        index = self._validate_page(index)
        page = torch.stack(
            [
                torch.cat(
                    [
                        region.tensor[index : index + self.page_size]
                        for region in regions
                        if region is not None
                    ],
                    dim=-1,
                )
                for regions in self.regions
            ]
        ).unsqueeze(2)
        return page.flatten() if flat else page

    def get_dummy_flat_data_page(self):
        self._require_live()
        return torch.zeros(
            self.layer_num * self.page_size * self.kv_cache_dim,
            dtype=self.dtype,
            device="cpu",
        )

    def set_from_flat_data_page(self, index, data_page):
        self._require_cpu_access()
        index = self._validate_page(index)
        if (
            data_page.device.type != "cpu"
            or data_page.dtype != self.dtype
            or data_page.numel() != self.layer_num * self.page_size * self.kv_cache_dim
        ):
            raise ValueError(
                "Flat page dtype, device or size differs from Host backing"
            )
        page = data_page.reshape(self.layer_num, self.page_size, self.kv_cache_dim)
        for layer, regions in enumerate(self.regions):
            offset = 0
            for region in regions:
                if region is not None:
                    width = region.tensor.shape[-1]
                    region.tensor[index : index + self.page_size].copy_(
                        page[layer, :, offset : offset + width]
                    )
                    offset += width

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
        self._require_binding()
        if is_draft or device_pool is not self.device_pool:
            raise NotImplementedError("This Host pool only binds its main KV pool")
        if not 0 <= layer_id < self.layer_num:
            raise IndexError("HiCache load layer index is out of range")
        if layer_id == 0:
            self.binding.restore_mapping(host_indices, device_indices)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        self._require_binding()
        if device_pool is not self.device_pool:
            raise NotImplementedError("This Host pool only binds its main KV pool")
        ready_event = self.binding.ready_event
        if ready_event is None:
            raise RuntimeError("Main KV backup has no ready event")
        torch.npu.current_stream().wait_event(ready_event)

    def _require_binding(self):
        self._require_live()
        if self.binding is None:
            raise RuntimeError(
                "Bind the HiCache offload runtime before transferring pages"
            )

    @synchronized
    def destroy(self):
        """Detach views only; backing owner retains all physical allocations."""
        if self._destroyed:
            return
        self._destroyed = True
        self.kv_buffer = None
        self.k_buffer = self.v_buffer = ()
        self.regions = ()
        self.free_slots = torch.empty(0, dtype=torch.int64)
        self.release_slots = []
        self.num_release_slots = 0

    def close(self):
        self.destroy()
