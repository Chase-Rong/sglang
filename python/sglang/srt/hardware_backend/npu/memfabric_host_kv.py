"""One MemFabric allocation domain for integrated Prefill DMA and Decode FSA.

Shared peers have device mappings only. Their CPU tensors are address/layout
metadata for native operators and must never be dereferenced by CPU code.
"""

import logging
from math import prod
from threading import Lock

import torch

from sglang.srt.hardware_backend.npu.fsa_native import RegisteredHostRegion, native_ops
from sglang.srt.hardware_backend.npu.prefill_dma import prefill_row_plans
from sglang.srt.hardware_backend.npu.registered_host_kv import (
    NpuRegisteredHostKvPool,
    RegisteredHostComponent,
)

logger = logging.getLogger(__name__)
_pool_lock = Lock()
_active_pool = None


class MemFabricRegion(RegisteredHostRegion):
    def export_handle(self):
        raise RuntimeError("MemFabric shared GVA is not a SysV or PD transport handle")

    def close(self, *, drained=False):
        if self.closed:
            return
        if not drained:
            raise RuntimeError("MemFabric readers and graphs must drain before close")
        native_ops().unregister_memfabric_region(self.tensor, self.device_id)
        self.closed = True
        self.host_ptr = self.device_ptr = 0
        self.tensor = torch.empty(0, dtype=self.tensor.dtype)


class MemFabricHostKvPool(NpuRegisteredHostKvPool):
    def __init__(self, device_id, *, required_bytes, shared_group=None):
        global _active_pool
        super().__init__(device_id, shared_group=shared_group)
        if required_bytes <= 0:
            raise ValueError("MemFabric Host KV size must be positive")
        self._allocations = []
        self._initialized = False
        self._shared = shared_group is not None and shared_group.group.world_size > 1
        self._writer = not self._shared or shared_group.is_writer
        self._bytes_left = required_bytes
        if self._shared and shared_group.writer_rank != 0:
            raise ValueError("MemFabric shared Host KV requires writer rank zero")
        with _pool_lock:
            if _active_pool is not None:
                raise RuntimeError(
                    "MemFabric supports one integrated Host KV pool per process"
                )
            _active_pool = self
        try:
            import memfabric_hybrid as fabric

            self.fabric = fabric
            operations = native_ops()
            if not all(
                hasattr(operations, name)
                for name in ("memfabric_region", "prefill_dma_copy")
            ):
                raise RuntimeError(
                    "Rebuild the standalone FSA binding with MemFabric Prefill DMA support"
                )
            config = fabric.offload.OffloadConfig()
            config.device_id = device_id
            config.rank_id = shared_group.group.rank_in_group if self._shared else 0
            config.world_size = shared_group.group.world_size if self._shared else 1
            config.scene = (
                fabric.offload.Scene.SHARED
                if self._shared
                else fabric.offload.Scene.LOCAL
            )
            config.reserve_size = ((required_bytes + (1 << 30) - 1) >> 30) << 30
            config.alloc_size = config.reserve_size if self._writer else 0
            if self._shared:
                shared_group.validate_identity(("memfabric", config.reserve_size))
                ports = shared_group.group.all_gather_object(
                    8500 + device_id // config.world_size
                )
                if len(set(ports)) != 1:
                    raise ValueError(
                        "MemFabric TP devices must share its local bootstrap port"
                    )
            result = fabric.offload.initialize(config)
            if result:
                raise RuntimeError(
                    f"MemFabric integrated Host KV initialization failed: {result}"
                )
            self._initialized = True
            self.physical_bytes = config.alloc_size
            logger.info(
                "Integrated Host KV: MemFabric DMA, logical=%d physical=%d shared=%s",
                required_bytes,
                self.physical_bytes,
                self._shared,
            )
        except BaseException:
            with _pool_lock:
                _active_pool = None
            raise

    def allocate_component(self, name, shape, dtype, *, page_size):
        if self._closed or name in self._components:
            raise RuntimeError("Host KV pool is closed or component already exists")
        size = prod(shape) * torch.empty((), dtype=dtype).element_size()
        if (
            page_size <= 0
            or not shape
            or any(dim <= 0 for dim in shape)
            or shape[0] % page_size
        ):
            raise ValueError("MemFabric component geometry is not page aligned")
        if size > self._bytes_left:
            raise ValueError("MemFabric component exceeds the reserved Host KV size")
        if self._shared:
            self.shared_group.validate_identity(
                (name, tuple(shape), str(dtype), page_size)
            )
        pointer, error = 0, None
        if self._writer:
            try:
                pointer = self.fabric.offload.malloc(size)
                if not pointer:
                    raise RuntimeError(
                        f"MemFabric Host KV allocation failed for {size} bytes"
                    )
            except Exception as exc:
                error = str(exc)
        if self._shared:
            pointer, error = self.shared_group.group.broadcast_object(
                (pointer, error), src=0
            )
        if error is not None:
            raise RuntimeError(error)
        region = None
        try:
            tensor, device_pointer = native_ops().memfabric_region(
                pointer,
                tuple(shape),
                torch.empty(0, dtype=dtype),
                self.device_id,
                self._shared,
            )
            region = MemFabricRegion(
                tensor, pointer, device_pointer, self.device_id, name
            )
        except Exception as exc:
            error = str(exc)
        errors = (
            self.shared_group.group.all_gather_object(error)
            if self._shared
            else [error]
        )
        if any(message is not None for message in errors):
            if region is not None:
                region.close(drained=True)
            if self._shared:
                self.shared_group.group.all_gather_object("registration rolled back")
            if self._writer:
                self.fabric.offload.free(pointer)
            raise RuntimeError(f"MemFabric KV registration failed: {errors}")
        self._allocations.append(pointer)
        self._bytes_left -= size
        component = RegisteredHostComponent(name, region, page_size)
        self._components[name] = component
        return component

    def attach_component(self, name, handle, *, page_size):
        raise RuntimeError("MemFabric integrated Host KV uses collective allocation")

    @staticmethod
    def plan_prefill_rows(prefix_rows, new_rows, capacity):
        return prefill_row_plans(prefix_rows, new_rows, capacity)

    def copy_prefill_from_host(self, name, destination, plan):
        region = self.component(name).region
        region._require_live()
        native_ops().prefill_dma_copy(
            destination, region.tensor, region.device_ptr, plan, False
        )

    def copy_prefill_to_host(self, name, source, plan):
        region = self.component(name).region
        region._require_live()
        if self._writer:
            native_ops().prefill_dma_copy(
                source, region.tensor, region.device_ptr, plan, True
            )

    def close(self, *, drained=False):
        global _active_pool
        if self._closed:
            return
        if not drained:
            raise RuntimeError("MemFabric Host KV users must drain before close")
        super().close(drained=True)
        if self._shared:
            self.shared_group.group.all_gather_object(
                "MemFabric readers drained and unregistered"
            )
        if self._writer:
            for pointer in reversed(self._allocations):
                self.fabric.offload.free(pointer)
        self._allocations.clear()
        if self._shared:
            self.shared_group.group.all_gather_object("MemFabric owner freed")
        if self._initialized:
            self.fabric.offload.uninitialize()
            self._initialized = False
        with _pool_lock:
            if _active_pool is self:
                _active_pool = None
