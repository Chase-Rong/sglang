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


class _MemFabricDomain:
    """Process-wide allocator lifetime shared by target and draft KV pools."""

    def __init__(self, device_id, reserve_bytes, shared_group, key):
        import memfabric_hybrid as fabric

        self.fabric = fabric
        self.key = key
        self.reserve_bytes = reserve_bytes
        self.claimed_bytes = 0
        self.references = 0
        self.next_namespace = 0
        operations = native_ops()
        if not all(
            hasattr(operations, name)
            for name in ("memfabric_region", "prefill_dma_copy")
        ):
            raise RuntimeError(
                "Rebuild the standalone FSA binding with MemFabric Prefill DMA support"
            )
        shared = shared_group is not None and shared_group.group.world_size > 1
        writer = not shared or shared_group.is_writer
        config = fabric.offload.OffloadConfig()
        config.device_id = device_id
        config.rank_id = shared_group.group.rank_in_group if shared else 0
        config.world_size = shared_group.group.world_size if shared else 1
        config.scene = fabric.offload.Scene.SHARED if shared else fabric.offload.Scene.LOCAL
        config.reserve_size = ((reserve_bytes + (1 << 30) - 1) >> 30) << 30
        config.alloc_size = config.reserve_size if writer else 0
        if shared:
            shared_group.validate_identity(
                ("memfabric", reserve_bytes, config.reserve_size)
            )
            ports = shared_group.group.all_gather_object(
                8500 + device_id // config.world_size
            )
            if len(set(ports)) != 1:
                raise ValueError("MemFabric TP devices must share its local bootstrap port")
        result = fabric.offload.initialize(config)
        if result:
            raise RuntimeError(
                f"MemFabric integrated Host KV initialization failed: {result}"
            )
        self.physical_bytes = config.alloc_size
        logger.info(
            "Integrated Host KV: MemFabric DMA, budget=%d physical=%d shared=%s",
            reserve_bytes,
            self.physical_bytes,
            shared,
        )


def _domain_key(device_id, shared_group):
    if shared_group is None or shared_group.group.world_size == 1:
        return (device_id, None)
    group = shared_group.group
    # Each SharedHostGroup wrapper has a fresh group_id. Identify the actual
    # TP collective domain using its underlying CPU process group instead.
    process_group = getattr(group, "cpu_group", None)
    return (
        device_id,
        id(process_group if process_group is not None else group),
        group.rank_in_group,
        group.world_size,
        shared_group.writer_rank,
    )


class MemFabricHostKvPool(NpuRegisteredHostKvPool):
    def __init__(self, device_id, *, required_bytes, reserve_bytes=None, shared_group=None):
        global _active_pool
        super().__init__(device_id, shared_group=shared_group)
        if required_bytes <= 0:
            raise ValueError("MemFabric Host KV size must be positive")
        reserve_bytes = required_bytes if reserve_bytes is None else reserve_bytes
        if reserve_bytes <= 0 or required_bytes > reserve_bytes:
            raise ValueError("MemFabric Host KV requirement exceeds the domain budget")
        self._allocations = []
        self._shared = shared_group is not None and shared_group.group.world_size > 1
        self._writer = not self._shared or shared_group.is_writer
        self._bytes_left = self._required_bytes = required_bytes
        if self._shared and shared_group.writer_rank != 0:
            raise ValueError("MemFabric shared Host KV requires writer rank zero")
        key = _domain_key(device_id, shared_group)
        with _pool_lock:
            domain = _active_pool
            domain_error = None
            if domain is not None:
                if domain.key != key:
                    domain_error = RuntimeError(
                        "MemFabric Host KV domain device or TP group differs"
                    )
                elif domain.reserve_bytes != reserve_bytes:
                    domain_error = ValueError(
                        "MemFabric Host KV pools must use the same domain budget"
                    )
                elif domain.claimed_bytes + required_bytes > domain.reserve_bytes:
                    domain_error = ValueError(
                        "Combined MemFabric Host KV pools exceed the domain budget: "
                        f"{domain.claimed_bytes + required_bytes} > {domain.reserve_bytes}"
                    )
            namespace = domain.next_namespace if domain is not None else 0
            if self._shared:
                # A rank must not reject its local budget while peers enter the
                # next collective. Compare the claims and failure state first.
                shared_group.validate_identity(
                    (
                        "memfabric-pool",
                        namespace,
                        required_bytes,
                        reserve_bytes,
                        domain.claimed_bytes if domain is not None else 0,
                        domain.references if domain is not None else 0,
                        str(domain_error) if domain_error is not None else None,
                    )
                )
            if domain_error is not None:
                raise domain_error
            if domain is None:
                domain = _MemFabricDomain(device_id, reserve_bytes, shared_group, key)
                _active_pool = domain
            self._domain = domain
            self._namespace = f"pool-{namespace}"
            self.fabric = domain.fabric
            self.physical_bytes = domain.physical_bytes
            domain.next_namespace += 1
            domain.references += 1
            domain.claimed_bytes += required_bytes

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
                (self._namespace, name, tuple(shape), str(dtype), page_size)
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
                tensor, pointer, device_pointer, self.device_id, f"{self._namespace}/{name}"
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
        if self._shared:
            # Geometry validation can mark the group failed after partial
            # allocation. Cleanup must still run, in the same order on every rank.
            identity = ("memfabric-close", self._namespace)
            peers = self.shared_group.group.all_gather_object(identity)
            if any(peer != identity for peer in peers):
                raise ValueError("MemFabric Host KV ranks disagree on pool close order")
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
        with _pool_lock:
            domain = self._domain
            domain.claimed_bytes -= self._required_bytes
            domain.references -= 1
            if domain.references == 0:
                self.fabric.offload.uninitialize()
                _active_pool = None
