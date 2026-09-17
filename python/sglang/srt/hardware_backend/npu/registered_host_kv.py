"""Optional registered Host KV component adapter for Ascend NPU."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence
from uuid import uuid4

import torch


@dataclass
class RegisteredHostComponent:
    """One component region and its lifetime owner."""

    name: str
    region: object
    page_size: int


class NpuLayerwiseKvSlotPool:
    """Bounded device KV storage indexed by physical layer-slot ID."""

    def __init__(self, physical_slots: int, device: str = "npu"):
        if physical_slots <= 0:
            raise ValueError("physical_slots must be positive")
        self.physical_slots = physical_slots
        self.device = device
        self._buffers: Dict[str, torch.Tensor] = {}
        self._owners: Dict[int, object] = {}

    def allocate_component(
        self,
        name: str,
        per_layer_shape: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if name in self._buffers:
            raise ValueError(f"component {name} already exists")
        shape = (self.physical_slots, *tuple(per_layer_shape))
        buffer = torch.empty(shape, dtype=dtype, device=self.device)
        self._buffers[name] = buffer
        return buffer

    def component(self, name: str) -> torch.Tensor:
        try:
            return self._buffers[name]
        except KeyError as exc:
            raise KeyError(f"unknown device KV component {name}") from exc

    def slot_view(self, name: str, slot_id: int) -> torch.Tensor:
        if slot_id < 0 or slot_id >= self.physical_slots:
            raise IndexError(f"invalid physical slot {slot_id}")
        return self.component(name)[slot_id]

    def reserve_slot(self, slot_id: int, owner: object) -> None:
        """Reserve a physical slot using the coordinator generation token."""
        self._validate_slot(slot_id)
        if owner is None:
            raise ValueError("slot owner must not be None")
        current = self._owners.get(slot_id)
        if current is not None:
            raise RuntimeError(f"slot {slot_id} is already reserved")
        self._owners[slot_id] = owner

    def release_slot(self, slot_id: int, owner: object) -> None:
        self._validate_slot(slot_id)
        if self._owners.get(slot_id) != owner:
            raise RuntimeError("slot owner is stale or does not match")
        del self._owners[slot_id]

    def validate_write(
        self, name: str, slot_id: int, value: torch.Tensor, owner: object
    ) -> None:
        self._validate_slot(slot_id)
        if owner is None or self._owners.get(slot_id) != owner:
            raise RuntimeError("slot owner is stale or does not match")
        target = self.slot_view(name, slot_id)
        if (
            target.shape != value.shape
            or target.dtype != value.dtype
            or target.device != value.device
        ):
            raise ValueError(f"slot layout mismatch for {name}")

    def write_slot(
        self, name: str, slot_id: int, value: torch.Tensor, *, owner: object
    ) -> None:
        """Copy one layer component into a reserved physical slot."""
        self.validate_write(name, slot_id, value, owner)
        target = self.slot_view(name, slot_id)
        target.copy_(value)

    @property
    def nbytes(self) -> int:
        return sum(buffer.nbytes for buffer in self._buffers.values())

    def _validate_slot(self, slot_id: int) -> None:
        if slot_id < 0 or slot_id >= self.physical_slots:
            raise IndexError(f"invalid physical slot {slot_id}")


class NpuRegisteredHostKvPool:
    """Own registered Host regions for latent, RoPE, and Indexer components.

    The import of ``sgl_kernel_npu`` is delayed until construction, keeping
    normal SGLang startup and CPU unit tests independent of the optional NPU
    extension.  This class is an adapter only; scheduling and slot ownership
    remain in ``prefill_offload.py``.
    """

    def __init__(self, device_id: int = 0, *, shared_group=None):
        self.device_id = device_id
        self.shared_group = shared_group
        self._components: Dict[str, RegisteredHostComponent] = {}
        self._kernel_module: Optional[object] = None
        self._allocation_id = uuid4().hex
        self._closed = False

    def allocate_component(
        self,
        name: str,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        page_size: int,
    ) -> RegisteredHostComponent:
        if self._closed:
            raise RuntimeError("Host KV pool is closed")
        if name in self._components:
            raise ValueError(f"component {name} already exists")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        kernel = self._load_kernel_module()
        if self.shared_group is None:
            region = kernel.RegisteredHostRegion.allocate(
                shape,
                dtype,
                device_id=self.device_id,
                name=f"{self._allocation_id}-{name}",
            )
        else:
            region = self.shared_group.allocate(
                kernel.RegisteredHostRegion, name, shape, dtype, page_size
            )
        component = RegisteredHostComponent(name, region, page_size)
        self._components[name] = component
        return component

    def attach_component(
        self, name: str, handle, *, page_size: int
    ) -> RegisteredHostComponent:
        if self._closed:
            raise RuntimeError("Host KV pool is closed")
        if name in self._components:
            raise ValueError(f"component {name} already exists")
        if page_size <= 0 or not handle.shape or handle.shape[0] % page_size:
            raise ValueError("shared component geometry is not page aligned")
        region = self._load_kernel_module().RegisteredHostRegion.attach(
            handle, device_id=self.device_id, name=f"{self._allocation_id}-{name}"
        )
        component = RegisteredHostComponent(name, region, page_size)
        self._components[name] = component
        return component

    def component(self, name: str) -> RegisteredHostComponent:
        try:
            return self._components[name]
        except KeyError as exc:
            raise KeyError(f"unknown Host KV component {name}") from exc

    def copy_to_host(self, name: str, src, src_index, dst_index, valid_mask):
        component = self.component(name)
        if self.shared_group is None or self.shared_group.is_writer:
            self._load_kernel_module().copy_rows_to_registered_host(
                src,
                component.region,
                src_index,
                dst_index,
                valid_mask,
            )
        if self.shared_group is not None:
            self.shared_group.publish_write()
        return component.region.tensor

    def copy_from_host(self, name: str, dst, src_index, dst_index, valid_mask):
        component = self.component(name)
        return self._load_kernel_module().copy_rows_from_registered_host(
            component.region,
            dst,
            src_index,
            dst_index,
            valid_mask,
        )

    def close(self, *, drained: bool = False) -> None:
        if self._closed:
            return
        if not drained:
            raise RuntimeError("Host KV users must be drained before close")
        for name in list(self._components):
            self._components[name].region.close(drained=True)
            del self._components[name]
        self._closed = True

    def _load_kernel_module(self):
        if self._kernel_module is None:
            try:
                from sglang.srt.hardware_backend.npu import (
                    fsa_native as sparsity_driven_kv_offload,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "the standalone FSA registered-memory extension is required "
                    "for NPU Host KV offload"
                ) from exc
            self._kernel_module = sparsity_driven_kv_offload
        return self._kernel_module
