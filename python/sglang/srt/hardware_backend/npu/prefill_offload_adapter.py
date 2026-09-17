"""Concrete Host-registration callbacks for the Prefill layer protocol."""

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from sglang.srt.hardware_backend.npu.prefill_offload import (
    HostPageAllocation,
    KvComponent,
    LayerWork,
    PdComponentDescriptor,
    PdSourceKind,
    SlotBusyError,
)
from sglang.srt.hardware_backend.npu.registered_host_kv import (
    NpuLayerwiseKvSlotPool,
    NpuRegisteredHostKvPool,
)


@dataclass
class LayerKvBinding:
    layer_id: int
    latent_k: torch.Tensor
    rope_k: torch.Tensor
    indexer_k: Optional[torch.Tensor]
    token_indices: torch.Tensor
    device_rows: torch.Tensor
    host_rows: torch.Tensor
    mapped_rows: torch.Tensor
    source_page_ids: Optional[Tuple[int, ...]]
    destination_pages: Optional[Dict[KvComponent, Tuple[int, ...]]]


class NpuPrefillOffloadAdapter:
    """Connect layer lifecycle callbacks to registered Host copies."""

    def __init__(
        self,
        host_pool: NpuRegisteredHostKvPool,
        *,
        page_size: int,
        publish: Optional[Callable] = None,
        slot_pool: Optional[NpuLayerwiseKvSlotPool] = None,
        record_event: Optional[Callable] = None,
    ):
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.host_pool = host_pool
        self.page_size = page_size
        self.publish = publish
        self.slot_pool = slot_pool
        self.record_event = record_event or self._record_npu_event
        self.bindings: Dict[tuple, LayerKvBinding] = {}
        self._layouts: Dict[tuple, tuple] = {}
        self._saved_prefixes: Dict[tuple, int] = {}
        self._save_events: Dict[object, object] = {}
        self._attention_events: Dict[object, object] = {}
        self._binding_events: Dict[object, object] = {}
        self._restore_events: Dict[object, object] = {}
        self._failed_saves = set()

    def register_layer_components(
        self,
        logical_layer: int,
        *,
        latent_shape,
        rope_shape,
        indexer_shape=None,
        dtype=torch.bfloat16,
    ) -> None:
        if logical_layer < 0:
            raise ValueError("logical_layer must be non-negative")
        components = [
            (KvComponent.LATENT_K, latent_shape),
            (KvComponent.ROPE_K, rope_shape),
        ]
        if indexer_shape is not None:
            components.append((KvComponent.INDEXER_K, indexer_shape))
        components = [
            (component, shape) for component, shape in components if shape[-1] != 0
        ]
        for component, shape in components:
            if len(shape) != 2 or min(shape) <= 0 or shape[0] % self.page_size:
                raise ValueError("Host layout must be [page_count * page_size, dim]")
            if shape[0] != latent_shape[0]:
                raise ValueError("Host components must use the same page capacity")
            if (logical_layer, component) in self._layouts:
                raise ValueError("Host layer component already registered")
        for component, shape in components:
            self.host_pool.allocate_component(
                self._component_name(logical_layer, component),
                shape,
                dtype,
                page_size=self.page_size,
            )
            self._layouts[logical_layer, component] = (tuple(shape), dtype)

    def bind_layer(
        self,
        work: LayerWork,
        latent_k: torch.Tensor,
        rope_k: torch.Tensor,
        indexer_k: Optional[torch.Tensor] = None,
        *,
        token_indices: Optional[torch.Tensor] = None,
        device_row_indices: Optional[torch.Tensor] = None,
        source_page_ids: Optional[Tuple[int, ...]] = None,
        destination_pages: Optional[Dict[KvComponent, Tuple[int, ...]]] = None,
    ) -> LayerKvBinding:
        key = (work.token.request_id, work.token.generation, work.token.logical_layer)
        if key in self.bindings:
            raise ValueError("layer binding already exists")
        shared_group = getattr(self.host_pool, "shared_group", None)
        if shared_group is not None:
            shared_group.validate_identity(
                (
                    key,
                    work.valid_range.start_token,
                    work.valid_range.end_token,
                    tuple(work.host_allocation.page_ids),
                    source_page_ids,
                )
            )
        tensors = [(KvComponent.LATENT_K, latent_k), (KvComponent.ROPE_K, rope_k)]
        if indexer_k is not None:
            tensors.append((KvComponent.INDEXER_K, indexer_k))
        tensors = [
            (component, tensor)
            for component, tensor in tensors
            if tensor.shape[-1] != 0
        ]
        row_views = {}
        for component, tensor in tensors:
            rows = self._token_rows(tensor)
            shape, dtype = self._layouts[work.token.logical_layer, component]
            if rows.shape[-1] != shape[-1] or rows.dtype != dtype:
                raise ValueError("bound component does not match its Host layout")
            if rows.device != latent_k.device:
                raise ValueError("bound components must use the same device")
            if max(work.host_allocation.page_ids) >= shape[0] // self.page_size:
                raise ValueError("Host page allocation exceeds component capacity")
            row_views[component] = rows
        latent_rows = row_views[KvComponent.LATENT_K]
        if any(rows.shape[0] != latent_rows.shape[0] for rows in row_views.values()):
            raise ValueError("bound components must have matching token-row capacity")
        if token_indices is None:
            token_indices = torch.arange(
                latent_rows.shape[0], device=latent_k.device, dtype=torch.int64
            )
        if token_indices.ndim != 1 or token_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("token_indices must be a vector of integer token IDs")
        token_indices = token_indices.to(device=latent_k.device, dtype=torch.int64)
        if device_row_indices is None:
            if token_indices.numel() > latent_rows.shape[0]:
                raise ValueError("token mapping exceeds the bound row capacity")
            device_row_indices = torch.arange(
                token_indices.numel(), device=latent_k.device, dtype=torch.int64
            )
        if (
            device_row_indices.shape != token_indices.shape
            or device_row_indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("device row mapping must match logical token IDs")
        device_rows = device_row_indices.to(
            device=latent_k.device, dtype=torch.int64
        ).contiguous()
        allocation = work.host_allocation
        if allocation.page_size != self.page_size:
            raise ValueError("Host allocation page size differs from the adapter")
        pages = torch.tensor(
            allocation.page_ids, device=latent_k.device, dtype=torch.int64
        )
        valid = (token_indices >= 0) & (token_indices < pages.numel() * self.page_size)
        valid = valid & (device_rows >= 0) & (device_rows < latent_rows.shape[0])
        page_offsets = (token_indices // self.page_size).clamp(0, pages.numel() - 1)
        host_rows = (
            pages[page_offsets] * self.page_size + token_indices % self.page_size
        )
        if self.slot_pool is not None:
            self.slot_pool.reserve_slot(work.token.slot_id, work.token)
            try:
                for component, tensor in tensors:
                    self.slot_pool.validate_write(
                        component.value, work.token.slot_id, tensor, work.token
                    )
            except Exception:
                self.slot_pool.release_slot(work.token.slot_id, work.token)
                raise
            latent_k = self.slot_pool.slot_view("latent_k", work.token.slot_id)
            rope_k = self.slot_pool.slot_view("rope_k", work.token.slot_id)
            indexer_k = (
                None
                if indexer_k is None
                else self.slot_pool.slot_view("indexer_k", work.token.slot_id)
            )
        binding = LayerKvBinding(
            work.token.logical_layer,
            latent_k,
            rope_k,
            indexer_k,
            token_indices,
            device_rows,
            host_rows.contiguous(),
            valid,
            source_page_ids,
            destination_pages,
        )
        self.bindings[key] = binding
        if self.slot_pool is not None:
            try:
                for component, tensor in tensors:
                    self.slot_pool.write_slot(
                        component.value, work.token.slot_id, tensor, owner=work.token
                    )
            except Exception:
                self._binding_events[work.token] = self.record_event()
                raise
        return binding

    def validate_layer_release(self, work: LayerWork) -> None:
        self._binding(work)
        for events in (
            self._save_events,
            self._attention_events,
            self._binding_events,
            self._restore_events,
        ):
            event = events.get(work.token)
            if event is not None and not event.query():
                raise SlotBusyError("bound layer still has pending local device work")

    def release_layer(self, work: LayerWork) -> None:
        """Release the bounded source slot after coordinator completion."""
        key = (work.token.request_id, work.token.generation, work.token.logical_layer)
        self.validate_layer_release(work)
        if self.slot_pool is not None:
            self.slot_pool.release_slot(work.token.slot_id, work.token)
        del self.bindings[key]
        self._save_events.pop(work.token, None)
        self._attention_events.pop(work.token, None)
        self._binding_events.pop(work.token, None)
        self._restore_events.pop(work.token, None)
        self._failed_saves.discard(work.token)

    def save_layer(self, work: LayerWork) -> None:
        binding = self._binding(work)
        if work.token in self._save_events:
            raise RuntimeError("Host save already enqueued")
        try:
            for component, tensor in self._components(binding):
                if tensor is None:
                    continue
                valid = binding.mapped_rows & self._valid_token_mask(
                    work, binding.token_indices
                )
                self.host_pool.copy_to_host(
                    self._component_name(work.token.logical_layer, component),
                    self._token_rows(tensor),
                    binding.device_rows,
                    binding.host_rows,
                    valid.contiguous(),
                )
        except Exception:
            self._failed_saves.add(work.token)
            raise
        finally:
            self._save_events[work.token] = self.record_event()

    def restore_layer(self, work: LayerWork) -> None:
        binding = self._binding(work)
        prefix_key = self._prefix_key(work)
        if self._saved_prefixes.get(prefix_key, 0) < work.valid_range.start_token:
            raise RuntimeError("historical KV has not completed its Host save")
        if work.valid_range.start_token == 0:
            return
        try:
            for component, tensor in self._components(binding):
                if tensor is None:
                    continue
                valid = binding.mapped_rows & (
                    binding.token_indices < work.valid_range.start_token
                )
                self.host_pool.copy_from_host(
                    self._component_name(work.token.logical_layer, component),
                    self._token_rows(tensor),
                    binding.host_rows,
                    binding.device_rows,
                    valid.contiguous(),
                )
        finally:
            self._restore_events[work.token] = self.record_event()

    def complete_host_save(self, work: LayerWork) -> None:
        self._binding(work)
        if work.token in self._failed_saves:
            raise RuntimeError("a failed Host save cannot publish a valid prefix")
        event = self._save_events.get(work.token)
        if event is None:
            raise RuntimeError("Host save has not been enqueued")
        if not event.query():
            raise SlotBusyError("Host save DMA has not completed")
        key = self._prefix_key(work)
        prefix = self._saved_prefixes.get(key, 0)
        if work.valid_range.start_token > prefix:
            raise RuntimeError("cannot publish a Host prefix with a gap")
        self._saved_prefixes[key] = max(prefix, work.valid_range.end_token)

    def record_attention_done(self, work: LayerWork) -> None:
        self._binding(work)
        if work.token in self._attention_events:
            raise RuntimeError("attention completion already recorded")
        self._attention_events[work.token] = self.record_event()

    def complete_attention(self, work: LayerWork) -> None:
        self._binding(work)
        event = self._attention_events.get(work.token)
        if event is None:
            raise RuntimeError("attention completion has not been recorded")
        if not event.query():
            raise SlotBusyError("attention device work has not completed")

    def wait_local_completion(self, work: LayerWork, *, deadline: float) -> None:
        self._binding(work)
        for events in (
            self._save_events,
            self._attention_events,
            self._binding_events,
            self._restore_events,
        ):
            event = events.get(work.token)
            if event is None:
                continue
            while not event.query():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Prefill local device completion timed out")
                time.sleep(0.001)

    def binding(self, work: LayerWork) -> LayerKvBinding:
        return self._binding(work)

    def source_ready_event(self, work: LayerWork):
        self._binding(work)
        return self._save_events[work.token]

    @staticmethod
    def _record_npu_event():
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        return event

    def release_request(self, allocation: HostPageAllocation) -> None:
        if any(key[0] == allocation.request_id for key in self.bindings):
            raise RuntimeError("request still owns bound layer tensors")
        for key in list(self._saved_prefixes):
            if key[:2] == (allocation.request_id, allocation.generation):
                del self._saved_prefixes[key]

    def descriptors(self, work: LayerWork) -> list[PdComponentDescriptor]:
        binding = self._binding(work)
        if binding.source_page_ids is None or binding.destination_pages is None:
            raise RuntimeError("PD requires explicit source and peer destination pages")
        records = []
        for component, tensor in self._components(binding):
            if tensor is None:
                continue
            source_pages = binding.source_page_ids
            destination_pages = binding.destination_pages[component]
            end_page = (
                work.valid_range.end_token + self.page_size - 1
            ) // self.page_size
            if len(source_pages) < end_page or len(destination_pages) < end_page:
                raise ValueError(
                    "PD page mappings do not cover the published token range"
                )
            records.append(
                PdComponentDescriptor(
                    request_id=work.token.request_id,
                    logical_layer=work.token.logical_layer,
                    generation=work.token.generation,
                    component=component,
                    source_kind=PdSourceKind.NPU_REGISTERED,
                    source_slot=work.token.slot_id,
                    source_page_ids=source_pages,
                    destination_page_ids=destination_pages,
                    valid_range=work.valid_range,
                    bytes_per_page=self._token_rows(tensor)[0].nbytes * self.page_size,
                )
            )
        return records

    @staticmethod
    def _component_name(logical_layer, component):
        return f"layer:{logical_layer}:{component.value}"

    @staticmethod
    def _prefix_key(work):
        return (
            work.token.request_id,
            work.host_allocation.generation,
            work.token.logical_layer,
        )

    @staticmethod
    def _token_rows(tensor):
        if tensor.ndim not in (2, 3, 4) or not tensor.is_contiguous():
            raise ValueError("KV binding must be contiguous token rows or paged KV")
        if tensor.ndim == 4 and tensor.shape[-2] != 1:
            raise ValueError("paged KV binding requires one KV head")
        return tensor.view(-1, tensor.shape[-1])

    @staticmethod
    def _components(binding):
        return tuple(
            (component, tensor)
            for component, tensor in (
                (KvComponent.LATENT_K, binding.latent_k),
                (KvComponent.ROPE_K, binding.rope_k),
                (KvComponent.INDEXER_K, binding.indexer_k),
            )
            if tensor is not None and tensor.shape[-1] != 0
        )

    def _binding(self, work: LayerWork) -> LayerKvBinding:
        key = (work.token.request_id, work.token.generation, work.token.logical_layer)
        try:
            return self.bindings[key]
        except KeyError as exc:
            raise RuntimeError("layer tensors were not bound") from exc

    @staticmethod
    def _valid_token_mask(work: LayerWork, indices: torch.Tensor) -> torch.Tensor:
        return (indices >= work.valid_range.start_token) & (
            indices < work.valid_range.end_token
        )

    def save_and_describe(self, work: LayerWork) -> list[PdComponentDescriptor]:
        """Save the bound layer and return descriptors for PD publication."""
        self.save_layer(work)
        return self.descriptors(work)
