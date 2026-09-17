"""Prefill layer-slot ownership for bounded NPU KV residency.

This module deliberately contains no device or transport calls.  It is the
small ownership protocol shared by the future Host KV pool and MemFabric
adapter.  Keeping it independent makes slot reuse testable without an NPU.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple


class SlotBusyError(RuntimeError):
    """Raised when a physical slot still has an active owner."""


class StaleGenerationError(RuntimeError):
    """Raised when a completion refers to an older slot ownership token."""


class SlotFailedError(RuntimeError):
    """Raised when a failed slot is used before it is explicitly released."""


@dataclass(frozen=True)
class PrefillKvMemoryBudget:
    logical_layers: int
    physical_slots: int
    bytes_per_token_per_layer: int
    host_bytes: int
    resident_index_bytes_per_token: int = 0

    def __post_init__(self):
        if self.resident_index_bytes_per_token < 0:
            raise ValueError("resident index bytes must be nonnegative")
        if not 0 < self.physical_slots <= self.logical_layers:
            raise ValueError("physical slots must fit logical layer count")
        if self.bytes_per_token_per_layer <= 0 or self.host_bytes <= 0:
            raise ValueError(
                "Prefill offload requires positive row size and Host budget"
            )

    @property
    def device_bytes_per_token(self):
        return (
            self.physical_slots * self.bytes_per_token_per_layer
            + self.resident_index_bytes_per_token
        )

    @property
    def host_bytes_per_token(self):
        return self.logical_layers * self.bytes_per_token_per_layer

    def max_tokens(self, device_bytes, page_size):
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        rows = min(
            device_bytes // self.device_bytes_per_token,
            self.host_bytes // self.host_bytes_per_token,
        )
        return max(0, (rows // page_size - 1) * page_size)

    def validate_host_capacity(self, token_count, page_size):
        if token_count < 0 or page_size <= 0:
            raise ValueError("invalid Prefill capacity")
        required = (
            (token_count // page_size + 1) * page_size * self.host_bytes_per_token
        )
        if required > self.host_bytes:
            raise MemoryError(
                f"Prefill Host KV needs {required} bytes; budget is {self.host_bytes}"
            )


class KvComponent(str, Enum):
    """KV components with independent PD destination layouts."""

    LATENT_K = "latent_k"
    ROPE_K = "rope_k"
    INDEXER_K = "indexer_k"


class PdSourceKind(str, Enum):
    """Address-space kind used by the PD data plane."""

    NPU_REGISTERED = "npu_registered"
    HOST_REGISTERED = "host_registered"


@dataclass(frozen=True)
class LayerSlotToken:
    """Stable identity for one logical layer occupying one physical slot."""

    slot_id: int
    request_id: str
    logical_layer: int
    generation: int


@dataclass
class _SlotOwner:
    token: LayerSlotToken
    host_saved: bool = False
    attention_done: bool = False
    pending_pd_reads: int = 0
    failed: bool = False

    @property
    def reusable(self) -> bool:
        return (
            not self.failed
            and self.host_saved
            and self.attention_done
            and self.pending_pd_reads == 0
        )


class PrefillLayerSlotManager:
    """Maps logical layers onto a bounded set of reusable physical slots.

    The manager is intentionally non-blocking.  A caller that receives
    :class:`SlotBusyError` must wait on the device, Host-copy, or PD completion
    primitive and retry.  This prevents a queued notification from being
    mistaken for transfer completion.
    """

    def __init__(self, logical_layers: int, physical_slots: int):
        if logical_layers <= 0:
            raise ValueError("logical_layers must be positive")
        if physical_slots <= 0:
            raise ValueError("physical_slots must be positive")
        self.logical_layers = logical_layers
        self.physical_slots = physical_slots
        self._owners: Dict[int, _SlotOwner] = {}
        self._generations = [0] * physical_slots

    def preferred_slot(self, logical_layer: int) -> int:
        self._validate_layer(logical_layer)
        return logical_layer % self.physical_slots

    def acquire(self, request_id: str, logical_layer: int) -> LayerSlotToken:
        """Acquire the deterministic slot for a layer and request chunk."""

        if not request_id:
            raise ValueError("request_id must not be empty")
        self._validate_layer(logical_layer)
        slot_id = self.preferred_slot(logical_layer)
        current = self._owners.get(slot_id)
        if current is not None:
            if (
                current.token.request_id == request_id
                and current.token.logical_layer == logical_layer
            ):
                raise SlotBusyError("slot is already owned by this request")
            if not current.reusable:
                raise SlotBusyError(
                    f"slot {slot_id} is owned by request {current.token.request_id} "
                    f"layer {current.token.logical_layer}"
                )
        self._generations[slot_id] += 1
        token = LayerSlotToken(
            slot_id=slot_id,
            request_id=request_id,
            logical_layer=logical_layer,
            generation=self._generations[slot_id],
        )
        self._owners[slot_id] = _SlotOwner(token=token)
        return token

    def set_pd_readers(self, token: LayerSlotToken, reader_count: int) -> None:
        if reader_count < 0:
            raise ValueError("reader_count must be non-negative")
        owner = self._get_owner(token)
        owner.pending_pd_reads = reader_count

    def mark_host_saved(self, token: LayerSlotToken) -> None:
        self._get_owner(token).host_saved = True

    def mark_attention_done(self, token: LayerSlotToken) -> None:
        self._get_owner(token).attention_done = True

    def mark_pd_read_done(self, token: LayerSlotToken, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        owner = self._get_owner(token)
        if count > owner.pending_pd_reads:
            raise ValueError("PD completion exceeds pending reader count")
        owner.pending_pd_reads -= count

    def mark_failed(self, token: LayerSlotToken) -> None:
        self._get_owner(token).failed = True

    def pending_pd_readers(self, token: LayerSlotToken) -> int:
        return self._get_owner(token).pending_pd_reads

    def mark_failed_local_work_drained(self, token: LayerSlotToken) -> None:
        """Confirm failed device work is joined or was never enqueued."""
        owner = self._get_owner(token)
        if not owner.failed:
            raise ValueError("drain confirmation requires a failed slot")
        owner.host_saved = True
        owner.attention_done = True

    def release(self, token: LayerSlotToken) -> None:
        """Release a completed token or discard a failed token."""

        owner = self._get_owner(token)
        if owner.failed and (
            owner.pending_pd_reads or not owner.host_saved or not owner.attention_done
        ):
            raise SlotBusyError("failed slot still has pending PD or device work")
        if not owner.reusable and not owner.failed:
            raise SlotBusyError("slot cannot be released before all completion gates")
        del self._owners[token.slot_id]

    def is_reusable(self, token: LayerSlotToken) -> bool:
        return self._get_owner(token).reusable

    def owner(self, slot_id: int) -> Optional[LayerSlotToken]:
        if slot_id < 0 or slot_id >= self.physical_slots:
            raise IndexError(f"invalid slot_id {slot_id}")
        owner = self._owners.get(slot_id)
        return None if owner is None else owner.token

    def active_tokens(self) -> Tuple[LayerSlotToken, ...]:
        return tuple(owner.token for owner in self._owners.values())

    def _get_owner(self, token: LayerSlotToken) -> _SlotOwner:
        owner = self._owners.get(token.slot_id)
        if owner is None or owner.token != token:
            raise StaleGenerationError(
                f"stale or unknown slot token: slot={token.slot_id}, "
                f"generation={token.generation}"
            )
        return owner

    def _validate_layer(self, logical_layer: int) -> None:
        if logical_layer < 0 or logical_layer >= self.logical_layers:
            raise IndexError(f"invalid logical_layer {logical_layer}")


@dataclass(frozen=True)
class KvPageRange:
    """A valid logical token range, including non-aligned tail pages."""

    start_token: int
    end_token: int

    def __post_init__(self) -> None:
        if self.start_token < 0 or self.end_token <= self.start_token:
            raise ValueError("invalid KV page range")


class LayerComponentManifest:
    """Describes component ranges published for one layer generation."""

    def __init__(self, ranges: Optional[Dict[KvComponent, KvPageRange]] = None):
        self._ranges: Dict[KvComponent, KvPageRange] = dict(ranges or {})

    def publish(self, component: KvComponent, valid_range: KvPageRange) -> None:
        self._ranges[component] = valid_range

    def get(self, component: KvComponent) -> Optional[KvPageRange]:
        return self._ranges.get(component)

    def components(self) -> Set[KvComponent]:
        return set(self._ranges)


@dataclass(frozen=True)
class HostPageAllocation:
    """Logical request pages mapped to stable Host page IDs."""

    request_id: str
    page_ids: Tuple[int, ...]
    page_size: int
    generation: int = 0

    def page_for_token(self, token_index: int) -> int:
        if token_index < 0:
            raise ValueError("token_index must be non-negative")
        page_offset = token_index // self.page_size
        if page_offset >= len(self.page_ids):
            raise IndexError(f"token {token_index} is outside the allocation")
        return self.page_ids[page_offset]


class HostKvPagePool:
    """Logical Host-page allocator independent of a tensor implementation.

    A concrete NPU adapter can use ``page_ids`` to index registered Host
    tensors or Memcache GVAs.  The pool intentionally never exposes a raw
    process pointer as a transport address.
    """

    def __init__(self, page_count: int, page_size: int):
        if page_count <= 0 or page_size <= 0:
            raise ValueError("page_count and page_size must be positive")
        self.page_count = page_count
        self.page_size = page_size
        self._free_pages = set(range(page_count))
        self._allocations: Dict[str, HostPageAllocation] = {}
        self._generation = 0

    def allocate(self, request_id: str, token_count: int) -> HostPageAllocation:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if request_id in self._allocations:
            raise ValueError(f"request {request_id} already has a Host allocation")
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        page_count = (token_count + self.page_size - 1) // self.page_size
        if page_count > len(self._free_pages):
            raise MemoryError(
                f"Host KV pool exhausted: need {page_count} pages, "
                f"available {len(self._free_pages)}"
            )
        page_ids = tuple(sorted(self._free_pages)[:page_count])
        self._free_pages.difference_update(page_ids)
        self._generation += 1
        allocation = HostPageAllocation(
            request_id, page_ids, self.page_size, self._generation
        )
        self._allocations[request_id] = allocation
        return allocation

    def release(self, allocation: HostPageAllocation) -> None:
        current = self._allocations.get(allocation.request_id)
        if current != allocation:
            raise ValueError("unknown or stale Host page allocation")
        del self._allocations[allocation.request_id]
        overlap = self._free_pages.intersection(allocation.page_ids)
        if overlap:
            raise ValueError(f"Host pages released twice: {sorted(overlap)}")
        self._free_pages.update(allocation.page_ids)

    def grow(self, request_id: str, token_count: int) -> HostPageAllocation:
        current = self.allocation(request_id)
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        page_count = (token_count + self.page_size - 1) // self.page_size
        additional = page_count - len(current.page_ids)
        if additional <= 0:
            return current
        if additional > len(self._free_pages):
            raise MemoryError("Host KV pool exhausted while growing request")
        new_pages = tuple(sorted(self._free_pages)[:additional])
        allocation = HostPageAllocation(
            request_id,
            current.page_ids + new_pages,
            self.page_size,
            current.generation,
        )
        self._free_pages.difference_update(new_pages)
        self._allocations[request_id] = allocation
        return allocation

    def allocation(self, request_id: str) -> HostPageAllocation:
        try:
            return self._allocations[request_id]
        except KeyError as exc:
            raise KeyError(f"request {request_id} has no Host allocation") from exc

    @property
    def free_page_count(self) -> int:
        return len(self._free_pages)


@dataclass(frozen=True)
class PdComponentDescriptor:
    """Serializable mapping for one layer/component transfer."""

    request_id: str
    logical_layer: int
    generation: int
    component: KvComponent
    source_kind: PdSourceKind
    source_slot: int
    source_page_ids: Tuple[int, ...]
    destination_page_ids: Tuple[int, ...]
    valid_range: KvPageRange
    bytes_per_page: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.source_slot < 0:
            raise ValueError("source_slot must be non-negative")
        if len(self.source_page_ids) != len(self.destination_page_ids):
            raise ValueError("source and destination page counts must match")
        if self.bytes_per_page <= 0:
            raise ValueError("bytes_per_page must be positive")


@dataclass(frozen=True)
class LayerReadyRecord:
    """PD publication emitted only after the source layer is ready."""

    token: LayerSlotToken
    components: Tuple[PdComponentDescriptor, ...]

    def component(self, component: KvComponent) -> PdComponentDescriptor:
        for descriptor in self.components:
            if descriptor.component == component:
                return descriptor
        raise KeyError(f"component {component.value} is not published")


class PdTransferState(str, Enum):
    """Control-plane state for one published layer generation."""

    PUBLISHED = "published"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class _PendingTransfer:
    record: LayerReadyRecord
    state: PdTransferState = PdTransferState.PUBLISHED


class LayerReadyTransfer:
    """Small control-plane adapter around a layer-ready transport.

    ``send`` is injected by the real MemFabric implementation.  The adapter
    tracks acknowledgements separately from publication, so a queued control
    message cannot release a P slot before the D-side read has completed.
    """

    def __init__(self, send: Callable[[LayerReadyRecord], None]):
        self.send = send
        self._pending: Dict[LayerSlotToken, _PendingTransfer] = {}

    def publish(self, record: LayerReadyRecord) -> None:
        if record.token in self._pending:
            raise ValueError("layer generation was published twice")
        self._pending[record.token] = _PendingTransfer(record)
        try:
            self.send(record)
        except Exception:
            self._pending[record.token].state = PdTransferState.FAILED
            raise

    def complete(self, token: LayerSlotToken) -> LayerReadyRecord:
        pending = self._get_pending(token)
        if pending.state != PdTransferState.PUBLISHED:
            raise RuntimeError(f"transfer is already {pending.state.value}")
        pending.state = PdTransferState.COMPLETED
        return pending.record

    def fail(self, token: LayerSlotToken) -> LayerReadyRecord:
        pending = self._get_pending(token)
        if pending.state != PdTransferState.PUBLISHED:
            raise RuntimeError(f"transfer is already {pending.state.value}")
        pending.state = PdTransferState.FAILED
        return pending.record

    def state(self, token: LayerSlotToken) -> PdTransferState:
        return self._get_pending(token).state

    def forget(self, token: LayerSlotToken) -> None:
        pending = self._get_pending(token)
        if pending.state == PdTransferState.PUBLISHED:
            raise SlotBusyError("cannot forget an active transfer")
        del self._pending[token]

    def _get_pending(self, token: LayerSlotToken) -> _PendingTransfer:
        try:
            return self._pending[token]
        except KeyError as exc:
            raise StaleGenerationError("unknown layer-ready transfer") from exc


class StagingCreditPool:
    """Bounded credits for a fallback Host/NPU staging receive window."""

    def __init__(self, chunk_bytes: int, chunks: int = 2):
        if chunk_bytes <= 0 or chunks <= 0:
            raise ValueError("chunk_bytes and chunks must be positive")
        self.chunk_bytes = chunk_bytes
        self.capacity = chunks
        self.available = chunks

    def acquire(self, requested_bytes: int) -> int:
        if requested_bytes <= 0:
            raise ValueError("requested_bytes must be positive")
        if self.available == 0:
            raise SlotBusyError("staging window has no available credit")
        self.available -= 1
        return min(requested_bytes, self.chunk_bytes)

    def release(self) -> None:
        if self.available >= self.capacity:
            raise ValueError("staging credit released twice")
        self.available += 1


class LayerReadyTransportAdapter:
    """Stable transport boundary for Mooncake/MemFabric implementations.

    ``send`` receives a layer record and must return an opaque transfer handle.
    Completion is reported by the existing ``LayerReadyTransfer`` protocol;
    this class deliberately does not equate send-return with DMA completion.
    """

    def __init__(self, send: Callable[[LayerReadyRecord], object], staging=None):
        self.send = send
        self.staging = staging
        self.handles: Dict[LayerSlotToken, object] = {}
        self.completed: Set[LayerSlotToken] = set()
        self.failed: Set[LayerSlotToken] = set()

    def publish(self, record: LayerReadyRecord) -> object:
        if record.token in self.handles:
            raise ValueError("layer record was published twice")
        handle = self.send(record)
        self.handles[record.token] = handle
        return handle

    def reserve_staging(self, byte_count: int) -> int:
        if self.staging is None:
            raise RuntimeError("staging fallback is not configured")
        return self.staging.acquire(byte_count)

    def complete_staging(self) -> None:
        if self.staging is None:
            raise RuntimeError("staging fallback is not configured")
        self.staging.release()

    def acknowledge(self, token: LayerSlotToken, *, success: bool = True) -> object:
        """Record an explicit remote completion for a published layer."""
        if token not in self.handles:
            raise StaleGenerationError("unknown layer-ready transport handle")
        if token in self.completed or token in self.failed:
            raise ValueError("layer transport was acknowledged twice")
        if success:
            self.completed.add(token)
        else:
            self.failed.add(token)
        return self.handles[token]

    def state(self, token: LayerSlotToken) -> str:
        if token in self.completed:
            return "completed"
        if token in self.failed:
            return "failed"
        if token in self.handles:
            return "published"
        raise StaleGenerationError("unknown layer-ready transport handle")

    def forget(self, token: LayerSlotToken) -> object:
        if self.state(token) == "published":
            raise SlotBusyError("cannot forget an active layer-ready transfer")
        try:
            handle = self.handles.pop(token)
            self.completed.discard(token)
            self.failed.discard(token)
            return handle
        except KeyError as exc:
            raise StaleGenerationError("unknown transport handle") from exc


@dataclass
class LayerWork:
    """Lifecycle state for one request chunk and logical layer."""

    token: LayerSlotToken
    host_allocation: HostPageAllocation
    valid_range: KvPageRange
    restored: bool = False
    kv_written: bool = False
    host_save_started: bool = False
    ready_record: Optional[LayerReadyRecord] = None


class PrefillLayerwiseOffload:
    """Orchestrate P layerwise restore, save, and PD publication.

    The callbacks are the only integration points for device copies and
    MemFabric.  They receive immutable metadata and may enqueue asynchronous
    work, but completion must be reported explicitly through this object.
    """

    def __init__(
        self,
        slot_manager: PrefillLayerSlotManager,
        host_pool: HostKvPagePool,
        *,
        restore: Optional[Callable[[LayerWork], None]] = None,
        save: Optional[Callable[[LayerWork], None]] = None,
        publish: Optional[Callable[[LayerReadyRecord], None]] = None,
        pd_transfer: Optional[LayerReadyTransfer] = None,
    ):
        self.slot_manager = slot_manager
        self.host_pool = host_pool
        self.restore_callback = restore
        self.save_callback = save
        self.publish_callback = publish
        self.pd_transfer = pd_transfer
        self._work: Dict[LayerSlotToken, LayerWork] = {}

    def begin_request(self, request_id: str, token_count: int) -> HostPageAllocation:
        """Reserve the request's Host pages before the first layer is run."""

        return self.host_pool.allocate(request_id, token_count)

    def request_allocation(self, request_id: str) -> HostPageAllocation:
        return self.host_pool.allocation(request_id)

    def grow_request(self, request_id: str, token_count: int) -> HostPageAllocation:
        if any(work.token.request_id == request_id for work in self._work.values()):
            raise SlotBusyError("drain previous layer work before growing a request")
        return self.host_pool.grow(request_id, token_count)

    def end_request(self, request_id: str) -> None:
        """Release Host pages after every layer generation is released."""

        self.host_pool.release(self.validate_request_release(request_id))

    def validate_request_release(self, request_id: str) -> HostPageAllocation:
        active = [
            work for work in self._work.values() if work.token.request_id == request_id
        ]
        if active:
            raise SlotBusyError(
                f"request {request_id} still owns {len(active)} layer slots"
            )
        return self.host_pool.allocation(request_id)

    def begin_layer(
        self,
        request_id: str,
        logical_layer: int,
        token_count: int,
        valid_range: KvPageRange,
        *,
        pd_readers: int = 0,
        restore: bool = True,
    ) -> LayerWork:
        allocation = self.host_pool.allocation(request_id)
        if token_count <= 0 or valid_range.end_token > token_count:
            raise ValueError("layer range must fit the valid token count")
        if pd_readers < 0:
            raise ValueError("pd_readers must be non-negative")
        if token_count > len(allocation.page_ids) * allocation.page_size:
            raise ValueError("token_count exceeds the request Host allocation")
        token = self.slot_manager.acquire(request_id, logical_layer)
        self.slot_manager.set_pd_readers(token, pd_readers)
        work = LayerWork(token, allocation, valid_range)
        self._work[token] = work
        if restore and self.restore_callback is not None:
            try:
                self.restore_callback(work)
            except Exception:
                self.slot_manager.mark_failed(token)
                raise
        work.restored = restore
        return work

    def mark_kv_written(self, work: LayerWork) -> None:
        self._get_work(work).kv_written = True

    def start_host_save(self, work: LayerWork) -> None:
        work = self._get_work(work)
        if not work.restored or not work.kv_written:
            raise RuntimeError("Host save requires restored and written KV")
        if work.host_save_started:
            raise RuntimeError("Host save already started")
        work.host_save_started = True
        if self.save_callback is not None:
            try:
                self.save_callback(work)
            except Exception:
                self.slot_manager.mark_failed(work.token)
                raise

    def complete_host_save(self, work: LayerWork) -> None:
        work = self._get_work(work)
        if not work.host_save_started:
            raise RuntimeError("Host save was not started")
        self.slot_manager.mark_host_saved(work.token)

    def publish_layer_ready(
        self,
        work: LayerWork,
        descriptors: List[PdComponentDescriptor],
    ) -> LayerReadyRecord:
        work = self._get_work(work)
        if not work.kv_written:
            raise RuntimeError("PD publication requires written KV")
        if work.ready_record is not None:
            raise RuntimeError("layer generation was published twice")
        record = LayerReadyRecord(work.token, tuple(descriptors))
        for descriptor in record.components:
            if descriptor.request_id != work.token.request_id:
                raise ValueError("PD descriptor request does not match layer work")
            if descriptor.logical_layer != work.token.logical_layer:
                raise ValueError("PD descriptor layer does not match layer work")
            if descriptor.generation != work.token.generation:
                raise StaleGenerationError("PD descriptor generation is stale")
        work.ready_record = record
        try:
            if self.publish_callback is not None:
                self.publish_callback(record)
            if self.pd_transfer is not None:
                self.pd_transfer.publish(record)
        except Exception:
            self.slot_manager.mark_failed(work.token)
            raise
        return record

    def mark_attention_done(self, work: LayerWork) -> None:
        self.slot_manager.mark_attention_done(self._get_work(work).token)

    def mark_pd_read_done(self, work: LayerWork, count: int = 1) -> None:
        self.slot_manager.mark_pd_read_done(self._get_work(work).token, count)

    def acknowledge_pd_read(self, work: LayerWork, *, success: bool = True) -> None:
        """Apply one D-side transfer result to the slot lifetime."""

        work = self._get_work(work)
        pending_readers = self.slot_manager.pending_pd_readers(work.token)
        if pending_readers <= 0:
            raise ValueError("PD completion exceeds pending reader count")
        if self.pd_transfer is not None:
            if self.pd_transfer.state(work.token) == PdTransferState.PUBLISHED:
                if not success:
                    self.pd_transfer.fail(work.token)
                elif pending_readers == 1:
                    self.pd_transfer.complete(work.token)
            if not success:
                self.slot_manager.mark_failed(work.token)
        elif not success:
            self.slot_manager.mark_failed(work.token)
        self.slot_manager.mark_pd_read_done(work.token)

    def release_layer(self, work: LayerWork) -> None:
        work = self._get_work(work)
        if self.pd_transfer is not None and work.ready_record is not None:
            if self.pd_transfer.state(work.token) == PdTransferState.PUBLISHED:
                raise SlotBusyError("layer still has an active transport")
        self.slot_manager.release(work.token)
        if self.pd_transfer is not None and work.ready_record is not None:
            self.pd_transfer.forget(work.token)
        del self._work[work.token]

    def fail_layer(self, work: LayerWork) -> None:
        self.slot_manager.mark_failed(self._get_work(work).token)

    def _get_work(self, work: LayerWork) -> LayerWork:
        current = self._work.get(work.token)
        if current is not work:
            raise StaleGenerationError(
                "layer work is stale or not owned by this coordinator"
            )
        return current
