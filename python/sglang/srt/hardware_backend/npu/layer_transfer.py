"""Byte-exact layer transfers using registered transport address spaces."""

from dataclasses import dataclass
from typing import Callable, Mapping

from sglang.srt.hardware_backend.npu.prefill_offload import (
    KvComponent,
    LayerReadyRecord,
    PdSourceKind,
)


@dataclass(frozen=True)
class RegisteredLayerRegion:
    """A transport registration retained by its local allocation owner.

    Direct addresses belong to the transfer-engine registration. With staged
    receive, local Host owners retain HAL mappings and the wire uses opaque
    component identities instead of exposing these Host addresses.
    """

    address: int
    nbytes: int
    bytes_per_page: int
    registration_id: str
    kind: PdSourceKind
    dtype: str
    owner: object

    def __post_init__(self):
        if min(self.address, self.nbytes, self.bytes_per_page) <= 0:
            raise ValueError("registered region address and geometry must be positive")
        if self.nbytes % self.bytes_per_page:
            raise ValueError("registered region must contain complete pages")
        if not self.registration_id or not self.dtype or self.owner is None:
            raise ValueError("registered region requires identity, dtype and an owner")


@dataclass(frozen=True)
class PrefillLayerTarget:
    """One receiver lease negotiated for a request and logical layer."""

    session_id: str
    regions: Mapping[KvComponent, RegisteredLayerRegion]
    page_ids: Mapping[KvComponent, tuple]
    notify: Callable
    transfer_components: frozenset | None = None
    staging: dict | None = None

    def __post_init__(self):
        if not self.session_id or not callable(self.notify):
            raise ValueError(
                "layer target requires a session and receiver notification"
            )
        if set(self.regions) != set(self.page_ids):
            raise ValueError("layer target component regions and page mappings differ")
        if (
            self.transfer_components is not None
            and not self.transfer_components.issubset(self.regions)
        ):
            raise ValueError("layer target transfer components are not registered")


@dataclass(frozen=True)
class LayerTransferSpan:
    component: KvComponent
    source_address: int
    destination_offset: int
    nbytes: int


def build_layer_transfer_spans(
    record: LayerReadyRecord,
    sources: Mapping[KvComponent, RegisteredLayerRegion],
    destinations: Mapping[KvComponent, RegisteredLayerRegion],
    *,
    page_size: int,
    transfer_components: frozenset | None = None,
) -> list[LayerTransferSpan]:
    """Translate logical token ranges without copying prefix or padded tail bytes."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    components = {descriptor.component for descriptor in record.components}
    if KvComponent.LATENT_K not in components:
        raise ValueError("layer publication requires a latent component")
    if components != set(sources) or components != set(destinations):
        raise ValueError("layer publication components do not match registered regions")
    if transfer_components is not None and transfer_components not in (
        frozenset(destinations),
        frozenset({KvComponent.INDEXER_K}.intersection(destinations)),
    ):
        raise ValueError(
            "partial layer transfer must be a shared-Host Indexer consumer"
        )
    seen = set()
    blocks = []
    for descriptor in record.components:
        component = descriptor.component
        if component in seen:
            raise ValueError("duplicate layer component")
        seen.add(component)
        if (
            descriptor.request_id != record.token.request_id
            or descriptor.logical_layer != record.token.logical_layer
            or descriptor.generation != record.token.generation
            or descriptor.source_slot != record.token.slot_id
        ):
            raise ValueError("descriptor does not match the published layer generation")
        source = sources[component]
        destination = destinations[component]
        expected_destination = (
            PdSourceKind.NPU_REGISTERED
            if component == KvComponent.INDEXER_K
            else PdSourceKind.HOST_REGISTERED
        )
        if (
            source.kind != descriptor.source_kind
            or destination.kind != expected_destination
        ):
            raise ValueError("component transfer uses an incorrect address-space kind")
        page_bytes = descriptor.bytes_per_page
        if (
            source.bytes_per_page != page_bytes
            or destination.bytes_per_page != page_bytes
            or source.dtype != destination.dtype
            or page_bytes % page_size
        ):
            raise ValueError("source and destination component layouts do not match")
        token_bytes = page_bytes // page_size
        first_page = descriptor.valid_range.start_token // page_size
        end_page = (descriptor.valid_range.end_token + page_size - 1) // page_size
        if (
            len(descriptor.source_page_ids) < end_page
            or len(descriptor.destination_page_ids) < end_page
        ):
            raise ValueError("page mapping does not cover the valid token range")
        for page in range(first_page, end_page):
            source_page = descriptor.source_page_ids[page]
            destination_page = descriptor.destination_page_ids[page]
            if (
                not 0 <= source_page < source.nbytes // page_bytes
                or not 0 <= destination_page < destination.nbytes // page_bytes
            ):
                raise ValueError("transfer page exceeds its registered region")
            begin = max(descriptor.valid_range.start_token, page * page_size)
            end = min(descriptor.valid_range.end_token, (page + 1) * page_size)
            offset = (begin % page_size) * token_bytes
            if transfer_components is not None and component not in transfer_components:
                continue
            blocks.append(
                LayerTransferSpan(
                    component,
                    source.address + source_page * page_bytes + offset,
                    destination_page * page_bytes + offset,
                    (end - begin) * token_bytes,
                )
            )
    return blocks


def build_layer_transfer_blocks(
    record, sources, destinations, *, page_size, transfer_components=None
):
    return [
        (
            span.source_address,
            destinations[span.component].address + span.destination_offset,
            span.nbytes,
        )
        for span in build_layer_transfer_spans(
            record,
            sources,
            destinations,
            page_size=page_size,
            transfer_components=transfer_components,
        )
    ]


class LayerTransferOperation:
    """Keep registrations alive across transfer failures and unacknowledged writes."""

    def __init__(self, record, sources, destinations, blocks):
        self.record = record
        self.sources = dict(sources)
        self.destinations = dict(destinations)
        self.blocks = tuple(blocks)
        self.state = "pending"
        self.transport_drained = True

    def execute(self, transfer, session_id):
        if self.state != "pending":
            raise RuntimeError("layer transfer was already submitted")
        self.state = "transferring"
        self.transport_drained = False
        try:
            status = transfer(session_id, self.blocks)
            if status != 0:
                raise RuntimeError(f"layer transfer failed with status {status}")
        except Exception:
            self.state = "failed"
            raise
        self.state = "transferred"
        self.transport_drained = True

    def acknowledge(self):
        if self.state != "transferred":
            raise RuntimeError("receiver ACK requires a completed layer transfer")
        self.state = "acknowledged"

    def confirm_failed_drain(self):
        if self.state != "failed":
            raise RuntimeError("drain confirmation requires a failed transfer")
        self.state = "drained"
        self.transport_drained = True
