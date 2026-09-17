"""Versioned layer leases and visibility acknowledgements for NPU PD."""

import json
from dataclasses import dataclass, field
from uuid import uuid4

from sglang.srt.hardware_backend.npu.layer_transfer import (
    PrefillLayerTarget,
    RegisteredLayerRegion,
)
from sglang.srt.hardware_backend.npu.prefill_offload import KvComponent, PdSourceKind

VERSION = 1
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def encode(payload):
    message = json.dumps({"version": VERSION, **payload}, separators=(",", ":")).encode()
    if len(message) > MAX_MESSAGE_BYTES:
        raise ValueError("layer control message exceeds size limit")
    return message


def decode(message):
    if len(message) > MAX_MESSAGE_BYTES:
        raise ValueError("layer control message exceeds size limit")
    payload = json.loads(message)
    if not isinstance(payload, dict) or payload.get("version") not in (VERSION, 2, 3, 4):
        raise ValueError("unsupported layer protocol version")
    return payload


def token_identity(token):
    return [token.slot_id, token.request_id, token.logical_layer, token.generation]


@dataclass
class LayerReceiveLease:
    room: int
    session_id: str
    page_size: int
    layers: dict
    page_ids: dict
    visibility: object
    prefix: int = 0
    lease_id: str = field(default_factory=lambda: uuid4().hex)
    failed: bool = False
    drained: bool = False
    prefixes: dict = field(default_factory=dict)
    completed: dict = field(default_factory=dict)
    request_id: str = ""
    writer_endpoint: tuple | None = None
    shared_group_id: str = ""
    shared_host_writer: bool = True
    staging: dict | None = None
    staged_progress: dict = field(default_factory=dict)
    staged_intervals: dict = field(default_factory=dict)
    layer_writers: dict = field(default_factory=dict)
    drained_writers: set = field(default_factory=set)

    def __post_init__(self):
        if self.room < 0 or not self.session_id or self.page_size <= 0 or not self.layers:
            raise ValueError("invalid layer receive lease")
        if not callable(self.visibility) or self.prefix < 0:
            raise ValueError("layer lease requires visibility completion and a valid prefix")
        components = set(self.page_ids)
        if KvComponent.LATENT_K not in components or not components.issubset(set(KvComponent)):
            raise ValueError("DSA lease requires latent mappings and known optional components")
        for layer_id, regions in self.layers.items():
            if layer_id < 0 or set(regions) != set(self.page_ids):
                raise ValueError("layer lease component inventory mismatch")
            for component, region in regions.items():
                pages = self.page_ids[component]
                expected = (PdSourceKind.NPU_REGISTERED if component == KvComponent.INDEXER_K
                            else PdSourceKind.HOST_REGISTERED)
                if region.kind != expected or region.bytes_per_page % self.page_size:
                    raise ValueError("layer registration address space or page layout mismatch")
                if not pages or len(set(pages)) != len(pages) or any(
                    page < 0 or page >= region.nbytes // region.bytes_per_page for page in pages
                ):
                    raise ValueError("receiver pages exceed registration or alias within a request")
            if self.prefix > min(map(len, self.page_ids.values())) * self.page_size:
                raise ValueError("receiver prefix exceeds its page mapping")
            self.prefixes[layer_id] = self.prefix

    def manifest(self):
        multiple_writers = len(self.writer_endpoints()) > 1
        return encode({
            "version": 4 if multiple_writers else (3 if self.staging is not None else (2 if self.shared_group_id else VERSION)),
            "layer_writers": ({str(layer): endpoint for layer, endpoint in self.layer_writers.items()}
                              if multiple_writers else None),
            "staging": self.staging,
            "shared_host": ({"group": self.shared_group_id, "writer": self.shared_host_writer}
                            if self.shared_group_id else None),
            "room": self.room, "session": self.session_id, "lease": self.lease_id,
            "page_size": self.page_size,
            "pages": {component.value: pages for component, pages in self.page_ids.items()},
            "layers": {
                str(layer): {
                    component.value: {
                        "address": 1 if self.staging is not None else region.address,
                        "nbytes": region.nbytes,
                        "page_bytes": region.bytes_per_page,
                        "registration": region.registration_id,
                        "kind": region.kind.value, "dtype": region.dtype,
                    } for component, region in regions.items()
                } for layer, regions in self.layers.items()
            },
        })

    def receive(self, payload):
        identity, layer, start, end = self.validate_notification(payload)
        if identity in self.completed:
            return
        if self.staging is not None:
            progress = self.staged_progress.get(identity, {})
            for component in self.receive_components():
                region = self.layers[layer][component]
                required = (end - start) * (region.bytes_per_page // self.page_size)
                if progress.get(component, 0) != required:
                    raise RuntimeError("layer READY arrived before all staged bytes completed")
        self.visibility(layer, start, end)
        self.request_id = identity[1]
        self.prefixes[layer] = end
        self.completed[identity] = (start, end)
        self.staged_progress.pop(identity, None)
        self.staged_intervals.pop(identity, None)

    def receive_components(self):
        if self.shared_group_id and not self.shared_host_writer:
            return frozenset({KvComponent.INDEXER_K}.intersection(self.page_ids))
        return frozenset(self.page_ids)

    def validate_notification(self, payload):
        if self.failed:
            raise RuntimeError("layer receive lease has failed")
        self.validate_writer(payload)
        if (payload["room"], payload["session"], payload["lease"]) != (
            self.room, self.session_id, self.lease_id
        ):
            raise ValueError("layer notification names a stale receiver lease")
        slot, request, layer, generation = payload["token"]
        if slot < 0 or generation <= 0 or not request or layer not in self.layers:
            raise ValueError("invalid layer notification identity")
        if self.request_id and request != self.request_id:
            raise ValueError("layer notification changed request identity")
        start, end = payload["range"]
        identity = (slot, request, layer, generation)
        expected = {component.value: region.registration_id
                    for component, region in self.layers[layer].items()}
        if payload["registrations"] != expected:
            raise ValueError("layer notification registration identities differ")
        if identity in self.completed:
            if self.completed[identity] != (start, end):
                raise ValueError("duplicate layer notification changed its valid range")
            return identity, layer, start, end
        if start != self.prefixes[layer] or end <= start:
            raise ValueError("layer notification is stale, overlapping or has a prefix gap")
        if end > min(map(len, self.page_ids.values())) * self.page_size:
            raise ValueError("layer notification exceeds receiver page capacity")
        if self.staged_intervals.get(identity, (start, end)) != (start, end):
            raise ValueError("staged generation changed its valid interval")
        return identity, layer, start, end

    def validate_writer(self, payload):
        endpoint = payload.get("reply_ip"), payload.get("reply_port")
        if "token" in payload and self.layer_writers:
            expected = self.layer_writers.get(payload["token"][2])
            if endpoint != expected:
                raise ValueError("layer notification did not come from its negotiated layer writer")
        elif self.writer_endpoints() and endpoint not in self.writer_endpoints():
            raise ValueError("layer notification did not come from the negotiated writer")

    def writer_endpoints(self):
        if self.layer_writers:
            return set(self.layer_writers.values())
        return {self.writer_endpoint} if self.writer_endpoint is not None else set()

    def note_writer_drain(self, payload):
        self.validate_writer(payload)
        endpoints = self.writer_endpoints()
        if not endpoints:
            raise RuntimeError("receiver lease has no negotiated abort writers")
        self.drained_writers.add((payload["reply_ip"], payload["reply_port"]))
        return endpoints.issubset(self.drained_writers)


def select_layer_writer(bootstrap_infos):
    writers = [peer for peer in bootstrap_infos if not peer["is_dummy"]]
    if len(writers) != 1:
        raise ValueError("replicated MLA layer receive requires exactly one data writer")
    peer = writers[0]
    return peer["rank_ip"], int(peer["rank_port"])


def select_layer_writers(bootstrap_infos, layer_ids):
    writers = [peer for peer in bootstrap_infos if not peer["is_dummy"]]
    selected = {}
    for layer in layer_ids:
        candidates = []
        for peer in writers:
            start, end = peer.get("layer_start"), peer.get("layer_end")
            if start is None and end is None and len(writers) == 1:
                candidates.append(peer)
            elif type(start) is int and type(end) is int and 0 <= start < end:
                if start <= layer < end:
                    candidates.append(peer)
            else:
                raise ValueError("multiple Prefill writers require explicit layer intervals")
        if len(candidates) != 1:
            raise ValueError(f"layer {layer} must have exactly one Prefill data writer")
        selected[layer] = (candidates[0]["rank_ip"], int(candidates[0]["rank_port"]))
    return selected


def parse_targets(message, *, room, session_id, page_size, notify, writer_endpoint=None):
    payload = decode(message)
    if (payload["room"], payload["session"], payload["page_size"]) != (
        room, session_id, page_size
    ) or not payload.get("lease"):
        raise ValueError("receiver manifest disagrees with the bootstrap endpoint")
    pages = {KvComponent(name): tuple(values) for name, values in payload["pages"].items()}
    layers = {}
    shared = payload.get("shared_host")
    transfer_components = None
    if shared is not None:
        if (payload["version"] not in (2, 3, 4) or not isinstance(shared, dict)
                or not isinstance(shared.get("group"), str) or not shared["group"]
                or type(shared.get("writer")) is not bool):
            raise ValueError("invalid shared Host receiver ownership")
        if not shared["writer"]:
            transfer_components = frozenset({KvComponent.INDEXER_K})
    staging = payload.get("staging")
    if staging is not None:
        if (payload["version"] not in (3, 4) or not isinstance(staging, dict)
                or type(staging.get("window_bytes")) is not int or staging["window_bytes"] < 32
                or staging["window_bytes"] % 32 or not isinstance(staging.get("registration"), str)
                or not staging["registration"]):
            raise ValueError("invalid layer staging negotiation")
    for layer, components in payload["layers"].items():
        regions = {}
        for name, region in components.items():
            regions[KvComponent(name)] = RegisteredLayerRegion(
                region["address"], region["nbytes"], region["page_bytes"],
                region["registration"], PdSourceKind(region["kind"]), region["dtype"], payload,
            )
        layers[int(layer)] = regions
    writers = payload.get("layer_writers")
    if writers is not None:
        if payload["version"] != 4 or not isinstance(writers, dict):
            raise ValueError("invalid per-layer writer negotiation")
        writers = {int(layer): tuple(endpoint) for layer, endpoint in writers.items()}
        if set(writers) != set(layers) or any(
            len(endpoint) != 2 or not isinstance(endpoint[0], str) or not endpoint[0]
            or type(endpoint[1]) is not int or not 0 < endpoint[1] <= 65535
            for endpoint in writers.values()
        ):
            raise ValueError("per-layer writer inventory does not match receiver layers")
    LayerReceiveLease(room, session_id, page_size, layers, pages, lambda *args: None,
                      layer_writers=writers or {})
    return {
        layer: PrefillLayerTarget(
            session_id, regions, pages,
            lambda record, operation, lease=payload["lease"]: notify(lease, record, operation),
            transfer_components,
            staging,
        ) for layer, regions in layers.items()
        if writers is None or writer_endpoint is None or writers[layer] == writer_endpoint
    }


def ready_payload(room, session, lease, record, operation):
    intervals = {(entry.valid_range.start_token, entry.valid_range.end_token)
                 for entry in record.components}
    if len(intervals) != 1:
        raise ValueError("all layer components must publish the same valid interval")
    return {
        "room": room, "session": session, "lease": lease,
        "token": token_identity(record.token), "range": list(intervals.pop()),
        "registrations": {component.value: region.registration_id
                          for component, region in operation.destinations.items()},
    }
