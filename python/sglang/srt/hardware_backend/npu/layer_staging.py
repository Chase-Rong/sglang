"""One bounded receive window, with credit returned after local copy completion."""

import threading
import time
from uuid import uuid4

from sglang.srt.hardware_backend.npu.prefill_offload import KvComponent, PdSourceKind


class LayerStagingReceiver:
    def __init__(self, address, window_bytes, *, owner, copy, drain):
        if address <= 0 or window_bytes < 32 or window_bytes % 32:
            raise ValueError("layer staging requires a positive, 32-byte aligned window")
        if owner is None or not callable(copy) or not callable(drain):
            raise ValueError("layer staging requires retained storage and local copy/drain")
        self.address = address
        self.window_bytes = window_bytes
        self.owner = owner
        self.copy = copy
        self.drain = drain
        self.registration = uuid4().hex
        self.generation = 0
        self.active = None
        self.last_commits = {}
        self.layer_owners = {}
        self.lock = threading.RLock()

    def manifest(self):
        return {"window_bytes": self.window_bytes, "registration": self.registration}

    def _validate(self, lease, payload):
        if (lease.staging != self.manifest()
                or payload.get("staging_registration") != self.registration):
            raise ValueError("staging request names an old window registration")
        identity, layer, start, end = lease.validate_notification(payload)
        if identity in lease.completed:
            raise ValueError("staging request names an already completed layer")
        component = KvComponent(payload["component"])
        if component not in lease.receive_components():
            raise ValueError("staging request targets a component owned by another rank")
        offset, count = payload["offset"], payload["bytes"]
        if (type(offset) is not int or type(count) is not int or offset < 0
                or not 0 < count <= self.window_bytes or not isinstance(payload.get("chunk"), str)
                or not payload["chunk"]):
            raise ValueError("invalid staged chunk geometry or identity")
        region = lease.layers[layer][component]
        row_bytes = region.bytes_per_page // lease.page_size
        logical_byte = start * row_bytes + offset
        logical_page, page_offset = divmod(logical_byte, region.bytes_per_page)
        if (offset + count > (end - start) * row_bytes
                or page_offset + count > region.bytes_per_page):
            raise ValueError("staged chunk exceeds its valid interval or crosses a mapped page")
        destination_offset = lease.page_ids[component][logical_page] * region.bytes_per_page + page_offset
        signature = (lease.lease_id, identity, component, offset, count, payload["chunk"])
        return signature, region, destination_offset

    def acquire(self, lease, payload):
        with self.lock:
            signature, region, destination_offset = self._validate(lease, payload)
            if self.active is not None:
                if self.active["signature"] == signature:
                    return self._grant()
                return {"busy": True}
            _, identity, component, offset, _, _ = signature
            owner_key = (lease.lease_id, identity[2])
            previous = self.layer_owners.get(owner_key)
            if previous is not None and previous != identity and previous not in lease.completed:
                raise ValueError("another generation already owns this staged layer")
            if lease.staged_progress.get(identity, {}).get(component, 0) != offset:
                raise ValueError("staged chunks must consume each component in order")
            self.layer_owners[owner_key] = identity
            lease.staged_intervals[identity] = tuple(payload["range"])
            lease.request_id = identity[1]
            self.generation += 1
            self.active = dict(
                signature=signature, lease=lease, region=region,
                destination_offset=destination_offset, grant=self.generation, event=None,
            )
            return self._grant()

    def _grant(self):
        return dict(busy=False, grant=self.active["grant"], address=self.address,
                    window_bytes=self.window_bytes)

    def commit(self, lease, payload):
        with self.lock:
            signature, _, _ = self._validate(lease, payload)
            receipt_key = (lease.lease_id, signature[1])
            receipt = (signature, payload.get("grant"))
            if self.last_commits.get(receipt_key) == receipt:
                return {"committed": True, "grant": payload["grant"]}
            active = self.active
            if (active is None or active["signature"] != signature
                    or active["grant"] != payload.get("grant")):
                raise ValueError("staged completion does not own the current window credit")
            try:
                active["event"] = self.copy(
                    active["region"], active["destination_offset"], signature[4]
                )
                if active["event"] is None:
                    raise RuntimeError("staging copy did not return a completion event")
                active["event"].synchronize()
                if not active["event"].query():
                    raise RuntimeError("staging copy has not completed")
            except Exception:
                lease.failed = True
                raise
            _, identity, component, offset, count, _ = signature
            lease.staged_progress.setdefault(identity, {})[component] = offset + count
            self.last_commits[receipt_key] = receipt
            self.active = None
            return {"committed": True, "grant": payload["grant"]}

    def abort(self, lease, *, transport_drained):
        if not transport_drained:
            raise RuntimeError("cannot return staging credit before remote writers drain")
        with self.lock:
            lease.failed = True
            if self.active is not None and self.active["lease"] is lease:
                self.drain()
                self.active = None
            lease.staged_progress.clear()
            lease.staged_intervals.clear()

    def forget(self, lease):
        with self.lock:
            if self.active is not None and self.active["lease"] is lease:
                raise RuntimeError("cannot forget a lease owning a staging window")
            for mapping in (self.last_commits, self.layer_owners):
                for key in tuple(mapping):
                    if key[0] == lease.lease_id:
                        del mapping[key]


def create_npu_layer_staging(window_bytes, device_id):
    import torch
    from sglang.srt.hardware_backend.npu.fsa_native import unidex_copy_inplace

    if window_bytes < 32 or window_bytes % 32:
        raise ValueError("NPU staging window must be positive and 32-byte aligned")
    device = f"npu:{device_id}"
    with torch.npu.device(device_id):
        buffer = torch.empty(window_bytes, dtype=torch.uint8, device=device)
        indices = torch.zeros(1, dtype=torch.int64, device=device)
        valid = torch.ones(1, dtype=torch.bool, device=device)
        stream = torch.npu.Stream(device=device_id)
        stream.wait_stream(torch.npu.current_stream())

    def copy(region, offset, count):
        with torch.npu.device(device_id), torch.npu.stream(stream):
            if region.kind == PdSourceKind.HOST_REGISTERED:
                tensor = region.owner.tensor
                destination_address = region.owner.device_ptr + offset
            else:
                tensor = region.owner
                destination_address = tensor.data_ptr() + offset
            for chunk_offset in range(0, count, 32768):
                chunk_bytes = min(32768, count - chunk_offset)
                begin = offset + chunk_offset
                destination = tensor.view(torch.uint8).reshape(-1)[begin:begin + chunk_bytes].view(1, chunk_bytes)
                source = buffer[chunk_offset:chunk_offset + chunk_bytes].view(1, chunk_bytes)
                unidex_copy_inplace(
                    source, destination, indices, indices, valid, 1, 1,
                    dst_ptr=destination_address + chunk_offset,
                )
            event = torch.npu.Event()
            event.record(stream)
            return event

    def drain():
        with torch.npu.device(device_id):
            stream.synchronize()

    return LayerStagingReceiver(
        buffer.data_ptr(), window_bytes, owner=(buffer, indices, valid, stream), copy=copy, drain=drain,
    )


def execute_staged_layer(operation, spans, manifest, payload, *, rpc, transfer, timeout_s):
    if operation.state != "pending":
        raise RuntimeError("staged layer was already submitted")
    operation.state = "transferring"
    operation.transport_drained = True
    deadline = time.monotonic() + timeout_s
    offsets = {}
    try:
        for span in spans:
            consumed = 0
            while consumed < span.nbytes:
                count = min(manifest["window_bytes"], span.nbytes - consumed)
                offset = offsets.get(span.component, 0)
                chunk = {
                    **payload, "component": span.component.value, "offset": offset,
                    "bytes": count, "chunk": uuid4().hex,
                    "staging_registration": manifest["registration"],
                }
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("layer staging window credit timed out")
                    grant = rpc(b"FSA_STAGE_ACQUIRE", chunk, remaining)
                    if not grant.get("busy"):
                        break
                    threading.Event().wait(min(0.002, remaining))
                if (type(grant.get("grant")) is not int or grant["grant"] <= 0
                        or type(grant.get("address")) is not int or grant["address"] <= 0
                        or grant.get("window_bytes") != manifest["window_bytes"]):
                    raise ValueError("receiver granted an invalid staging window")
                status = transfer((span.source_address + consumed, grant["address"], count))
                if status != 0:
                    raise RuntimeError(f"staged transport failed with status {status}")
                receipt = rpc(b"FSA_STAGE_COMMIT", {**chunk, "grant": grant["grant"]},
                              deadline - time.monotonic())
                if not receipt.get("committed") or receipt.get("grant") != grant["grant"]:
                    raise RuntimeError("receiver did not complete the staged local copy")
                consumed += count
                offsets[span.component] = offset + count
        if operation.state == "drained":
            raise RuntimeError("staged layer aborted before final publication")
    except Exception:
        if operation.state != "drained":
            operation.state = "failed"
        raise
    operation.state = "transferred"


def request_layer_staging(manager, peer, header, payload, timeout_s):
    from sglang.srt.disaggregation.base.conn import KVPoll
    from sglang.srt.hardware_backend.npu.layer_protocol import encode
    from sglang.srt.utils.network import NetworkAddress

    if timeout_s <= 0:
        raise TimeoutError("layer staging RPC deadline expired")
    rpc_id = uuid4().hex
    message = {**payload, "rpc": rpc_id, "reply_ip": manager.local_ip, "reply_port": manager.rank_port}
    waiter = dict(event=threading.Event(), result=None, expected=(
        payload["room"], payload["session"], payload["lease"],
        payload["staging_registration"], peer.endpoint, peer.dst_port,
    ))
    deadline = time.monotonic() + timeout_s
    try:
        with manager._layer_send_lock:
            if manager.check_status(payload["room"]) == KVPoll.Failed:
                raise RuntimeError("request failed before staging RPC")
            manager._layer_rpc_waiters[rpc_id] = waiter
            address = NetworkAddress(peer.endpoint, peer.dst_port)
            manager._send_multipart_locked(
                address.to_tcp(), [header, encode(message)], is_ipv6=address.is_ipv6
            )
        while not waiter["event"].wait(max(0, min(0.05, deadline - time.monotonic()))):
            if manager.check_status(payload["room"]) == KVPoll.Failed:
                raise RuntimeError("request aborted while waiting for staging RPC")
            if time.monotonic() >= deadline:
                raise TimeoutError("layer staging RPC timed out")
        result = waiter["result"]
        if result.get("error"):
            raise RuntimeError(f"receiver staging failed: {result['error']}")
        return result
    finally:
        with manager._layer_send_lock:
            manager._layer_rpc_waiters.pop(rpc_id, None)


def handle_layer_staging_message(manager, message):
    import logging

    from sglang.srt.disaggregation.base.conn import KVPoll
    from sglang.srt.hardware_backend.npu.layer_protocol import decode, encode
    from sglang.srt.utils.network import NetworkAddress

    logger = logging.getLogger(__name__)
    payload = None
    lease = None
    try:
        if len(message) != 2:
            raise ValueError("invalid staging frame count")
        payload = decode(message[1])
        if message[0] == b"FSA_STAGE_REPLY":
            with manager._layer_send_lock:
                waiter = manager._layer_rpc_waiters.get(payload["rpc"])
                if waiter is None:
                    return
                identity = (payload["room"], payload["session"], payload["lease"],
                            payload["staging_registration"], payload["reply_ip"], payload["reply_port"])
                if identity != waiter["expected"]:
                    raise ValueError("staging reply does not match its receiver lease")
                waiter["result"] = payload
                waiter["event"].set()
            return
        if message[0] not in (b"FSA_STAGE_ACQUIRE", b"FSA_STAGE_COMMIT"):
            raise ValueError("unknown staging control message")
        lease = manager._layer_leases[payload["room"]]
        if manager.check_status(payload["room"]) == KVPoll.Failed:
            raise RuntimeError("staging request arrived after request failure")
        receiver = manager._layer_staging
        if receiver is None:
            raise RuntimeError("receiver has no negotiated layer staging window")
        result = (receiver.acquire(lease, payload) if message[0] == b"FSA_STAGE_ACQUIRE"
                  else receiver.commit(lease, payload))
    except Exception as exc:
        logger.exception("NPU staged layer control failed")
        if payload is None or "room" not in payload:
            return
        manager.record_failure(payload["room"], str(exc))
        manager.update_status(payload["room"], KVPoll.Failed)
        if lease is not None:
            lease.failed = True
        if message[0] == b"FSA_STAGE_REPLY":
            return
        result = {"error": str(exc)}
    try:
        address = NetworkAddress(payload["reply_ip"], payload["reply_port"])
        reply = {**payload, **result, "reply_ip": manager.local_ip, "reply_port": manager.rank_port}
        manager._send_multipart_locked(address.to_tcp(), [b"FSA_STAGE_REPLY", encode(reply)],
                                      is_ipv6=address.is_ipv6)
    except Exception:
        logger.exception("Could not send layer staging response")
