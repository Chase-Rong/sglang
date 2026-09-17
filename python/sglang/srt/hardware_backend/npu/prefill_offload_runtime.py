"""Batch ownership and real KV-pool boundaries for layerwise Prefill offload."""

import contextlib
import time
from dataclasses import dataclass, field, replace

import torch

from sglang.srt.hardware_backend.npu.prefill_offload import (
    HostKvPagePool,
    KvPageRange,
    LayerSlotToken,
    LayerWork,
    PrefillLayerSlotManager,
    SlotBusyError,
)


def validate_prefill_parallel_layout(
    *, cp_size, dcp_size, replicated_cp, layer_split, all_cp_transfer
):
    if dcp_size != 1 or layer_split:
        raise ValueError(
            "bounded NPU Prefill requires replicated KV, not DCP or layer-sharded KV"
        )
    if cp_size > 1 and not replicated_cp:
        raise ValueError(
            "bounded NPU Prefill CP requires DSA all-gather KV reconstruction"
        )
    if cp_size > 1 and all_cp_transfer:
        raise ValueError(
            "replicated bounded Prefill KV uses CP rank 0; disable all-CP transfer"
        )


@dataclass
class PrefillRequestRows:
    allocation: object
    prefix_length: int
    sequence_length: int
    token_indices: torch.Tensor
    device_rows: torch.Tensor
    source_pages: tuple
    sender: object


@dataclass
class PrefillBatchLayer:
    slot_token: LayerSlotToken
    works: list = field(default_factory=list)
    finished: bool = False
    published: bool = False
    transfers: list = field(default_factory=list)
    failed: bool = False


class NpuPrefillOffloadRuntime:
    """Own one physical layer slot for all requests in a Prefill batch.

    ``publish`` returns operations whose state becomes ``acknowledged`` only
    after receiver visibility is confirmed. A failed or timed-out operation
    retains all layer bindings; cancellation must explicitly drain it.
    """

    def __init__(self, pool, request_pool, adapter, *, publish, timeout_s=30.0):
        if pool.prefill_layer_slots is None:
            raise ValueError("Prefill runtime requires a bounded physical layer pool")
        if timeout_s <= 0:
            raise ValueError("Prefill drain timeout must be positive")
        self.pool = pool
        self.request_pool = request_pool
        self.adapter = adapter
        self.publish = publish
        self.timeout_s = timeout_s
        self.slots = PrefillLayerSlotManager(pool.layer_num, pool.prefill_layer_slots)
        self.host_pages = HostKvPagePool(
            pool.size // pool.page_size + 1, pool.page_size
        )
        self.requests = ()
        self.layers = {}
        self.batch_id = 0
        self.active = False
        self.failed = False
        self.completed_layers = set()
        self.failure_event = None

    def register_host_layers(self):
        rows = self.host_pages.page_count * self.pool.page_size
        for layer_id in range(
            self.pool.start_layer, self.pool.start_layer + self.pool.layer_num
        ):
            self.adapter.register_layer_components(
                layer_id,
                latent_shape=(rows, self.pool.kv_lora_rank),
                rope_shape=(rows, self.pool.qk_rope_head_dim),
                indexer_shape=(
                    None
                    if self.pool.index_head_dim is None
                    or getattr(self.pool, "_kpool_use_compress", False)
                    else (rows, self.pool.index_head_dim)
                ),
                dtype=self.pool.store_dtype,
            )

    @contextlib.contextmanager
    def forward(self, batch):
        self.begin_batch(batch)
        try:
            yield
            self.finish_batch()
        except BaseException:
            self.failed = True
            self.failure_event = self.adapter.record_event()
            for layer in self.layers.values():
                layer.failed = True
                self.slots.mark_failed(layer.slot_token)
            raise

    def begin_batch(self, batch):
        if self.active or self.layers or self.failed:
            raise SlotBusyError("previous Prefill batch has not drained")
        mode = batch.forward_mode
        if not mode.is_extend() or mode.is_target_verify() or mode.is_draft_extend_v2():
            raise ValueError("bounded Prefill runtime requires ordinary eager extend")
        request_ids = tuple(batch.rids or ())
        if not request_ids or len(set(request_ids)) != len(request_ids):
            raise ValueError("Prefill batch requires distinct request IDs")
        if batch.seq_lens_cpu is None or batch.extend_prefix_lens_cpu is None:
            raise ValueError(
                "Prefill offload requires CPU sequence and prefix metadata"
            )
        lengths = [int(value) for value in batch.seq_lens_cpu]
        prefixes = [int(value) for value in batch.extend_prefix_lens_cpu]
        request_slots = batch.req_pool_indices.detach().cpu().tolist()
        senders = batch.prefill_offload_senders
        if not (
            len(lengths)
            == len(prefixes)
            == len(request_slots)
            == len(request_ids)
            == len(senders or ())
        ):
            raise ValueError("Prefill request metadata counts must match")
        for prefix, length, slot, sender in zip(
            prefixes, lengths, request_slots, senders
        ):
            if not 0 <= prefix < length <= self.request_pool.req_to_token.shape[1]:
                raise ValueError("invalid Prefill token interval")
            if not 0 < slot < self.request_pool.req_to_token.shape[0]:
                raise ValueError("invalid Prefill request pool slot")
            if sender is None:
                raise RuntimeError("Prefill request has no layer-ready sender")
            sender.require_prefill_layer_targets(
                [
                    layer + getattr(self.pool, "fsa_transfer_layer_offset", 0)
                    for layer in self.pool.fsa_wire_layer_ids
                ]
                if hasattr(self.pool, "fsa_wire_layer_ids")
                else range(
                    self.pool.start_layer
                    + getattr(self.pool, "fsa_transfer_layer_offset", 0),
                    self.pool.start_layer
                    + self.pool.layer_num
                    + getattr(self.pool, "fsa_transfer_layer_offset", 0),
                )
            )
        prepared = []
        newly_allocated = []
        occupied_device_pages = set()
        try:
            for request_id, length, prefix, slot, sender in zip(
                request_ids, lengths, prefixes, request_slots, senders
            ):
                try:
                    allocation = self.host_pages.allocation(request_id)
                except KeyError:
                    if prefix:
                        raise RuntimeError(
                            "cannot restore an unowned Prefill prefix"
                        ) from None
                    allocation = self.host_pages.allocate(request_id, length)
                    newly_allocated.append(allocation)
                else:
                    allocation = self.host_pages.grow(request_id, length)
                rows = (
                    self.request_pool.req_to_token[slot, :length]
                    .to(torch.int64)
                    .contiguous()
                )
                row_list = rows.detach().cpu().tolist()
                capacity = (
                    self.pool.size // self.pool.page_size + 1
                ) * self.pool.page_size
                if any(row < 0 or row >= capacity for row in row_list):
                    raise ValueError(
                        "request mapping exceeds physical KV token capacity"
                    )
                source_pages = tuple(
                    row // self.pool.page_size
                    for row in row_list[:: self.pool.page_size]
                )
                if len(set(source_pages)) != len(
                    source_pages
                ) or occupied_device_pages.intersection(source_pages):
                    raise ValueError(
                        "Prefill requests must own disjoint physical device pages"
                    )
                occupied_device_pages.update(source_pages)
                if any(
                    row
                    != source_pages[index // self.pool.page_size] * self.pool.page_size
                    + index % self.pool.page_size
                    for index, row in enumerate(row_list)
                ):
                    raise ValueError(
                        "Prefill transfer requires page-aligned request token mapping"
                    )
                prepared.append(
                    PrefillRequestRows(
                        allocation,
                        prefix,
                        length,
                        torch.arange(length, device=rows.device, dtype=torch.int64),
                        rows,
                        source_pages,
                        sender,
                    )
                )
        except BaseException:
            for allocation in newly_allocated:
                self.host_pages.release(allocation)
            raise
        self.batch_id += 1
        self.requests = tuple(prepared)
        self.completed_layers.clear()
        self.active = True

    def prepare_layer(self, layer_id):
        if not self.active or self.failed:
            raise RuntimeError("KV access has no active healthy Prefill batch")
        if layer_id in self.completed_layers:
            raise RuntimeError(
                "Prefill layer was accessed after its batch work completed"
            )
        if layer_id in self.layers:
            if self.layers[layer_id].finished:
                raise RuntimeError(
                    "Prefill layer was accessed after attention completion"
                )
            return
        local_layer = layer_id - self.pool.start_layer
        slot_id = self.slots.preferred_slot(local_layer)
        for previous_id, previous in tuple(self.layers.items()):
            if previous.slot_token.slot_id == slot_id:
                self.drain_layer(previous_id)
        token = self.slots.acquire(f"batch:{self.batch_id}", local_layer)
        layer = PrefillBatchLayer(token)
        self.layers[layer_id] = layer
        tensors = self.pool.raw_layer_components(layer_id)
        if getattr(self.pool, "_kpool_use_compress", False):
            tensors = (tensors[0], tensors[1], None)
        try:
            for request in self.requests:
                work = LayerWork(
                    LayerSlotToken(
                        slot_id,
                        request.allocation.request_id,
                        layer_id,
                        token.generation,
                    ),
                    request.allocation,
                    KvPageRange(request.prefix_length, request.sequence_length),
                )
                self.adapter.bind_layer(
                    work,
                    *tensors,
                    token_indices=request.token_indices,
                    device_row_indices=request.device_rows,
                    source_page_ids=request.source_pages,
                )
                layer.works.append(work)
                self.adapter.restore_layer(work)
                work.restored = True
        except BaseException:
            layer.failed = True
            self.failed = True
            self.slots.mark_failed(token)
            raise

    def finish_layer(self, layer_id):
        layer = self.layers[layer_id]
        if layer.finished or layer.failed:
            raise RuntimeError("Prefill layer cannot complete twice or after failure")
        for work in layer.works:
            work.kv_written = True
            self.adapter.record_attention_done(work)
            work.host_save_started = True
            self.adapter.save_layer(work)
        layer.finished = True

    def _wait(self, predicate, deadline, reason):
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError(reason)
            time.sleep(0.001)

    def drain_layer(self, layer_id):
        layer = self.layers[layer_id]
        if not layer.finished or layer.failed:
            raise SlotBusyError(
                "cannot reuse a layer before successful attention completion"
            )
        deadline = time.monotonic() + self.timeout_s
        for work in layer.works:
            self.adapter.wait_local_completion(work, deadline=deadline)
            self.adapter.complete_host_save(work)
            self.adapter.complete_attention(work)
        self.slots.mark_host_saved(layer.slot_token)
        self.slots.mark_attention_done(layer.slot_token)
        if not layer.published:
            layer.published = True
            for request, work in zip(self.requests, layer.works):
                operations = self.publish(
                    request.sender,
                    work,
                    self.adapter.binding(work),
                    self.adapter.source_ready_event(work),
                )
                layer.transfers.extend(operations)
            self.slots.set_pd_readers(layer.slot_token, len(layer.transfers))

        def acknowledged():
            if any(
                operation.state in ("failed", "drained")
                for operation in layer.transfers
            ):
                layer.failed = True
                self.slots.mark_failed(layer.slot_token)
                raise RuntimeError("Prefill layer transfer failed")
            return all(
                operation.state == "acknowledged" for operation in layer.transfers
            )

        self._wait(acknowledged, deadline, "Prefill layer receiver ACK timed out")
        for _ in layer.transfers:
            self.slots.mark_pd_read_done(layer.slot_token)
        for work in layer.works:
            self.adapter.validate_layer_release(work)
        for work in layer.works:
            self.adapter.release_layer(work)
        self.slots.release(layer.slot_token)
        del self.layers[layer_id]
        self.completed_layers.add(layer_id)

    def finish_batch(self):
        for layer_id in tuple(self.layers):
            self.drain_layer(layer_id)
        if len(self.completed_layers) != self.pool.layer_num:
            raise RuntimeError(
                "Prefill batch did not execute every local attention layer"
            )
        self.requests = ()
        self.active = False

    def can_release_request(self, request_id):
        return not (
            any(
                request.allocation.request_id == request_id for request in self.requests
            )
            or any(
                work.token.request_id == request_id
                for layer in self.layers.values()
                for work in layer.works
            )
        )

    def release_request(self, request_id):
        if not self.can_release_request(request_id):
            raise SlotBusyError("request release requires its Prefill work to drain")
        try:
            allocation = self.host_pages.allocation(request_id)
        except KeyError:
            return
        self.adapter.release_request(allocation)
        self.host_pages.release(allocation)

    def drain_failed_batch(self):
        if not self.failed or self.failure_event is None:
            raise RuntimeError("failed-batch drain requires a recorded failure event")
        deadline = time.monotonic() + self.timeout_s
        self._wait(
            self.failure_event.query,
            deadline,
            "failed Prefill device work has not drained",
        )
        offset = getattr(self.pool, "fsa_transfer_layer_offset", 0)
        tokens = {
            replace(work.token, logical_layer=work.token.logical_layer + offset)
            for layer in self.layers.values()
            for work in layer.works
        }
        for layer in self.layers.values():
            if any(
                operation.state not in ("acknowledged", "drained")
                for operation in layer.transfers
            ):
                raise SlotBusyError("failed Prefill transfer has not drained")
        for request in self.requests:
            for operation in request.sender.prefill_layer_operations():
                if operation.record.token in tokens and operation.state not in (
                    "acknowledged",
                    "drained",
                ):
                    raise SlotBusyError("failed Prefill transfer has not drained")
        for layer in self.layers.values():
            for work in layer.works:
                self.adapter.wait_local_completion(work, deadline=deadline)
                self.adapter.release_layer(work)
            self.slots.mark_failed(layer.slot_token)
            self.slots.mark_failed_local_work_drained(layer.slot_token)
            readers = self.slots.pending_pd_readers(layer.slot_token)
            if readers:
                self.slots.mark_pd_read_done(layer.slot_token, readers)
            self.slots.release(layer.slot_token)
        self.layers.clear()
        self.requests = ()
        self.active = False
        self.failed = False
        self.failure_event = None


class PrefillReleaseCheck:
    def __init__(self, runtimes, request_id, sender):
        self.runtimes = runtimes
        self.request_id = request_id
        self.sender = sender

    def poll(self):
        from sglang.srt.disaggregation.base.conn import KVPoll

        if any(
            not runtime.can_release_request(self.request_id)
            for runtime in self.runtimes
        ):
            return KVPoll.WaitingForInput
        if any(
            operation.state not in ("acknowledged", "drained")
            for operation in self.sender.prefill_layer_operations()
        ):
            return KVPoll.WaitingForInput
        return KVPoll.Success
