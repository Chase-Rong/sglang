"""Stream-ordered prefix restores and slot-backed writes for integrated Prefill."""

import contextlib
from collections import deque

import torch


class PipelinedIntegratedOffloadRuntime:
    def __init__(self, pool, request_pool, *, h2d_prefetch=True, d2h_async=True):
        if not pool.integrated_offload or pool.decode_host_backing is None:
            raise ValueError(
                "integrated runtime requires a shared Prefill/Decode backing"
            )
        self.pool = pool
        self.request_pool = request_pool
        self.active = False
        self.failed = False
        self.prepared = set()
        self.completed = set()
        self.prefix_rows = None
        self.slot_events = {}
        self.slot_layers = {}
        self.h2d_prefetch = h2d_prefetch and pool.prefill_layer_slots > 1
        self.d2h_async = d2h_async
        self.h2d_stream = None
        self.d2h_stream = None
        self.compute_stream = None
        self.host_visible = None
        self.slot_saves = {}
        self.layers = {}
        self.retired = deque()
        self.generation = 0
        self.next_compute_layer = pool.start_layer

    @staticmethod
    def _record(stream):
        event = torch.npu.Event()
        event.record(stream)
        return event

    def _check_stream(self):
        if torch.npu.current_stream() != self.compute_stream:
            raise RuntimeError(
                "Prefill layer hooks must use the forward compute stream"
            )

    def _check_layer(self, layer_id):
        if self.failed or not self.active:
            raise RuntimeError("Prefill pipeline has no healthy active generation")
        self.pool.physical_layer_index(layer_id)
        self._check_stream()

    def _reap(self):
        while self.retired and self.retired[0][0].query():
            self.retired.popleft()
        if len(self.retired) >= 2:
            self.retired[0][0].synchronize()
            self.retired.popleft()

    @contextlib.contextmanager
    def forward(self, batch):
        if self.failed or self.active:
            raise RuntimeError("Prefill pipeline is failed or already active")
        if self.pool.decode_host_backing.page_binding is None:
            raise RuntimeError("Prefill pipeline requires HiCache page ownership")
        self.compute_stream = torch.npu.current_stream()
        if not batch.forward_mode.is_extend_without_speculative():
            if self.host_visible is not None:
                self.compute_stream.wait_event(self.host_visible)
            try:
                yield
            finally:
                self.pool.decode_host_backing.page_binding.ready_event = self._record(
                    self.compute_stream
                )
            return
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Pipelined Prefill does not support graph capture")
        self._reap()
        if self.host_visible is not None:
            self.compute_stream.wait_event(self.host_visible)
        prefixes = batch.extend_prefix_lens_cpu
        if prefixes is None or len(prefixes) != batch.req_pool_indices.numel():
            raise ValueError("Prefill pipeline requires per-request prefix lengths")
        rows = []
        for slot, prefix in zip(batch.req_pool_indices.cpu().tolist(), prefixes):
            if not 0 <= slot < self.request_pool.req_to_token.shape[0]:
                raise ValueError("Prefill request slot is outside the mapping")
            if not 0 <= prefix <= self.request_pool.req_to_token.shape[1]:
                raise ValueError("Prefill prefix length is outside the mapping")
            rows.append(self.request_pool.req_to_token[slot, :prefix].to(torch.int64))
        self.prefix_rows = (
            torch.cat(rows)
            if rows
            else batch.out_cache_loc.new_empty(0, dtype=torch.int64)
        )
        backing = self.pool.decode_host_backing
        self.new_rows = batch.out_cache_loc.reshape(-1).to(dtype=torch.int64).clone()
        self.new_locations_ptr = batch.out_cache_loc.data_ptr()
        self.new_valid = (self.new_rows > 0) & (
            self.new_rows < backing.pages * backing.page_size
        )
        self.prefix_plan, self.new_plan = backing.plan_prefill_rows(
            self.prefix_rows, self.new_rows
        )
        self.inputs_ready = self._record(self.compute_stream)
        if self.h2d_prefetch and self.h2d_stream is None:
            self.h2d_stream = torch.npu.Stream(device=self.pool.k_buffer.device)
        if self.d2h_async and self.d2h_stream is None:
            self.d2h_stream = torch.npu.Stream(device=self.pool.k_buffer.device)
        self.layers = {}
        self.prepared = set()
        self.completed = set()
        self.slot_layers = {}
        self.next_compute_layer = self.pool.start_layer
        self.generation += 1
        self.active = True
        try:
            if self.h2d_prefetch:
                self._restore(self.pool.start_layer, self.inputs_ready)
            yield
            if len(self.completed) != self.pool.layer_num:
                raise RuntimeError("Prefill pipeline did not finish every DSA layer")
            for layer_id in range(
                self.pool.start_layer, self.pool.start_layer + self.pool.layer_num
            ):
                state = self.layers[layer_id]
                if self.d2h_async:
                    if "save_done" not in state:
                        raise RuntimeError(
                            "Prefill layer never submitted its Host write"
                        )
                    self.compute_stream.wait_event(state["save_done"])
            if self.d2h_async:
                backing.publish_prefill_writes()
            self.host_visible = self._record(self.compute_stream)
            backing.page_binding.ready_event = self.host_visible
            self.retired.append(
                (
                    self.host_visible,
                    (
                        self.layers,
                        self.prefix_rows,
                        self.new_rows,
                        self.new_valid,
                        self.prefix_plan,
                        self.new_plan,
                    ),
                )
            )
        except BaseException as error:
            self.failed = True
            try:
                self.drain()
            except BaseException as drain_error:
                raise error from drain_error
            raise
        finally:
            self.active = False

    def _restore(self, layer_id, gate=None):
        if layer_id in self.layers:
            return
        physical = self.pool.physical_layer_index(layer_id)
        previous = self.slot_layers.get(physical)
        if previous is not None and previous not in self.completed:
            raise RuntimeError("Prefill restore would overwrite an active layer")
        stream = self.h2d_stream if self.h2d_prefetch else self.compute_stream
        with torch.npu.stream(stream):
            stream.wait_event(self.inputs_ready)
            if gate is not None:
                stream.wait_event(gate)
            for events in (self.slot_events, self.slot_saves):
                if physical in events:
                    stream.wait_event(events[physical])
            backing = self.pool.decode_host_backing
            local_layer = layer_id - self.pool.start_layer
            if self.prefix_rows.numel():
                for name, target, width in zip(
                    backing._component_names[local_layer],
                    self.pool.raw_layer_components(layer_id)[:2],
                    (self.pool.kv_lora_rank, self.pool.qk_rope_head_dim),
                ):
                    if width:
                        backing.pool.copy_prefill_from_host(
                            name, target.view(-1, width), self.prefix_plan
                        )
            ready = self._record(stream)
        self.layers[layer_id] = {"ready": ready}
        self.slot_layers[physical] = layer_id

    def prepare_layer(self, layer_id):
        self._check_layer(layer_id)
        if layer_id in self.completed:
            raise RuntimeError("Prefill accessed a completed layer")
        if layer_id in self.prepared:
            return
        if self.h2d_prefetch and layer_id not in self.layers:
            raise RuntimeError("Prefill layer was not prefetched by its predecessor")
        self._restore(layer_id)
        self.compute_stream.wait_event(self.layers[layer_id]["ready"])
        self.prepared.add(layer_id)

    def validate_write(self, layer_id):
        self._check_layer(layer_id)
        if layer_id in self.completed or "save_done" in self.layers.get(layer_id, {}):
            raise RuntimeError("Duplicate or late Prefill Host write")

    def on_kv_written(self, layer_id, locations):
        self.validate_write(layer_id)
        if layer_id not in self.prepared:
            raise RuntimeError("Prefill scatter preceded layer preparation")
        if not self.d2h_async:
            return
        backing = self.pool.decode_host_backing
        if (
            locations.data_ptr() == self.new_locations_ptr
            and locations.numel() == self.new_rows.numel()
        ):
            rows, valid = self.new_rows, self.new_valid
            plan = self.new_plan
        else:
            rows = locations.reshape(-1).to(dtype=torch.int64).clone()
            valid = (rows > 0) & (rows < backing.pages * backing.page_size)
            plan = backing.plan_prefill_rows(rows[:0], rows)[1]
        written = self._record(self.compute_stream)
        with torch.npu.stream(self.d2h_stream):
            self.d2h_stream.wait_event(written)
            if rows.numel():
                backing.enqueue_prefill_rows_from_slot(
                    layer_id - self.pool.start_layer,
                    self.pool.raw_layer_components(layer_id)[:2],
                    rows,
                    valid,
                    plan=plan,
                )
            saved = self._record(self.d2h_stream)
        self.layers[layer_id].update(
            save_done=saved, written=written, rows=rows, valid=valid, plan=plan
        )
        self.slot_saves[self.pool.physical_layer_index(layer_id)] = saved

    def on_compute_start(self, layer_id):
        self._check_layer(layer_id)
        if layer_id != self.next_compute_layer or layer_id not in self.prepared:
            raise RuntimeError("Prefill compute hooks are out of DSA layer order")
        if self.d2h_async and "save_done" not in self.layers[layer_id]:
            if self.new_rows.numel():
                raise RuntimeError("Prefill compute started without a KV-written hook")
            self.on_kv_written(layer_id, self.new_rows)
        started = self._record(self.compute_stream)
        self.layers[layer_id]["started"] = started
        self.next_compute_layer += 1
        if (
            self.h2d_prefetch
            and self.next_compute_layer < self.pool.start_layer + self.pool.layer_num
        ):
            self._restore(self.next_compute_layer, started)

    def finish_layer(self, layer_id):
        self._check_layer(layer_id)
        if layer_id in self.completed or "started" not in self.layers.get(layer_id, {}):
            raise RuntimeError("Prefill finish lacks its compute-start hook")
        event = self._record(self.compute_stream)
        self.layers[layer_id]["readers_done"] = event
        self.slot_events[self.pool.physical_layer_index(layer_id)] = event
        self.completed.add(layer_id)

    def drain(self):
        for stream in (self.compute_stream, self.h2d_stream, self.d2h_stream):
            if stream is not None:
                stream.synchronize()
        self.retired.clear()
        self.host_visible = None

    def release_request(self, request_id):
        if self.active:
            raise RuntimeError(
                "Cannot release a request during an active Prefill generation"
            )
        self.drain()
