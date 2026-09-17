"""One service shares logical Host KV between bounded Prefill and fused Decode."""

import contextlib

import torch


class IntegratedOffloadRuntime:
    def __init__(self, pool, request_pool):
        if not pool.integrated_offload or pool.decode_host_backing is None:
            raise ValueError("integrated runtime requires a shared Prefill/Decode backing")
        self.pool = pool
        self.request_pool = request_pool
        self.active = False
        self.failed = False
        self.prepared = set()
        self.completed = set()
        self.prefix_rows = None
        self.slot_events = {}
        self.slot_layers = {}

    @contextlib.contextmanager
    def forward(self, batch):
        if self.failed or self.active:
            raise RuntimeError("integrated offload runtime is failed or already active")
        for event in self.slot_events.values():
            torch.npu.current_stream().wait_event(event)
        if not batch.forward_mode.is_extend_without_speculative():
            yield
            return
        prefixes = batch.extend_prefix_lens_cpu
        if prefixes is None or len(prefixes) != batch.req_pool_indices.numel():
            raise ValueError("integrated Prefill requires per-request prefix lengths")
        slots = batch.req_pool_indices.cpu().tolist()
        rows = []
        for slot, prefix in zip(slots, prefixes):
            if not 0 <= slot < self.request_pool.req_to_token.shape[0]:
                raise ValueError("integrated Prefill request slot is outside the pool")
            if not 0 <= prefix <= self.request_pool.req_to_token.shape[1]:
                raise ValueError("integrated Prefill prefix exceeds token mapping")
            rows.append(self.request_pool.req_to_token[slot, :prefix].to(torch.int64))
        self.prefix_rows = torch.cat(rows) if rows else batch.out_cache_loc.new_empty(0, dtype=torch.int64)
        self.prepared.clear()
        self.completed.clear()
        self.slot_layers.clear()
        self.active = True
        try:
            yield
            if len(self.completed) != self.pool.layer_num:
                raise RuntimeError("integrated Prefill did not finish every DSA layer")
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False
            self.prefix_rows = None

    def prepare_layer(self, layer_id):
        if not self.active or self.failed or layer_id in self.completed:
            raise RuntimeError("invalid integrated Prefill layer access")
        if layer_id in self.prepared:
            return
        physical = self.pool.physical_layer_index(layer_id)
        previous = self.slot_layers.get(physical)
        if previous is not None and previous not in self.completed:
            raise RuntimeError("cannot reuse a Prefill slot before attention finishes")
        if physical in self.slot_events:
            torch.npu.current_stream().wait_event(self.slot_events[physical])
        backing = self.pool.decode_host_backing
        local_layer = layer_id - self.pool.start_layer
        if self.prefix_rows.numel():
            valid = (self.prefix_rows > 0) & (self.prefix_rows < backing.pages * backing.page_size)
            for name, target, width in zip(
                backing._component_names[local_layer], self.pool.raw_layer_components(layer_id)[:2],
                (self.pool.kv_lora_rank, self.pool.qk_rope_head_dim),
            ):
                if width:
                    backing.pool.copy_from_host(name, target.view(-1, width),
                                                self.prefix_rows, self.prefix_rows, valid)
        self.slot_layers[physical] = layer_id
        self.prepared.add(layer_id)

    def finish_layer(self, layer_id):
        if not self.active or layer_id not in self.prepared or layer_id in self.completed:
            raise RuntimeError("integrated Prefill layer completion is unbalanced")
        event = torch.npu.Event()
        event.record()
        self.slot_events[self.pool.physical_layer_index(layer_id)] = event
        self.completed.add(layer_id)
