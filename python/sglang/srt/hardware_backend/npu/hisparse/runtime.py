"""HiSparse lifecycle over the NPU resources already owned by ModelRunner."""

import contextlib
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch


@dataclass
class _RetiredRequest:
    kv: object
    retractions: int
    slot: int


class NpuHiSparseCoordinator:
    """Compose layerwise runtimes while HiCache owns request and cache pages.

    Request generations remain in ReqToTokenPool. Retiring a request changes
    its planner identity, while release_kv_cache/allocator retain the only
    path that releases its HiCache references. Backing shutdown stays with
    the existing backing owner, closed after all service users drain.
    """

    uses_staging = False

    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.mem_pool_device = model_runner.token_to_kv_pool
        self.device = model_runner.device
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device="cpu")
        self.ack_staging_queue = []
        self.decode_producer_stream = None
        self._retired_requests = WeakKeyDictionary()
        self._prefill_initialized = False
        self._destroyed = False
        self._shutdown_prepared = False
        self._last_publication = None

    @property
    def prefill_offload_runtime(self):
        return getattr(self.model_runner, "prefill_offload_runtime", None)

    @property
    def _backing(self):
        pool = getattr(self.mem_pool_device, "full_kv_pool", self.mem_pool_device)
        return getattr(pool, "decode_host_backing", None)

    def initialize_prefill(self):
        if self._prefill_initialized or self.prefill_offload_runtime is not None:
            self._prefill_initialized = True
            return self.prefill_offload_runtime
        from sglang.srt.hardware_backend.npu.hisparse.prefill import (
            initialize_prefill_runtime,
        )

        initialize_prefill_runtime(self.model_runner)
        self._prefill_initialized = True
        return self.prefill_offload_runtime

    def bind_prefill_runtime(self, runtime) -> None:
        current = self.prefill_offload_runtime
        if current is not None and current is not runtime:
            raise RuntimeError("NPU HiSparse already has a Prefill runtime")
        self.model_runner.prefill_offload_runtime = runtime
        self.mem_pool_device.prefill_offload_runtime = runtime

    def prefill_offload_forward(self, forward_batch):
        runtime = self.prefill_offload_runtime
        if runtime is None or forward_batch.forward_mode.is_idle():
            if not forward_batch.forward_mode.is_idle():
                self.wait_for_pending_backup()
            return contextlib.nullcontext()
        return runtime.forward(forward_batch)

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def wait_for_pending_backup(self) -> None:
        """Wait for existing layer publications, without another KV backup."""
        if self.prefill_offload_runtime is not None:
            return
        binding = getattr(self._backing, "page_binding", None)
        event = getattr(binding, "ready_event", None)
        stream = self.decode_producer_stream
        if event is not None:
            if stream is None:
                stream = torch.npu.current_stream()
            if self._last_publication != (event, stream):
                stream.wait_event(event)
                self._last_publication = (event, stream)

    def map_last_loc_to_buffer(
        self,
        seq_lens,
        out_cache_loc,
        req_pool_indices,
        seq_lens_cpu,
        req_pool_indices_cpu,
    ) -> None:
        """NPU pool writes already publish current KV at its original location."""

    def admit_request_direct(self, req) -> None:
        req.hisparse_staging = False

    def admit_request_into_staging(self, req) -> None:
        self.admit_request_direct(req)

    def warmup_publication_committed_length(self, req):
        """Read committed authority only for an enabled, supported owner.

        Disabled PD receive keeps its existing request interface. Unknown
        committed metadata cannot authorize a warmup publication snapshot.
        """
        state = getattr(self.mem_pool_device, "fsa_runtime_state", None)
        if (
            state is None
            or getattr(state, "execution_abi", None) != "shared_v3"
            or not getattr(state, "warmup_threshold", 0)
        ):
            return None
        kv = getattr(req, "kv", None)
        committed = getattr(kv, "kv_committed_len", None)
        return committed if type(committed) is int and committed >= 0 else None

    def record_local_warmup_publication(self, requests):
        """Consume a successful integrated Prefill all-layer coverage snapshot."""
        state = getattr(self.mem_pool_device, "fsa_runtime_state", None)
        if state is None or not getattr(state, "warmup_threshold", 0):
            return
        publication = getattr(self.prefill_offload_runtime, "warmup_publication", None)
        if publication is None:
            return
        slots, generations, lengths, epoch, event = publication
        binding = getattr(self._backing, "page_binding", None)
        if binding is None or binding.mapping_epoch != epoch:
            return
        coverage = {
            slot: (generation, length)
            for slot, generation, length in zip(slots, generations, lengths)
        }
        for req in requests:
            owner = req.kv.req_pool_idx
            if owner not in coverage or req.finished():
                continue
            generation, length = coverage[owner]
            if int(self.req_to_token_pool.req_generation[owner]) != generation:
                continue
            committed = int(req.kv.kv_committed_len)
            if 0 < committed <= length:
                self.record_warmup_publication(req, committed, publication_event=event)

    def record_warmup_publication(
        self, req, received_committed_length, *, publication_event=None
    ):
        """Bind a successful PD receive publication to its exact owned prefix.

        Local Prefill without this per-request coverage proof remains lazy.
        The existing receive-success consensus covers all physical layers.
        """
        state = getattr(self.mem_pool_device, "fsa_runtime_state", None)
        binding = getattr(self._backing, "page_binding", None)
        if (
            state is None
            or not getattr(state, "warmup_threshold", 0)
            or binding is None
            or binding.ready_event is None
        ):
            return
        owner = req.kv.req_pool_idx
        if owner is None or not 0 < owner < state.generations.numel():
            return
        if not hasattr(self, "_warmup_publications"):
            self._warmup_publications = torch.full(
                (state.generations.numel(), 4), -1, dtype=torch.int64, device="cpu"
            )
            self._warmup_publication_events = {}
        state.refresh_generations()
        self._warmup_publications[owner] = torch.tensor(
            [
                int(state.generations[owner]),
                binding.mapping_epoch,
                received_committed_length,
                id(req.kv),
            ],
            dtype=torch.int64,
        )
        self._warmup_publication_events[owner] = (
            publication_event if publication_event is not None else binding.ready_event
        )

    def warmup_committed_requests(self, requests):
        """Scheduler-only eager boundary using authoritative Req KV snapshots.

        Warm one owner per microbatch on the actual forward stream. Admission
        can precede publication; no work is submitted until ordinary Decode.
        """
        state = getattr(self.mem_pool_device, "fsa_runtime_state", None)
        if state is None or not getattr(state, "warmup_threshold", 0):
            return 0
        binding = getattr(self._backing, "page_binding", None)
        if binding is None or binding.ready_event is None:
            return 0
        from .shared_warmup import warm_committed_prefix
        from .fused_sparse_attention import paged_kv_view
        from .fsa_backend import sparse_layer_ids, sparse_pool_and_layer

        warmed = 0
        for req in requests:
            if req.finished() or req.kv.req_pool_idx is None:
                continue
            owner = int(req.kv.req_pool_idx)
            # Never substitute batch.seq_lens, allocated_len or verify width.
            committed = int(req.kv.kv_committed_len)
            if not 0 < committed <= state.warmup_threshold:
                continue
            if not 0 < owner < self.req_to_token_pool.req_generation.numel():
                continue
            state.refresh_generations()
            generation = int(state.generations[owner])
            epoch = binding.mapping_epoch
            publications = getattr(self, "_warmup_publications", None)
            if publications is None:
                continue
            published = publications[owner]
            if (
                int(published[0]) != generation
                or int(published[1]) != epoch
                or int(published[2]) < committed
                or int(published[3]) != id(req.kv)
            ):
                continue
            # Keep the event attached to this exact snapshot; a later unrelated
            # binding.ready_event alone cannot extend its coverage.
            publication_event = self._warmup_publication_events.get(owner)
            if publication_event is None:
                continue
            if state.device.type == "npu":
                torch.npu.current_stream().wait_event(publication_event)
            # Read only complete logical prefix pages, with explicit invalid
            # mapping rejection instead of the ordinary padding-to-zero helper.
            tokens = self.req_to_token_pool.req_to_token[owner, :committed:128].long()
            device_pages = tokens // 128
            mapping = binding.device_to_host
            if bool(
                (
                    (tokens < 0)
                    | (device_pages <= 0)
                    | (device_pages >= mapping.numel())
                ).any()
            ):
                continue
            host_pages = mapping[device_pages]
            if host_pages.dtype not in (torch.int32, torch.int64) or bool(
                ((host_pages <= 0) | (host_pages > torch.iinfo(torch.int32).max)).any()
            ):
                continue
            table = host_pages.to(torch.int32).reshape(1, -1).contiguous()
            for layer_id in sparse_layer_ids(self.mem_pool_device):
                pool, local = sparse_pool_and_layer(self.mem_pool_device, layer_id)
                backing = pool.decode_host_backing
                if backing.page_binding is not binding:
                    # Heterogeneous backings require their own proven snapshot.
                    continue
                full, rope = backing.layer_views(local)
                full = paged_kv_view(full, 128, 512)
                rope = paged_kv_view(rope, 128, 0)
                if layer_id not in state.layers:
                    topk = getattr(state, "shared_topk", None)
                    if topk is None:
                        continue
                    state.reserve(layer_id, state.row_capacity, topk, rope, full)
                warmed += warm_committed_prefix(
                    state,
                    layer_id,
                    owner=owner,
                    generation=generation,
                    committed_length=committed,
                    mapping_epoch=epoch,
                    block_table=table,
                    full_kv=full,
                    region=backing.regions[local][0],
                    binding=binding,
                    publication_length=int(published[2]),
                    publication_event=publication_event,
                )
        return warmed

    def collect_ready_reqs(self):
        return []

    def has_ongoing_staging(self) -> bool:
        runtime = self.prefill_offload_runtime
        if runtime is None:
            return False
        if getattr(runtime, "active", False) or getattr(runtime, "failed", False):
            return True
        publication = getattr(runtime, "host_visible", None)
        if publication is not None:
            return not publication.query()
        return bool(getattr(runtime, "layers", None))

    def prepare_spec_verify(self, batch) -> None:
        """Keep fixed-chain verify writes at their canonical Host-backed slots."""
        if batch.forward_mode.is_idle():
            return
        from sglang.srt.runtime_context import get_spec

        spec = get_spec()
        width = spec.speculative_num_draft_tokens
        if spec.speculative_eagle_topk != 1 or not width:
            raise ValueError("NPU HiSparse verify requires a fixed topk=1 chain")
        if batch.out_cache_loc.numel() != batch.req_pool_indices.numel() * width:
            raise ValueError("NPU HiSparse verify slots must match the chain width")

    def commit_spec_accept_tokens(self, batch, accept_indices) -> None:
        """Accepted chain tokens already occupy their logical prefix positions.

        The KV pool publishes every verify write in forward-stream order.
        Rejected suffix rows remain outside the committed sequence length and
        are overwritten or invalidated by the existing mutable-tail planner.
        """
        if batch.forward_mode.is_idle():
            return
        self.prepare_spec_verify(batch)
        if (
            accept_indices.ndim != 2
            or accept_indices.shape[0] != batch.req_pool_indices.numel()
        ):
            raise ValueError("NPU HiSparse accept rows must match active requests")

    def _drain_prefill(self) -> None:
        runtime = self.prefill_offload_runtime
        if runtime is None:
            return
        shutdown = getattr(runtime, "drain_for_shutdown", None)
        if shutdown is not None:
            shutdown()
            return
        drain = getattr(runtime, "drain", None)
        if drain is not None:
            if runtime.active:
                raise RuntimeError(
                    "NPU Prefill generation must finish before retirement"
                )
            drain()
        elif runtime.failed:
            runtime.drain_failed_batch()
        elif runtime.active or runtime.layers:
            runtime.finish_batch()

    def _retire_request(self, req) -> None:
        slot = req.kv.req_pool_idx
        if slot is None:
            return
        generations = self.req_to_token_pool.req_generation
        if not 0 < slot < generations.numel():
            raise ValueError("NPU request slot must exclude padding and fit the pool")
        retractions = req.retraction_count
        previous = self._retired_requests.get(req)
        if (
            previous is not None
            and previous.kv is req.kv
            and previous.retractions == retractions
            and previous.slot == slot
        ):
            return
        runtime = self.prefill_offload_runtime
        if runtime is not None and hasattr(runtime, "can_release_request"):
            if not runtime.can_release_request(req.rid):
                raise RuntimeError("NPU Prefill request still has in-flight work")
        fsa_state = getattr(self.mem_pool_device, "fsa_runtime_state", None)
        external_resident = getattr(fsa_state, "external_resident", None)
        if external_resident is not None:
            if self.decode_producer_stream is not None and str(self.device).startswith(
                "npu"
            ):
                # Scheduler normally binds the FSA work to this producer
                # stream; order retirement on the current stream before reuse.
                torch.npu.current_stream().wait_stream(self.decode_producer_stream)
            elif str(self.device).startswith("npu"):
                # A missing producer binding is abnormal, so conservatively
                # drain the device rather than release a live resident slot.
                torch.npu.synchronize()
            validate = getattr(external_resident, "validate_lifecycle", None)
            if validate is not None:
                # The producer stream has been ordered/drained above, so a
                # sticky device lifecycle error can be consumed outside graph
                # capture before the request slot is reused.
                validate()
            slot_tensor = torch.tensor(
                [slot], dtype=torch.int64, device=external_resident.device
            )
            external_resident.retire(slot_tensor)
        if fsa_state is not None:
            for layer in fsa_state.layers.values():
                if hasattr(layer, "warmup"):
                    layer.warmup.invalidate(
                        torch.tensor([slot], device=fsa_state.device)
                    )
        if hasattr(self, "_warmup_publications"):
            self._warmup_publications[slot] = -1
            self._warmup_publication_events.pop(slot, None)
        generations[slot] += 1
        retired = _RetiredRequest(
            req.kv,
            retractions,
            slot,
        )
        self._retired_requests[req] = retired
        req.hisparse_staging = False

    def request_finished(self, req) -> None:
        from sglang.srt.runtime_context import get_disagg

        disagg = get_disagg()
        if (
            disagg.disaggregation_mode == "decode"
            and not disagg.disaggregation_decode_enable_radix_cache
        ):
            req.skip_radix_cache_insert = True
        self._retire_request(req)

    def retract_req(self, req) -> None:
        self._retire_request(req)

    def abort_staging_request(self, req) -> None:
        self._retire_request(req)

    def get_token_stats(self):
        """Keep the existing NPU allocator/KDA statistics in the pool observer."""
        return None

    def prepare_shutdown(self) -> None:
        if self._shutdown_prepared:
            return
        self._drain_prefill()
        if self.decode_producer_stream is not None:
            self.decode_producer_stream.synchronize()
        if str(self.device).split(":")[0] == "npu":
            torch.npu.synchronize()
        self._shutdown_prepared = True

    def destroy(self) -> None:
        if self._destroyed:
            return
        self.prepare_shutdown()
        backing = self._backing
        if backing is not None:
            if backing.hicache_pool is not None:
                backing.hicache_pool.destroy()
            backing.pool.close(drained=True)
        runtime = self.prefill_offload_runtime
        if runtime is not None and hasattr(runtime, "adapter"):
            runtime.adapter.host_pool.close(drained=True)
        self._retired_requests.clear()
        self._destroyed = True
