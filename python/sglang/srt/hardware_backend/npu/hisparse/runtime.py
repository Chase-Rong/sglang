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

    def collect_ready_reqs(self):
        return []

    def has_ongoing_staging(self) -> bool:
        return False

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
