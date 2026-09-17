"""Capacity accounting for registered Decode KV and persistent FSA state."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeKvMemoryBudget:
    layers: int
    planner_groups: int
    latent_dim: int
    rope_dim: int
    index_dim: int
    element_bytes: int
    topk: int
    resident_capacity: int
    host_bytes: int
    graph_rows: int = 0
    callback_instances: int = 1
    shared_fence_bytes: int = 0
    staging_bytes: int = 0
    rows_per_request: int = 1
    prefill_layer_slots: int = 0

    def __post_init__(self):
        if (
            min(
                self.layers,
                self.planner_groups,
                self.latent_dim,
                self.index_dim,
                self.element_bytes,
                self.host_bytes,
                self.callback_instances,
            )
            <= 0
        ):
            raise ValueError("Decode offload requires positive dimensions and budgets")
        if self.rope_dim < 0:
            raise ValueError("RoPE dimension must be nonnegative")
        if (
            not 0 < self.topk <= 2051
            or not self.topk <= self.resident_capacity <= 32767
        ):
            raise ValueError("Decode selection capacity is outside the fused plan ABI")
        if (
            self.planner_groups > self.layers
            or self.graph_rows < 0
            or self.shared_fence_bytes < 0
        ):
            raise ValueError("invalid planner group or graph capacity")
        if self.staging_bytes < 0 or self.staging_bytes % 32:
            raise ValueError(
                "Decode staging budget must be nonnegative and 32-byte aligned"
            )
        if self.rows_per_request <= 0:
            raise ValueError("Decode query rows per request must be positive")
        if not 0 <= self.prefill_layer_slots <= self.layers:
            raise ValueError("Prefill slots must fit the Host layer count")

    @property
    def host_bytes_per_token(self):
        return self.layers * (self.latent_dim + self.rope_dim) * self.element_bytes

    @property
    def device_bytes_per_token(self):
        return (
            self.layers * self.index_dim
            + self.prefill_layer_slots * (self.latent_dim + self.rope_dim)
        ) * self.element_bytes

    def selection_capacity(self, page_size):
        if page_size <= 0:
            raise ValueError("page size must be positive")
        capacity = (self.resident_capacity + page_size - 1) // page_size * page_size
        if capacity > 32767:
            raise ValueError(
                "page-aligned selection capacity exceeds the fused plan ABI"
            )
        return capacity

    def row_capacity(self, requests):
        return max(requests, self.graph_rows) * self.rows_per_request

    def fixed_bytes(self, request_rows, page_size):
        rows = self.row_capacity(request_rows)
        if not 0 < rows <= 32767:
            raise ValueError("Decode request rows must fit the fused plan ABI")
        capacity = self.selection_capacity(page_size)
        status_stride = (self.topk + 8) // 8 * 8
        layer_bytes = (
            capacity * (self.latent_dim + self.rope_dim) * self.element_bytes
            + capacity // page_size * 4
            + status_stride * 4
            + 16400 * 2
            + (capacity + 1) * 8
            + 12
        )
        planner_device = self.topk * 8 + 9
        planner_host = (2 * self.topk + 2 + 2 * capacity + 3) * 8
        scratch = capacity * 33 * self.callback_instances
        device = rows * (
            self.layers * layer_bytes + self.planner_groups * planner_device
        )
        host = self.planner_groups * (rows * planner_host + scratch)
        staging = self.staging_bytes + 9 if self.staging_bytes else 0
        return host, device + self.shared_fence_bytes + staging

    def required_bytes(self, tokens, request_rows, page_size):
        if page_size <= 0 or tokens < 0 or tokens % page_size:
            raise ValueError(
                "Decode token capacity must be nonnegative and page aligned"
            )
        host_fixed, device_fixed = self.fixed_bytes(request_rows, page_size)
        storage_rows = tokens + page_size
        return (
            host_fixed + storage_rows * self.host_bytes_per_token,
            device_fixed + storage_rows * self.device_bytes_per_token,
        )

    def max_tokens(self, device_bytes, page_size, request_rows_for_tokens):
        if page_size <= 0 or device_bytes < 0:
            raise ValueError("invalid Decode page size or NPU byte budget")
        lower = 0
        upper = (
            min(
                self.host_bytes // self.host_bytes_per_token,
                device_bytes // self.device_bytes_per_token,
            )
            // page_size
        )
        while lower < upper:
            candidate = (lower + upper + 1) // 2
            tokens = candidate * page_size
            rows = max(1, request_rows_for_tokens(tokens))
            host_required, device_required = self.required_bytes(
                tokens, rows, page_size
            )
            if host_required <= self.host_bytes and device_required <= device_bytes:
                lower = candidate
            else:
                upper = candidate - 1
        return lower * page_size

    def validate(self, tokens, request_rows, page_size, device_bytes=None):
        host_required, device_required = self.required_bytes(
            tokens, request_rows, page_size
        )
        if host_required > self.host_bytes:
            raise MemoryError(
                f"Decode Host backing and planner require {host_required} bytes; budget {self.host_bytes}"
            )
        if device_bytes is not None and device_required > device_bytes:
            raise MemoryError(
                f"Decode Indexer and selection require {device_required} bytes; budget {device_bytes}"
            )


@dataclass(frozen=True)
class JointDsaKdaCapacity:
    target_tokens: int
    requests: int
    state_slots: int
    dsa_bytes: int
    kda_bytes: int
    unused_bytes: int
    request_bytes: int = 0


def solve_joint_dsa_kda_tokens(
    budget,
    available_bytes,
    token_limit,
    page_size,
    requests,
    state_bytes_per_slot,
    slots_per_request=1,
    ring_bytes_per_slot=0,
    intermediate_bytes_per_request=0,
    explicit_state_slots=None,
    request_bytes_per_row=0,
):
    """Prefer the requested concurrency, then maximize the shared token pool."""
    if (
        min(token_limit, page_size, requests, state_bytes_per_slot, slots_per_request)
        <= 0
    ):
        raise ValueError("joint capacity dimensions must be positive")
    if (
        min(
            available_bytes,
            ring_bytes_per_slot,
            intermediate_bytes_per_request,
            request_bytes_per_row,
        )
        < 0
    ):
        raise ValueError("joint capacity byte budgets must be nonnegative")
    if explicit_state_slots is not None and explicit_state_slots <= 0:
        raise ValueError("explicit KDA state slots must be positive")
    budget.fixed_bytes(1, page_size)
    minimum_per_request = max(page_size, 2)
    upper = min(
        requests, token_limit // minimum_per_request, 32767 // budget.rows_per_request
    )
    if explicit_state_slots is not None:
        upper = min(upper, explicit_state_slots // slots_per_request)

    def fits(candidate_requests):
        slots = (
            candidate_requests * slots_per_request
            if explicit_state_slots is None
            else explicit_state_slots
        )
        minimum_tokens = (
            (candidate_requests * minimum_per_request + page_size - 1)
            // page_size
            * page_size
        )
        host_bytes, device_bytes = budget.required_bytes(
            minimum_tokens, candidate_requests, page_size
        )
        state_bytes = (slots + 1) * (state_bytes_per_slot + ring_bytes_per_slot) + (
            candidate_requests + 1
        ) * intermediate_bytes_per_request
        mapping_bytes = (candidate_requests + 1) * request_bytes_per_row
        return (
            host_bytes <= budget.host_bytes
            and device_bytes + state_bytes + mapping_bytes <= available_bytes
        )

    lower = 0
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        if fits(midpoint):
            lower = midpoint
        else:
            upper = midpoint - 1
    if lower == 0:
        raise MemoryError(
            "Joint DSA/KDA budget cannot fit one request with the configured "
            f"state, Graph and token limits: requested={requests}, "
            f"available_bytes={available_bytes}"
        )
    requests = lower
    state_slots = (
        requests * slots_per_request
        if explicit_state_slots is None
        else explicit_state_slots
    )
    kda_bytes = (state_slots + 1) * (state_bytes_per_slot + ring_bytes_per_slot) + (
        requests + 1
    ) * intermediate_bytes_per_request
    request_bytes = (requests + 1) * request_bytes_per_row
    dsa_available = available_bytes - kda_bytes - request_bytes
    tokens = min(
        token_limit // page_size * page_size,
        budget.max_tokens(dsa_available, page_size, lambda _: requests),
    )
    _, dsa_bytes = budget.required_bytes(tokens, requests, page_size)
    return JointDsaKdaCapacity(
        tokens,
        requests,
        state_slots,
        dsa_bytes,
        kda_bytes,
        available_bytes - dsa_bytes - kda_bytes - request_bytes,
        request_bytes,
    )


def budget_from_config(kvc, layers):
    import torch

    from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.npu.speculative_sparse import (
        sparse_mtp_draft_layers,
        sparse_mtp_rows,
    )
    from sglang.srt.runtime_context import get_disagg, get_exec, get_spec

    if (
        kvc.device != "npu"
        or not kvc.use_mla_backend
        or not is_deepseek_dsa(kvc.model_config.hf_config)
        or get_disagg().disaggregation_mode
        != ("null" if envs.SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD.get() else "decode")
    ):
        raise ValueError("registered Decode sizing requires an NPU DSA Decode service")
    if kvc.kv_cache_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("registered Decode sizing requires BF16 or FP16 KV")
    model = kvc.model_config
    from sglang.srt.model_executor.pool_configurator import _get_dsa_cache_layer_ids

    layer_ids = _get_dsa_cache_layer_ids(kvc, layers)
    groups = 1 + sum(
        not dsa_layer_skips_topk(model.hf_config, layer) for layer in layer_ids[1:]
    )
    draft_layers = sparse_mtp_draft_layers(kvc)
    prefill_slots = 0
    if envs.SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD.get():
        configured_slots = envs.SGLANG_NPU_PREFILL_LAYER_SLOTS.get()
        prefill_slots = min(configured_slots, layers) + min(
            configured_slots, draft_layers
        )
    rows_per_request = sparse_mtp_rows(kvc.spec_algorithm, get_spec(), model)
    layers += draft_layers
    groups += draft_layers
    graph_config = get_exec().graph.cuda_graph_config.decode
    graph_sizes = () if graph_config.backend == "disabled" else (graph_config.bs or ())
    graph_rows = max(graph_sizes, default=0)
    if graph_config.backend != "disabled":
        graph_rows = max(graph_rows, graph_config.max_bs or 0)
    return DecodeKvMemoryBudget(
        layers,
        groups,
        model.kv_lora_rank,
        model.qk_rope_head_dim,
        model.index_head_dim,
        2,
        int(model.hf_text_config.index_topk)
        + int(getattr(model.hf_text_config, "index_kpool", 1))
        - 1,
        envs.SGLANG_NPU_FSA_RESIDENT_CAPACITY.get(),
        envs.SGLANG_NPU_DECODE_HOST_KV_BYTES.get(),
        graph_rows,
        max(1, len(graph_sizes) if graph_sizes else graph_rows)
        * (
            1
            if kvc.spec_algorithm.is_none()
            else int(get_spec().speculative_num_steps) + 2
        )
        + 1,
        shared_fence_bytes=4 if envs.SGLANG_NPU_SHARED_HOST_KV.get() else 0,
        staging_bytes=envs.SGLANG_NPU_LAYER_STAGING_BYTES.get(),
        rows_per_request=rows_per_request,
        prefill_layer_slots=prefill_slots,
    )
