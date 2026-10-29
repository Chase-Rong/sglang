"""The single external_plan fused planner/SFA execution state."""

from dataclasses import dataclass
import torch

@dataclass
class LogicalSelection:
    """Validated request-relative logical and physical selection metadata."""

    logical_topk: torch.Tensor
    physical_token_ids: torch.Tensor
    row_requests: torch.Tensor
    valid_rows: torch.Tensor
    miss_mask: torch.Tensor

def translate_logical_selection(
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    kv_lengths: torch.Tensor,
    query_lengths: torch.Tensor,
    *,
    page_size: int,
) -> LogicalSelection:
    """Validate and translate request-relative logical selections.

    TopK IDs remain logical token IDs.  Physical IDs are derived only through
    the request page table; this keeps the reference path from accidentally
    applying the fused operator's selection-layout translation twice.
    """
    if topk_indices.dim() == 2:
        logical = topk_indices.unsqueeze(1)
    elif topk_indices.dim() == 3 and topk_indices.shape[1] == 1:
        logical = topk_indices
    else:
        raise ValueError("reference TopK must have shape [T,K] or [T,1,K]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if block_table.dim() != 2 or block_table.shape[1] == 0:
        raise ValueError("block_table must have shape [B,pages]")
    kv_lengths = kv_lengths.to(dtype=torch.int64, device=logical.device).view(-1)
    query_lengths = query_lengths.to(dtype=torch.int64, device=logical.device).view(-1)
    if (
        kv_lengths.numel() != query_lengths.numel()
        or block_table.shape[0] != kv_lengths.numel()
        or kv_lengths.numel() == 0
    ):
        raise ValueError("KV and query length counts must match")
    rows = logical.shape[0]
    row_ids = torch.arange(rows, device=logical.device, dtype=torch.int64)
    row_requests = torch.bucketize(row_ids, query_lengths, right=True)
    valid_rows = row_requests < query_lengths.numel()
    row_requests = row_requests.clamp(max=max(query_lengths.numel() - 1, 0))
    logical_ids = logical[:, 0].to(dtype=torch.int64)
    valid = valid_rows[:, None] & (logical_ids >= 0)
    valid = valid & (logical_ids < kv_lengths[row_requests, None])
    pages = block_table.to(device=logical.device, dtype=torch.int64)[row_requests]
    page_ids = torch.div(logical_ids, page_size, rounding_mode="floor")
    offsets = logical_ids.remainder(page_size)
    selected_pages = pages.gather(
        1, page_ids.clamp(min=0, max=max(block_table.shape[1] - 1, 0))
    )
    valid = valid & (page_ids < block_table.shape[1]) & (selected_pages >= 0)
    physical = selected_pages * page_size + offsets
    physical = physical.masked_fill(~valid, -1).unsqueeze(1)
    miss_mask = valid.unsqueeze(1)
    return LogicalSelection(
        logical.clone(), physical, row_requests, valid_rows, miss_mask
    )

def paged_kv_view(tensor: torch.Tensor, page_size: int, head_dim: int) -> torch.Tensor:
    """Expose split PA storage without copying the full context to HBM."""
    if tensor.dim() == 4 and tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)
    if (
        tensor.dim() != 3
        or tensor.shape[1:] != (page_size, head_dim)
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            "fused KV requires contiguous split [pages,page_size,1,dim] storage"
        )
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("fused KV requires FP16 or BF16 storage")
    return tensor

def mask_causal_topk(topk, row_requests, query_lengths, kv_lengths, *, positions=None):
    rows = topk.shape[0]
    row_ids = torch.arange(rows, device=topk.device, dtype=torch.int32)
    starts = torch.cat((query_lengths.new_zeros(1), query_lengths[:-1]))
    counts = query_lengths - starts
    row_limit = (
        kv_lengths[row_requests]
        - counts[row_requests]
        + row_ids
        - starts[row_requests]
        + 1
    )
    if positions is not None:
        if positions.dim() != 1 or positions.numel() != rows:
            raise ValueError(
                "sparse query positions must have one logical position per query row"
            )
        row_limit = torch.minimum(row_limit, positions.to(device=topk.device) + 1)
    row_limit = torch.minimum(row_limit, kv_lengths[row_requests])
    return topk.masked_fill((topk < 0) | (topk >= row_limit[:, None, None]), -1)

def build_external_metadata_plan(*args, **kwargs):
    # Keep CPU contract users independent of Triton and the NPU runtime.
    from sglang.srt.hardware_backend.npu.hisparse.fsa_metadata import (
        build_external_metadata_plan as build,
    )

    return build(*args, **kwargs)

def normalize_topk(topk_indices):
    if topk_indices.ndim == 2:
        topk_indices = topk_indices.unsqueeze(1)
    if topk_indices.ndim != 3 or topk_indices.shape[1] != 1:
        raise ValueError("TopK must have shape [T,K] or [T,1,K]")
    return topk_indices.to(dtype=torch.int32).contiguous()


class FusedSparseAttentionState:
    """Per-layer resident and workspace ownership for the sole external path."""

    def __init__(self, device, req_to_token_pool, *, external_planner=True,
                 resident_capacity=None, layer_anchors=None, row_capacity=None,
                 mutable_tail_tokens=0, mapped_host_plan=False, max_query_routes=1,
                 page_size=128, execution_abi=None, batch_capacity=None,
                 load_mode=None, warmup_threshold=None):
        from .config import resolve_fsa_execution_abi, resolve_shared_load_mode, resolve_shared_warmup_threshold
        from .shared_execution import _INSTANCES
        if not external_planner or mapped_host_plan:
            raise ValueError("external_plan only supports the fused device planner; mapped_host_plan/row execution was removed")
        self.execution_abi = resolve_fsa_execution_abi(execution_abi)
        self._load_mode = resolve_shared_load_mode(load_mode)
        self._warmup_threshold = resolve_shared_warmup_threshold(warmup_threshold)
        self.device = torch.device(device)
        if self.device.type == "npu" and self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        self.pool = req_to_token_pool
        self.shared_instance = next(_INSTANCES)
        self.batch_capacity = batch_capacity or min(
            row_capacity or req_to_token_pool.req_generation.numel() - 1,
            max(req_to_token_pool.req_generation.numel() - 1,
                ((row_capacity or 1) + max_query_routes - 1) // max_query_routes))
        if not 0 < self.batch_capacity <= 1024 or req_to_token_pool.req_generation.numel() > 32767 or page_size != 128:
            raise ValueError("external_plan requires page128, Bcap<=1024 and Qcap<=32767")
        if not 0 < max_query_routes <= 6 or mutable_tail_tokens < 0:
            raise ValueError("external_plan requires 1..6 routes and a nonnegative mutable tail")
        self.shared_native_workspace_bytes = 0
        if self.device.type == "npu":
            from sglang.srt.hardware_backend.npu.fsa_native import shared_v3_workspace_bytes
            self.shared_native_workspace_bytes = shared_v3_workspace_bytes()
        self.layers = {}
        self.external_planner = True
        self.resident_capacity = resident_capacity
        self.layer_anchors = layer_anchors or {}
        self.row_capacity = row_capacity
        self.mutable_tail_tokens = mutable_tail_tokens
        self.max_query_routes = max_query_routes
        self.page_size = page_size
        self.planners = {}
        self.external_resident = None
        self.plan_versions = {}
        self.forward_serial = 0
        self.mapping_epoch_input = torch.zeros((), dtype=torch.int64, device=self.device)
        self.generations = torch.empty_like(self.pool.req_generation, device=self.device)
        self._generation_snapshot = self.pool.req_generation.clone()
        self._pool_epoch = getattr(self.pool, "req_generation_epoch", 0)
        self._epoch = 0
        self._generations_initialized = False
        self._generation_version = 0
        self.refresh_generations()

    @property
    def load_mode(self):
        return self._load_mode

    @property
    def warmup_threshold(self):
        return self._warmup_threshold

    def invalidate_warmup_content(self, owners):
        """Explicit same-prefix overwrite/replacement hook; append needs none."""
        for layer in self.layers.values():
            if hasattr(layer, "warmup"):
                layer.guard.assert_stream()
                layer.warmup.invalidate(owners)

    def refresh_mapping_epoch(self, mapping_epoch):
        """Update the fixed-address epoch input outside capture/replay execution."""
        for layer in self.layers.values():
            layer.guard.assert_stream()
        if (
            self.warmup_threshold
            and getattr(self, "_warmup_epoch", None) != mapping_epoch
        ):
            for layer in self.layers.values():
                if hasattr(layer, "warmup"):
                    layer.warmup.marker.fill_(-1)
                    layer.warmup.stamp.fill_(-1)
            self._warmup_epoch = mapping_epoch
        self.mapping_epoch_input.fill_(mapping_epoch)

    def refresh_generations(self):
        """Refresh outside capture; graph replay reads this stable device buffer."""
        for layer in self.layers.values():
            layer.guard.assert_stream()
        current = self.pool.req_generation
        self.forward_serial += 1
        pool_epoch = getattr(self.pool, "req_generation_epoch", 0)
        # Request generations live on CPU and usually stay unchanged throughout
        # decode. Avoid a blocking H2D copy while still advancing plan versions.
        if (
            self._generations_initialized
            and pool_epoch == self._pool_epoch
            and torch.equal(current, self._generation_snapshot)
        ):
            return
        # A failed upload must be retried even after the CPU snapshot advances.
        self._generations_initialized = False
        if (
            pool_epoch != self._pool_epoch
            or torch.any(current < self._generation_snapshot).item()
        ):
            self._epoch += 1
        if self.warmup_threshold:
            changed = current != self._generation_snapshot
            if pool_epoch != self._pool_epoch:
                changed = torch.ones_like(changed, dtype=torch.bool)
            for layer in self.layers.values():
                if hasattr(layer, "warmup"):
                    device_changed = changed.to(device=self.device)
                    layer.warmup.marker.masked_fill_(device_changed[:, None], -1)
                    layer.warmup.stamp.masked_fill_(device_changed[:, None], -1)
        self._pool_epoch = pool_epoch
        self._generation_snapshot.copy_(current)
        identities = current + (self._epoch << 32)
        self.generations.copy_(identities)
        self._generation_version += 1
        self._generations_initialized = True

    def layer(self, layer_id):
        if layer_id not in self.layers:
            raise RuntimeError("shared layer must be reserved with its geometry first")
        return self.layers[layer_id]

    def reserve(self, layer_id, token_capacity, topk, full_rope, full_kv):
        from .shared_execution import reserve_shared
        return reserve_shared(self, layer_id, token_capacity, topk, full_rope, full_kv)

    def build_inputs(self, **args):
        from .shared_execution import build_shared_inputs
        args.setdefault("query_positions", None)
        args.setdefault("mapping_epoch", 0)
        return build_shared_inputs(self, **args)


def run_fused_sparse_attention(inputs, mode="fused"):
    from .shared_execution import run_shared
    return run_shared(inputs, mode)
