"""Stream-ordered LRU planning over registered metadata, including graph replay."""

from dataclasses import dataclass
from uuid import uuid4

import torch


@dataclass
class SparseSelectionPlan:
    physical_keys: torch.Tensor
    slots: torch.Tensor
    miss_mask: torch.Tensor
    cacheable_mask: torch.Tensor | None = None


class NativeSparsePlanner:
    def __init__(self, device, rows, topk, capacity, *, device_slots=True):
        from sglang.srt.hardware_backend.npu import fsa_native as kernel

        self.kernel = kernel
        self.device = torch.device(device)
        self.rows, self.topk, self.capacity = rows, topk, capacity
        if not 0 < topk <= min(capacity, 2051) or capacity > 32767 or rows > 32767:
            raise ValueError(
                "planner capacity must cover TopK and fit the native int16 ABI"
            )
        allocation = uuid4().hex
        device_id = self.device.index
        if device_id is None:
            device_id = torch.npu.current_device()
        self.regions = {}
        self.closed = False
        for name, shape in (
            ("keys", (rows, topk)),
            ("owners", (rows, 1)),
            ("lengths", (rows, 1)),
            ("slots", (rows, topk)),
        ):
            self.regions[name] = kernel.RegisteredHostRegion.allocate(
                shape,
                torch.int64,
                device_id=device_id,
                name=f"fsa-plan-{allocation}-{name}",
            )
        self.resident = torch.full((rows, capacity), -1, dtype=torch.int64)
        self.ages = torch.zeros_like(self.resident)
        self.row_state = torch.full((rows, 3), -1, dtype=torch.int64)
        self.slots = (
            torch.empty((rows, topk), dtype=torch.int64, device=self.device)
            if device_slots else self.regions["slots"].tensor
        )
        self.indices = torch.arange(rows, dtype=torch.int64, device=self.device)
        self.valid = torch.ones(rows, dtype=torch.bool, device=self.device)

    def enqueue(self, keys, owners, lengths, *, device_slots=True):
        if self.closed:
            raise RuntimeError("native sparse planner is closed")
        rows = keys.shape[0]
        if rows > self.rows or keys.shape[1] != self.topk:
            raise ValueError("planner geometry changed after reservation")
        indices, valid = self.indices[:rows], self.valid[:rows]
        for name, tensor in (
            ("keys", keys),
            ("owners", owners[:, None]),
            ("lengths", lengths[:, None]),
        ):
            self.kernel.copy_rows_to_registered_host(
                tensor.to(dtype=torch.int64).contiguous(),
                self.regions[name],
                indices,
                indices,
                valid,
            )
        self.kernel.native_ops().fsa_enqueue_lru_plan(
            keys,
            self.regions["keys"].tensor,
            self.regions["owners"].tensor,
            self.regions["lengths"].tensor,
            self.resident,
            self.ages,
            self.row_state,
            self.regions["slots"].tensor,
            rows,
        )
        if device_slots:
            self.kernel.copy_rows_from_registered_host(
                self.regions["slots"], self.slots, indices, indices, valid
            )
        return self.slots[:rows]

    def close(self, *, drained=False):
        """Release registrations after stream drain and destruction of all using graphs."""
        if not self.regions:
            self.closed = True
            return
        if not drained:
            raise RuntimeError(
                "native planner callbacks and graphs must be drained before close"
            )
        self.closed = True
        for name, region in tuple(self.regions.items()):
            region.close(drained=True)
            del self.regions[name]


class MappedSparsePlanner(NativeSparsePlanner):
    def __init__(self, device, rows, topk, capacity):
        super().__init__(device, rows, topk, capacity, device_slots=False)
        self.regions["cacheable"] = self.kernel.RegisteredHostRegion.allocate(
            (rows, topk), torch.bool, device_id=self.device.index
        )
        self.regions["cacheable"].tensor.fill_(True)
        self.stage_indices = torch.arange(rows * 2, device=self.device, dtype=torch.int64)
        self.stage_valid = torch.ones(rows * 2, device=self.device, dtype=torch.bool)
        self.layers = {}

    def enqueue(self, keys, owners, lengths, *, cacheable=None):
        super().enqueue(keys, owners, lengths, device_slots=False)
        if cacheable is not None:
            rows = keys.shape[0]
            self.kernel.copy_rows_to_registered_host(
                cacheable.contiguous(), self.regions["cacheable"],
                self.indices[:rows], self.indices[:rows], self.valid[:rows],
            )

    def reserve_layer(self, layer_id):
        if self.closed:
            raise RuntimeError("native sparse planner is closed")
        if layer_id not in self.layers:
            region = self.kernel.RegisteredHostRegion.allocate(
                (self.rows, 16400), torch.int16, device_id=self.device.index
            )
            region.tensor.fill_(-1)
            self.regions[f"plan-{layer_id}"] = region
            self.layers[layer_id] = (
                torch.full((self.rows, self.capacity), -1, dtype=torch.int64),
                torch.full((self.rows, 2), -1, dtype=torch.int64),
                torch.empty((self.rows, self.topk), dtype=torch.bool),
                region,
            )

    def materialize(self, trigger, layer_id, rows, *, commit=False):
        if self.closed:
            raise RuntimeError("native sparse planner is closed")
        applied, state, misses, region = self.layers[layer_id]
        self.kernel.native_ops().fsa_host_plan(
            trigger, self.regions["keys"].tensor, self.regions["slots"].tensor,
            self.regions["owners"].tensor, self.regions["lengths"].tensor,
            self.regions["cacheable"].tensor, applied, state, region.tensor, misses,
            rows, commit,
        )
        return MappedSelectionPlan(self, layer_id, rows, trigger), region

    def stage(self, region, destination, rows):
        self.kernel.unidex_copy_inplace(
            region.tensor.view(-1, 8200), destination.view(-1, 8200),
            self.stage_indices[:rows * 2], self.stage_indices[:rows * 2],
            self.stage_valid[:rows * 2], src_address_ndims=1, dst_address_ndims=1,
            src_ptr=region.device_ptr,
        )


@dataclass
class MappedSelectionPlan:
    planner: MappedSparsePlanner
    layer_id: int
    rows: int
    trigger: torch.Tensor

    @property
    def physical_keys(self):
        return self.planner.regions["keys"].tensor[:self.rows]

    @property
    def slots(self):
        return self.planner.regions["slots"].tensor[:self.rows]

    @property
    def miss_mask(self):
        return self.planner.layers[self.layer_id][2][:self.rows]

    def commit(self):
        self.planner.materialize(self.trigger, self.layer_id, self.rows, commit=True)


def materialize_layer_plan(
    cache,
    physical_keys,
    slots,
    membership,
    *,
    control_offset=16384,
    cacheable_mask=None,
):
    rows, topk = physical_keys.shape
    if topk > 2051 or control_offset + 8 > membership.shape[1]:
        raise ValueError("external plan exceeds the fused kernel metadata ABI")
    capacity = (
        cache.selection_kv_cache.shape[0]
        * cache.selection_kv_cache.shape[1]
        // cache.max_tokens
    )
    valid = (physical_keys >= 0) & (slots >= 0) & (slots < capacity)
    safe_slots = slots.clamp(min=0, max=capacity - 1)
    applied = cache.applied_keys[:rows]
    hit = valid & (applied.gather(1, safe_slots) == physical_keys)
    if cacheable_mask is not None:
        if (
            cacheable_mask.shape != physical_keys.shape
            or cacheable_mask.dtype != torch.bool
        ):
            raise ValueError("cacheable selection mask must match the physical keys")
        hit = hit & cacheable_mask
    miss = valid & ~hit
    encoded = torch.where(hit, safe_slots + 1, -(safe_slots + 1))
    encoded = encoded.masked_fill(~valid, 0).to(torch.int16)
    membership[:, control_offset - topk : control_offset].copy_(encoded)
    control = membership[:, control_offset : control_offset + 8]
    control[:, 0] = 0x5A4D
    control[:, 1] = 0x5A45
    control[:, 2] = topk
    control[:, 3] = control_offset - topk
    control[:, 4].copy_(torch.arange(rows, device=slots.device, dtype=torch.int32))
    control[:, 5] = 0x5A44
    control[:, 6] = capacity
    control[:, 7] = 0
    return SparseSelectionPlan(physical_keys, safe_slots, miss, cacheable_mask)


def commit_layer_plan(cache, plan):
    if isinstance(plan, MappedSelectionPlan):
        plan.commit()
        return
    rows = plan.slots.shape[0]
    applied = cache.applied_keys[:rows]
    valid = plan.physical_keys >= 0
    destinations = plan.slots.masked_fill(~valid, applied.shape[1])
    reusable_keys = plan.physical_keys
    if plan.cacheable_mask is not None:
        reusable_keys = reusable_keys.masked_fill(~plan.cacheable_mask, -1)
    cache.applied_keys_with_padding[:rows].scatter_(1, destinations, reusable_keys)


def gather_selection_misses(inputs):
    from sglang.srt.hardware_backend.npu import fsa_native as kernel

    plan = inputs.selection_plan
    rows, _ = plan.slots.shape
    capacity = (
        inputs.selection_kv_cache.shape[0] * inputs.selection_kv_cache.shape[1] // rows
    )
    offsets = (
        torch.arange(rows, dtype=torch.int64, device=plan.slots.device)[:, None]
        * capacity
    )
    destination_ids = (offsets + plan.slots).flatten().contiguous()
    source_ids = plan.physical_keys.flatten().clamp(min=0).contiguous()
    valid = plan.miss_mask.flatten().contiguous()
    for source, destination, region in (
        (inputs.full_kv_cache, inputs.selection_kv_cache, inputs.full_kv_region),
        (inputs.full_k_rope, inputs.selection_k_rope, inputs.full_rope_region),
    ):
        if source.shape[-1] == 0:
            continue
        kernel.unidex_copy_inplace(
            source.reshape(-1, source.shape[-1]),
            destination.reshape(-1, destination.shape[-1]),
            source_ids,
            destination_ids,
            valid,
            src_address_ndims=1,
            dst_address_ndims=1,
            src_ptr=None if region is None else region.device_ptr,
        )


def run_reference_sparse_attention(inputs):
    import torch_npu

    gather_selection_misses(inputs)
    plan = inputs.selection_plan
    rows = plan.slots.shape[0]
    key = inputs.selection_kv_cache.unsqueeze(2)
    rope = inputs.selection_k_rope.unsqueeze(2)
    pages, page_size = key.shape[:2]
    capacity = pages * page_size // rows
    offsets = torch.arange(rows, device=plan.slots.device)[:, None] * capacity
    sparse_indices = (plan.slots + offsets).masked_fill(plan.physical_keys < 0, -1)
    valid = plan.physical_keys >= 0
    order = torch.argsort((~valid).to(torch.int32), dim=-1, stable=True)
    sparse_indices = sparse_indices.gather(1, order)
    table = (
        torch.arange(pages, device=key.device, dtype=torch.int32)[None]
        .expand(rows, -1)
        .contiguous()
    )
    lengths = valid.sum(dim=-1).to(torch.int32)
    query_lengths = torch.arange(1, rows + 1, device=key.device, dtype=torch.int32)
    width = key.shape[-1]
    query_rope = inputs.query[..., width:]
    if query_rope.shape[-1] == 0:
        query_rope = inputs.query.new_zeros((*inputs.query.shape[:-1], 64))
        rope = key.new_zeros((*key.shape[:-1], 64))
    output, _, _ = torch_npu.npu_sparse_flash_attention(
        query=inputs.query[..., :width],
        query_rope=query_rope,
        key=key,
        value=key,
        key_rope=rope,
        sparse_indices=sparse_indices[:, None].int(),
        block_table=table,
        actual_seq_lengths_query=query_lengths,
        actual_seq_lengths_kv=lengths,
        scale_value=inputs.scale_value,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=False,
    )
    return output
