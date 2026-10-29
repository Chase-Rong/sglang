"""Request-shared FSA ABI3 execution and its checked, outside-capture boundary.

No legacy row payload or CPU planner is used here. Tensor operations in prepare,
lease acquisition and completion are replayed; Python only guards stream ownership
and checks the aggregate error after a complete model forward.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from itertools import count

import torch

BACKEND_MAGIC = 0x485350335631
_INSTANCES = count(1)
_FORWARD = ContextVar("shared_fsa_forward", default=None)


def register_shared_resident(resident):
    collector = _FORWARD.get()
    if collector is not None:
        collector[id(resident)] = resident


def check_shared_residents(residents):
    groups = {}
    for resident in residents:
        groups.setdefault(resident.device, {})[id(resident)] = resident
    for device, group in groups.items():
        if device.type == "npu" and torch.npu.is_current_stream_capturing():
            raise RuntimeError("shared result validation must run outside capture")
        # One aggregate read per device per forward, never per layer/capture.
        error = torch.stack([r.lifecycle_error for r in group.values()]).amax()
        if error.item():
            raise RuntimeError(
                "external_plan execution failed; model results cannot be consumed"
            )


@contextmanager
def checked_shared_forward():
    """Checked component/model boundary, including target and draft forwards."""
    parent = _FORWARD.get()
    if parent is not None:
        yield parent
        return
    residents = {}
    token = _FORWARD.set(residents)
    try:
        yield residents
        check_shared_residents(residents.values())
    finally:
        _FORWARD.reset(token)


class WorkspaceGuard:
    """Exclusive prepare-through-publication owner, with drained transitions."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.stream = None
        self.main_stream = None
        self.active = False
        self.fence = None
        self.quarantine = None
        self.side_streams = ()
        self.retained = []

    def assert_stream(self, stream=None):
        if self.quarantine is not None:
            raise RuntimeError("shared workspace is quarantined after failed drain")
        if stream is None:
            stream = torch.npu.current_stream() if self.device.type == "npu" else "cpu"
        identity = getattr(stream, "npu_stream", stream)
        if self.stream is not None and identity != self.stream:
            if self.active or (self.fence is not None and not self.fence.query()):
                raise RuntimeError("shared workspace cannot change stream before drain")
        return identity

    def acquire(self, stream=None):
        if self.quarantine is not None:
            raise RuntimeError("shared workspace is quarantined after failed drain")
        if stream is None:
            stream = torch.npu.current_stream() if self.device.type == "npu" else "cpu"
        identity = self.assert_stream(stream)
        if self.active:
            raise RuntimeError("shared workspace has an overlapping prepare/replay")
        if self.stream is not None and identity != self.stream:
            if self.fence is not None and not self.fence.query():
                raise RuntimeError("shared workspace cannot change stream before drain")
        self.stream = identity
        self.main_stream = stream
        self.active = True

    def release(self):
        if self.device.type == "npu":
            self.fence = torch.npu.Event()
            self.fence.record(torch.npu.current_stream())
        self.active = False

    def drained(self):
        self.active = False
        self.fence = None
        self.stream = None

    def fail(self, error):
        self.quarantine = error


def expected_identity(resident, lease, request_slots, instance):
    owners = request_slots.to(device=resident.device, dtype=torch.long)
    safe = owners.clamp(0, resident.request_capacity - 1)
    valid = (owners > 0) & (owners < resident.request_capacity) & lease.active[safe]
    identity = torch.stack(
        (
            torch.full_like(owners, BACKEND_MAGIC),
            torch.full_like(owners, instance),
            torch.full_like(owners, lease.layer_id),
            owners,
            lease.generation[safe],
            lease.mapping_epoch[safe],
            lease.serial[safe],
            lease.serial[safe],
            lease.planner_version[safe],
        ),
        dim=1,
    )
    return identity.masked_fill(~valid[:, None], -1).contiguous()


def complete_shared(
    resident, lease, owners, expected, completion, status, ready, counts
):
    """Check B-indexed results, then scatter exact successful owners to Qcap."""
    # Error visibility does not depend on a lease: malformed/padding/empty
    # requests can still be rejected by native before any execution ref exists.
    resident._error(status != 0, 8)
    safe = owners.clamp(0, resident.request_capacity - 1).long()
    batch_active = (
        (owners > 0) & (owners < resident.request_capacity) & lease.active[safe]
    )
    seen = torch.zeros(
        resident.request_capacity, dtype=torch.int32, device=resident.device
    )
    seen.scatter_add_(0, safe, batch_active.to(torch.int32))
    selected = (
        torch.arange(ready.shape[1], device=resident.device)[None, :] < counts[:, None]
    )
    successful = (
        batch_active
        & (seen[safe] == 1)
        & (status == 0)
        & (completion == expected).all(dim=1)
        & (~selected | (ready == 2)).all(dim=1)
    )
    success_count = torch.zeros_like(seen)
    success_count.scatter_add_(0, safe, successful.to(torch.int32))
    success = success_count == 1
    executing = resident._consume_mask(lease.layer_id, lease)
    failed = executing & ~success
    resident._error(failed, 8)
    state = resident.layers[lease.layer_id]
    state["payload_valid"].masked_fill_(failed[:, None], False)
    state["reusable"].masked_fill_(failed[:, None], False)
    resident.complete(
        lease.layer_id,
        lease.row_request_slots,
        lease.physical_keys,
        lease.slots,
        execution_mask=lease,
        success_mask=success,
    )


@dataclass(frozen=True)
class SharedInputs:
    query: torch.Tensor
    full_kv_cache: torch.Tensor
    full_kv_block_table: torch.Tensor
    full_kv_actual_seq: torch.Tensor
    full_q_actual_seq: torch.Tensor
    selection_topk_indices: torch.Tensor
    request_slots: torch.Tensor
    plan: object
    workspace: object
    state: object
    layer_id: int
    scale_value: float
    mapping_epoch_input: object
    full_kv_region: object = None
    full_rope_region: object = None
    host_binding: object = None
    execution_abi: str = "shared_v3"
    execution_kind: str = "NORMAL"
    load_mode: str = "FUSED"


class SharedLayer:
    def __init__(self, state, layer_id, rows, topk, full_kv, full_rope):
        from .external_sparse_planner import (
            ExternalPlanResidentState,
            expanded_resident_capacity,
        )

        if (
            full_kv.dtype != torch.bfloat16
            or full_kv.shape[-2:] != (128, 512)
            or full_rope.shape[-1] != 0
            or state.page_size != 128
        ):
            raise ValueError("external_plan requires BF16/NoPE512/RoPE0/page128")
        batch = state.batch_capacity
        if (
            not 0 < batch <= min(rows, 1024)
            or state.pool.req_generation.numel() > 32767
        ):
            raise ValueError("external_plan requires Bcap<=1024 and Qcap<=32767")
        capacity = expanded_resident_capacity(
            state.resident_capacity or topk,
            state.max_query_routes,
            topk,
            state.page_size,
        )
        if (
            capacity > 16384 or state.max_query_routes > 6
        ):
            raise ValueError("FUSED requires C<=16384 and routes<=6")
        from .config import register_shared_geometry, shared_union_width

        shared_union_width(state.max_query_routes, topk)
        from .config import register_shared_execution

        register_shared_execution(state.pool, state.load_mode, state.warmup_threshold)
        register_shared_geometry(
            state.pool,
            topk=topk,
            routes=state.max_query_routes,
            capacity=capacity,
            page_size=state.page_size,
            dtype=full_kv.dtype,
            qcap=state.generations.numel(),
            bcap=batch,
            tcap=rows,
            graph_reservation=getattr(state.pool, "_shared_fsa_graph_reservation", ()),
        )
        if state.external_resident is None:
            state.external_resident = ExternalPlanResidentState(
                state.device,
                state.generations.numel(),
                capacity,
                max_routes=state.max_query_routes,
                expanded_topk=topk,
                page_size=state.page_size,
            )
        resident = state.external_resident
        if resident.capacity != capacity:
            raise ValueError("shared layers disagree on capacity")
        resident.ensure_layer(layer_id)
        self.capacity = capacity
        self.rows = rows
        self.topk = topk
        self.capacity_locked = True
        self.guard = WorkspaceGuard(state.device)
        self.guard.layer_id = layer_id
        from .external_plan_fused import ExternalPlanFusedPlanner
        self.planner = ExternalPlanFusedPlanner(resident, rows, topk, batch)
        state.planners[layer_id] = self.planner
        self.payload = torch.empty(
            (resident.request_capacity * capacity // 128, 128, 512),
            dtype=torch.bfloat16,
            device=state.device,
        )
        # All buckets of this serialized physical layer reuse one workspace.
        # Its owner outlives every native handler; handlers capture raw pointers.
        self.native_workspace = torch.empty(
            state.shared_native_workspace_bytes, dtype=torch.uint8, device=state.device
        )
        w = self.planner.union_width
        self.ready = torch.zeros((batch, w), dtype=torch.int32, device=state.device)
        self.status = torch.zeros(batch, dtype=torch.int32, device=state.device)
        self.completion = torch.full(
            (batch, 9), -1, dtype=torch.int64, device=state.device
        )
        self.counters = torch.zeros(
            (batch, 8),
            dtype=torch.int64,
            device=state.device,
        )
        self.safe_topk = torch.full(
            (rows, 1, topk), -1, dtype=torch.int32, device=state.device
        )
        self.membership = torch.zeros(
            (rows, 16400), dtype=torch.int16, device=state.device
        )
        self.selection_table = torch.zeros(
            (rows, capacity // 128), dtype=torch.int32, device=state.device
        )
        self.selection_status = torch.zeros(
            (rows, 1, (topk + 8) // 8 * 8), dtype=torch.int32, device=state.device
        )
        if state.warmup_threshold:
            from .shared_warmup import WarmupReservation

            self.warmup = WarmupReservation(resident, layer_id, capacity)


def reserve_shared(state, layer_id, rows, topk, full_rope, full_kv):
    if state.device.type == "npu":
        from sglang.srt.hardware_backend.npu.fsa_native import require_shared_v3_capability, require_external_plan_warmup_capability
        require_shared_v3_capability()
        if state.warmup_threshold:
            require_external_plan_warmup_capability()
    current = state.layers.get(layer_id)
    if current is None:
        current = SharedLayer(
            state, layer_id, state.row_capacity or rows, topk, full_kv, full_rope
        )
        state.layers[layer_id] = current
    if rows > current.rows or topk != current.topk:
        raise ValueError("shared reservation cannot grow after initialization")
    return current


def build_shared_inputs(state, **args):
    from .fused_sparse_attention import (
        normalize_topk,
        mask_causal_topk,
        translate_logical_selection,
    )

    q = args["q_nope"]
    rope = args["q_rope"]
    full = args["full_kv"]
    if (
        q.ndim != 3
        or q.shape[1:] != (32, 512)
        or q.dtype != torch.bfloat16
        or rope.shape[-1] != 0
    ):
        raise ValueError("external_plan requires BF16 query H32/NoPE512/RoPE0")
    logical = normalize_topk(args["topk_indices"])
    rows = q.shape[0]
    topk = logical.shape[-1]
    if logical.shape[0] > rows or logical.shape[1] != 1:
        raise ValueError("shared TopK shape differs from query")
    if logical.shape[0] < rows:
        logical = torch.cat(
            (logical, logical.new_full((rows - logical.shape[0], 1, topk), -1))
        )
    ws = reserve_shared(state, args["layer_id"], rows, topk, args["full_rope"], full)
    # This precedes epoch updates, metadata generation and resident.prepare.
    ws.guard.acquire()
    try:
        capturing = (
            state.device.type == "npu" and torch.npu.is_current_stream_capturing()
        )
        if capturing:
            from .fsa_graph_lifecycle import register_capture_shared

            register_capture_shared(
                state.external_resident,
                ws.guard,
                None,
                state.mapping_epoch_input,
                (),
                state,
            )
        if not capturing:
            state.refresh_generations()
            state.refresh_mapping_epoch(args["mapping_epoch"])
        ends = (
            args["query_lengths"]
            .to(device=state.device, dtype=torch.int32)
            .contiguous()
        )
        lengths = (
            args["kv_lengths"].to(device=state.device, dtype=torch.int32).contiguous()
        )
        owners = (
            args["request_slots"]
            .to(device=state.device, dtype=torch.int64)
            .contiguous()
        )
        if (
            not 0 < owners.numel() <= ws.planner.batch_capacity
            or owners.numel() != ends.numel()
            or lengths.numel() != ends.numel()
        ):
            raise ValueError(
                "shared batch exceeds reservation or mismatches request vectors"
            )
        if args["block_table"].shape[0] != ends.numel():
            raise ValueError("shared page table must have one row per request")
        table_input = args["block_table"].to(device=state.device)
        bad_table = (table_input < -1) | (table_input > torch.iinfo(torch.int32).max)
        state.external_resident._error(bad_table, 2)
        table = table_input.masked_fill(bad_table, -1).to(torch.int32).contiguous()
        state.external_resident._error((ends[-1] != rows).reshape(1), 2)
        if state.device.type == "npu":
            from .fsa_metadata import build_external_metadata_plan

            logical, physical, _, _, cacheable = build_external_metadata_plan(
                logical,
                table,
                lengths,
                ends,
                owners,
                state.generations,
                positions=args["query_positions"],
                page_size=128,
                full_token_count=full.shape[0] * 128,
                mutable_tail_tokens=state.mutable_tail_tokens,
            )
        else:
            row = torch.arange(rows, device=state.device)
            request = torch.bucketize(row, ends, right=True).clamp(max=ends.numel() - 1)
            logical = mask_causal_topk(
                logical, request, ends, lengths, positions=args["query_positions"]
            )
            ref = translate_logical_selection(
                logical, table, lengths, ends, page_size=128
            )
            physical = ref.physical_token_ids[:, 0].masked_fill(
                ref.physical_token_ids[:, 0] >= full.shape[0] * 128, -1
            )
            logical = logical.masked_fill(physical[:, None] < 0, -1)
            cacheable = None
            if state.mutable_tail_tokens:
                cacheable = (logical[:, 0] >= 0) & (
                    logical[:, 0]
                    < (lengths[request] - state.mutable_tail_tokens).clamp_min(0)[
                        :, None
                    ]
                )
        resident = state.external_resident
        if hasattr(ws, "warmup") and cacheable is not None:
            row = torch.arange(rows, device=state.device)
            request = torch.bucketize(row, ends, right=True).clamp(
                max=owners.numel() - 1
            )
            ws.warmup.invalidate_mutable(owners[request], logical[:, 0], cacheable)
        prepare = ws.planner.validate_inputs(
            physical, ends, owners, cacheable_mask=cacheable
        )
        resident.prepare(
            args["layer_id"],
            prepare,
            state.generations[prepare.clamp(0, state.generations.numel() - 1)],
            state.mapping_epoch_input,
        )
        planner_metadata = dict(workspace=ws, logical_topk=logical,
                                kv_lengths=lengths, host_tokens=full.shape[0] * 128)
        plan = ws.planner.plan(
            args["layer_id"],
            physical,
            ends,
            owners,
            cacheable_mask=cacheable,
            mapping_epoch=state.mapping_epoch_input,
            **planner_metadata,
        )
        return SharedInputs(
            q,
            full,
            table,
            lengths,
            ends,
            logical,
            owners,
            plan,
            ws,
            state,
            args["layer_id"],
            args["scale"],
            state.mapping_epoch_input,
            load_mode=state.load_mode,
        )
    except Exception:
        if not (state.device.type == "npu" and torch.npu.is_current_stream_capturing()):
            drain_shared(ws, None)
            ws.guard.drained()
        raise


def drain_streams(streams):
    """Attempt every actual producer drain, including after an earlier failure."""
    errors = []
    seen = set()
    for stream in streams:
        identity = getattr(stream, "npu_stream", id(stream))
        if identity in seen:
            continue
        seen.add(identity)
        try:
            stream.synchronize()
        except Exception as error:
            errors.append(error)
    if errors:
        raise RuntimeError(
            "shared stream drain failed: " + "; ".join(str(e) for e in errors)
        ) from errors[0]


def drain_shared(ws, binding):
    try:
        if ws.guard.device.type == "npu":
            drain_streams(
                (
                    ws.guard.main_stream or torch.npu.current_stream(),
                    *ws.guard.side_streams,
                )
            )
    except Exception as error:
        ws.guard.fail(error)
        if binding is not None:
            binding.mark_reader_failure(error)
        raise


def run_shared(inputs, mode="fused"):
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops

    ws = inputs.workspace
    state = inputs.state
    resident = state.external_resident
    plan = inputs.plan
    lease = None
    capturing = state.device.type == "npu" and torch.npu.is_current_stream_capturing()
    binding = inputs.host_binding
    ws.guard.retained = [inputs]
    try:
        if inputs.execution_kind != "NORMAL" or inputs.load_mode != state.load_mode:
            raise ValueError(
                "shared NORMAL execution mode differs from immutable owner"
            )
        if mode != "fused":
            raise ValueError("external_plan requires fused mode; reference execution was removed")
        if capturing:
            from .fsa_graph_lifecycle import register_capture_shared

            register_capture_shared(
                resident, ws.guard, binding, inputs.mapping_epoch_input, (), state
            )
        if binding is not None:
            binding.check_reader_health()
        resident.record_plan(
            inputs.layer_id, plan.row_request_slots, plan.physical_keys, plan.slots
        )
        lease = resident.begin_execution(
            inputs.layer_id,
            plan.row_request_slots,
            physical_keys=plan.physical_keys,
            slots=plan.slots,
            cacheable_mask=plan.cacheable_mask,
        )
        ws.guard.retained.append(lease)
        if capturing:
            register_capture_shared(
                resident, ws.guard, binding, inputs.mapping_epoch_input, (lease,), state
            )
        expected = expected_identity(
            resident, lease, inputs.request_slots, state.shared_instance
        )
        b = inputs.request_slots.numel()
        t = inputs.query.shape[0]
        ready = ws.ready[:b]
        status = ws.status[:b]
        completion = ws.completion[:b]
        counters = ws.counters[:b]
        region = inputs.full_kv_region
        if region is None:
            raise ValueError("external_plan requires a registered Host full-KV source")
        if (
            inputs.full_kv_cache.device.type != "cpu"
            or not inputs.full_kv_cache.is_contiguous()
            or inputs.full_kv_cache.data_ptr() != region.host_ptr
            or inputs.full_kv_cache.nbytes > region.nbytes
            or region.device_ptr <= 0
            or (state.device.type == "npu" and region.device_id != state.device.index)
        ):
            raise ValueError("shared full-KV metadata differs from Host registration")
        # Immutable expected is never passed as writable native completion.
        safe_owners = inputs.request_slots.clamp(0, resident.request_capacity - 1)
        batch_active = (
            (inputs.request_slots > 0)
            & (inputs.request_slots < resident.request_capacity)
            & lease.active[safe_owners]
        )
        native_owners = inputs.request_slots.masked_fill(~batch_active, 0)
        row_batch = plan.row_to_request.long().clamp(0, b - 1)
        row_active = (plan.row_to_request >= 0) & batch_active[row_batch]
        logical_source = ws.safe_topk[:t]
        native_logical = logical_source.masked_fill(~row_active[:, None, None], -1)
        native_rows = plan.row_to_request.masked_fill(~row_active, -1)
        native_union = plan.union_index.masked_fill(~row_active[:, None], -1)
        common = (
            inputs.query,
            ws.payload,
            inputs.full_kv_cache,
            region.device_ptr,
            inputs.full_kv_block_table,
            inputs.full_kv_actual_seq,
            inputs.full_q_actual_seq,
            native_logical,
            native_owners,
            native_rows,
            native_union,
            plan.union_keys,
            plan.union_slots,
            plan.union_count,
            plan.union_miss,
            plan.union_owner,
            plan.union_cacheable,
            expected,
            ready,
            status,
            completion,
            counters,
            ws.safe_topk[:t],
            ws.membership[:t],
            ws.selection_table[:t],
            ws.selection_status[:t],
            ws.native_workspace,
        )
        # Retain every native input through checked completion or a
        # failed-drain quarantine. These references precede the first enqueue.
        ws.guard.retained = [inputs, lease, expected, common]
        output = native_ops().npu_fused_sparse_attention_overlap(
            *common, ws.capacity, inputs.scale_value, 3
        )
        complete_shared(
            resident,
            lease,
            inputs.request_slots,
            expected,
            completion,
            status,
            ready,
            plan.union_count,
        )
        if hasattr(ws, "warmup"):
            ws.warmup.refresh_marker(resident, plan)
        if capturing:
            from .fsa_graph_lifecycle import register_capture_shared

            register_capture_shared(
                resident,
                ws.guard,
                binding,
                inputs.mapping_epoch_input,
                (
                    lease,
                    expected,
                    plan,
                    inputs.selection_topk_indices,
                    native_owners,
                    native_logical,
                    native_rows,
                    native_union,
                    ws.native_workspace,
                ),
                state,
            )
        elif binding is not None and state.device.type == "npu":
            event = torch.npu.Event()
            event.record(torch.npu.current_stream())
            binding.record_reader(event)
        ws.guard.release()
        ws.guard.retained = []
        if not capturing:
            register_shared_resident(resident)
        if not capturing and _FORWARD.get() is None:
            check_shared_residents((resident,))
        return output
    except Exception:
        if capturing:
            # Outer graph owner drains after leaving capture, then cancels only
            # retained leases. Synchronizing inside capture is not legal.
            raise
        drain_shared(ws, binding)
        try:
            if lease is not None:
                resident.cancel_execution(
                    inputs.layer_id, lease.row_request_slots, execution_mask=lease
                )
            # Cancellation queues device writes. Retain producer ownership until
            # those writes, too, are drained; a launch-only drain is insufficient.
            drain_shared(ws, binding)
            ws.guard.drained()
            ws.guard.retained = []
        except Exception as error:
            ws.guard.fail(error)
            if binding is not None:
                binding.mark_reader_failure(error)
            raise
        raise


def storage_inventory(*objects):
    """Actual reachable tensor storages, deduplicated by device/base allocation.

    Only supplied owners are walked; request pool/backing/native allocations are
    external references and must be supplied explicitly when desired.
    """
    seen_objects = set()
    storages = {}
    excluded = {
        "pool",
        "host_binding",
        "full_kv_region",
        "full_rope_region",
        "device",
        "guard",
        "state",
        "resident",
        "workspace",
    }

    def walk(value, path):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if storage.nbytes():
                storages.setdefault(
                    key,
                    {
                        "device": str(value.device),
                        "bytes": storage.nbytes(),
                        "paths": [],
                    },
                )["paths"].append(path)
            return
        if id(value) in seen_objects:
            return
        seen_objects.add(id(value))
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{path}.{k}")
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")
        elif hasattr(value, "__dict__"):
            for k, v in vars(value).items():
                if k not in excluded:
                    walk(v, f"{path}.{k}")

    for i, obj in enumerate(objects):
        walk(obj, f"owner[{i}]")
    return {
        "cpu_bytes": sum(v["bytes"] for v in storages.values() if v["device"] == "cpu"),
        "device_bytes": sum(
            v["bytes"] for v in storages.values() if v["device"] != "cpu"
        ),
        "storages": list(storages.values()),
    }

