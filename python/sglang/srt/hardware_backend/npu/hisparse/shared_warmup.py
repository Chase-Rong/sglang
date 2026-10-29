"""Optional eager committed-prefix warming, independent of Attention queries.

FILL_ONLY uses an independent direct native helper and no Attention OPP.
A device stamp prevents repeated whole-prefix loads after ordinary eviction.
The separate six-column marker is only a current-residency assertion.
"""

from contextlib import nullcontext
import torch

from .external_plan_device import ExternalDeviceSelectionPlan


class WarmupPrefixPlanner:
    """Assign one validated committed prefix, preserving every resident hit.

    There is no TopK union, multi-route policy or CLOCK/LRU state. The caller
    has checked prefix mappings are unique, in bounds, committed and published.
    Non-hit destinations are the unprotected slots in ascending slot order.
    """

    def __init__(self, resident, capacity):
        self.resident, self.capacity = resident, capacity
        self.union_width = capacity
        device = resident.device
        self.union_keys = torch.full((1, capacity), -1, dtype=torch.int64, device=device)
        self.union_slots = torch.full((1, capacity), -1, dtype=torch.int32, device=device)
        self.union_owner = torch.full_like(self.union_slots, -1)
        self.union_miss = torch.zeros_like(self.union_slots, dtype=torch.bool)
        self.union_cacheable = torch.zeros_like(self.union_miss)
        self.union_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.slots = torch.full_like(self.union_keys, -1)
        self.miss_mask = torch.zeros_like(self.union_miss)
        self.writer_mask = torch.zeros_like(self.union_miss)

    def plan(self, layer_id, physical, ends, owners, *, mapping_epoch):
        resident = self.resident
        layer = resident.ensure_layer(layer_id)
        owner = owners[0]
        active = (~resident.blocked_requests[owner]
                  & (resident.release_state[owner] == resident.RELEASE_ACTIVE)
                  & (layer["in_flight"][owner] == 0))
        valid = (physical[0] >= 0) & active
        sentinel = torch.iinfo(torch.int64).max
        current = layer["identity"][owner]
        eligible = (layer["slot_allocated"][owner] & layer["payload_valid"][owner]
                    & layer["reusable"][owner])
        sorted_keys, indices = torch.sort(current.masked_fill(~eligible, sentinel))
        position = torch.searchsorted(sorted_keys, physical[0]).clamp(max=self.capacity - 1)
        hits = valid & (sorted_keys[position] == physical[0])
        hit_slots = indices[position]
        protected = torch.zeros(self.capacity + 1, dtype=torch.bool, device=physical.device)
        protected.scatter_(0, hit_slots.masked_fill(~hits, self.capacity), hits)
        available = torch.arange(self.capacity, device=physical.device).masked_fill(protected[:-1], self.capacity)
        available = available.sort().values
        misses = valid & ~hits
        miss_rank = misses.long().cumsum(0).sub(1).clamp_min(0)
        slots = torch.where(hits, hit_slots, available[miss_rank]).masked_fill(~valid, -1)
        # The fill-only helper consumes a sorted physical union while each
        # owner entry retains its original committed logical position.
        order = physical[0].masked_fill(~valid, sentinel).sort().indices
        union_valid = valid[order]
        self.union_keys.copy_(physical[:, order].masked_fill(~union_valid[None], -1))
        self.union_slots.copy_(slots[order][None].to(torch.int32))
        self.union_owner.copy_(order[None].to(torch.int32).masked_fill(~union_valid[None], -1))
        self.union_miss.copy_(misses[order][None])
        self.union_cacheable.copy_(union_valid[None])
        self.union_count.copy_(valid.sum().reshape(1).to(torch.int32))
        self.slots.copy_(slots[None])
        self.miss_mask.copy_(misses[None])
        self.writer_mask.copy_(misses[None])
        return ExternalDeviceSelectionPlan(
            physical_keys=physical, slots=self.slots, miss_mask=self.miss_mask,
            cacheable_mask=valid[None], writer_mask=self.writer_mask,
            row_request_slots=owners.masked_fill(~active, 0), union_keys=self.union_keys,
            union_slots=self.union_slots, union_count=self.union_count, layer_id=layer_id,
            row_to_request=torch.zeros(1, dtype=torch.int32, device=physical.device),
            union_index=order.argsort()[None].to(torch.int32).masked_fill(~valid[None], -1), overflow=(~active).reshape(1),
            union_owner=self.union_owner, union_miss=self.union_miss,
            union_cacheable=self.union_cacheable,
            mapping_epoch=resident._epoch_tensor(mapping_epoch).clone())


class WarmupReservation:
    def __init__(self, resident, layer_id, capacity):
        self.layer_id = layer_id
        # Eager microbatch B_w=1 bounds memory independently of serving Bcap.
        self.planner = WarmupPrefixPlanner(resident, capacity)
        self.ready = torch.zeros(
            (1, capacity), dtype=torch.int32, device=resident.device
        )
        self.status = torch.zeros(1, dtype=torch.int32, device=resident.device)
        self.completion = torch.full(
            (1, 9), -1, dtype=torch.int64, device=resident.device
        )
        self.counters = torch.zeros((1, 2), dtype=torch.int64, device=resident.device)
        self.marker = torch.full(
            (resident.request_capacity, 6),
            -1,
            dtype=torch.int64,
            device=resident.device,
        )
        # generation, epoch, initial length, content version, completion serial
        self.stamp = torch.full(
            (resident.request_capacity, 5),
            -1,
            dtype=torch.int64,
            device=resident.device,
        )
        self.content_version = torch.zeros(
            resident.request_capacity, dtype=torch.int64, device=resident.device
        )

    def invalidate(self, owners):
        owners = owners.to(device=self.content_version.device, dtype=torch.long)
        self.content_version[owners] += 1
        self.marker[owners] = -1
        self.stamp[owners] = -1
        layer = self.planner.resident.layers[self.layer_id]
        layer["payload_valid"][owners] = False
        layer["reusable"][owners] = False

    def invalidate_mutable(self, owners, logical, cacheable):
        """Only selected mutable positions inside an already warmed prefix."""
        safe = owners.clamp(0, self.marker.shape[0] - 1).long()
        length = self.stamp[safe, 2]
        affected = ((logical >= 0) & (logical < length[:, None]) & ~cacheable).any(1)
        hits = torch.zeros_like(self.content_version, dtype=torch.int32)
        hits.scatter_add_(0, safe, affected.to(torch.int32))
        mask = hits > 0
        self.content_version.add_(mask.to(torch.int64))
        self.marker.masked_fill_(mask[:, None], -1)
        self.stamp.masked_fill_(mask[:, None], -1)

    def refresh_marker(self, resident, plan):
        """Misses conservatively revoke residency, preserving one-shot success."""
        stale = (self.marker[:, 1] != resident.generation) | (
            self.marker[:, 2] != resident.mapping_epoch
        )
        owners = plan.row_request_slots.long()
        safe = owners.clamp(0, self.marker.shape[0] - 1)
        misses = plan.miss_mask.any(1) & (owners > 0)
        counts = torch.zeros_like(self.content_version, dtype=torch.int32)
        counts.scatter_add_(0, safe, misses.to(torch.int32))
        stale |= counts > 0
        self.marker.masked_fill_(stale[:, None], -1)


def warm_committed_prefix(
    state,
    layer_id,
    *,
    owner,
    generation,
    committed_length,
    mapping_epoch,
    block_table,
    full_kv,
    region,
    binding,
    publication_length=None,
    publication_event=None,
):
    """Return True only for a fresh successful exact completion; False means skip.

    Callers supply Req.kv.kv_committed_len and the binding's publication proof.
    No seq_lens, allocation length, speculative width or fake query is accepted.
    """
    from .shared_execution import (
        expected_identity,
        complete_shared,
        drain_shared,
        check_shared_residents,
    )
    from sglang.srt.hardware_backend.npu.fsa_native import (
        native_ops,
        require_external_plan_warmup_capability,
    )

    if not state.warmup_threshold:
        return False
    if (
        not 0
        < committed_length
        <= min(state.warmup_threshold, state.external_resident.capacity)
    ):
        return False
    if (
        publication_length is None
        or publication_length < committed_length
        or publication_event is None
    ):
        return False
    if binding is None or getattr(binding, "ready_event", None) is None:
        return False
    if state.device.type == "npu":
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("FILL_ONLY warmup is eager only")
        require_external_plan_warmup_capability()
    if not 0 < owner < state.generations.numel():
        return False
    ws = state.layers[layer_id]
    warm = ws.warmup
    if ws.guard.active or ws.guard.quarantine is not None:
        return False
    state.refresh_generations()
    if int(state.generations[owner]) != generation:
        return False
    stamp = warm.stamp[owner]
    if (
        int(stamp[0]) == generation
        and int(stamp[1]) == mapping_epoch
        and int(stamp[3]) == int(warm.content_version[owner])
    ):
        # Prefix growth is handled lazily; a prefix shrink is content invalidation.
        if committed_length >= int(stamp[2]):
            return False
        warm.invalidate(torch.tensor([owner], device=state.device))
    binding.check_reader_health()
    if binding.mapping_epoch != mapping_epoch:
        return False
    if (
        full_kv.device.type != "cpu"
        or not full_kv.is_contiguous()
        or full_kv.data_ptr() != region.host_ptr
        or full_kv.nbytes > region.nbytes
        or region.device_ptr <= 0
    ):
        raise ValueError("warmup source differs from registered Host backing")
    if state.device.type == "npu" and region.device_id != state.device.index:
        raise ValueError("warmup Host registration belongs to another physical device")
    capacity = ws.capacity
    if block_table.dtype not in (torch.int32, torch.int64):
        return False
    wide_table = block_table.to(device=state.device).reshape(1, -1)
    # Validate original wide values before narrowing; 2**32 must never wrap
    # to Host page0 (or any other valid alternate physical address).
    if bool(((wide_table < -1) | (wide_table > torch.iinfo(torch.int32).max)).any()):
        return False
    table = wide_table.to(dtype=torch.int32).contiguous()
    pos = torch.arange(capacity, device=state.device)
    if table.shape[1] == 0 or table.shape[1] * 128 < committed_length:
        return False
    safe_page = (pos // 128).clamp(max=table.shape[1] - 1)
    pages = table[0, safe_page].long()
    keys = pages * 128 + pos % 128
    valid = pos < committed_length
    # FILL_ONLY requires count=length: aliases and invalid mappings are
    # rejected, never deduplicated/truncated into a smaller accepted prefix.
    selected = keys[valid]
    if (
        bool(((pages[valid] < 0) | (selected >= full_kv.shape[0] * 128)).any())
        or selected.unique().numel() != committed_length
    ):
        return False
    physical = keys.masked_fill(~valid, -1).reshape(1, capacity)
    owners = torch.tensor([owner], dtype=torch.int64, device=state.device)
    ends = torch.ones(1, dtype=torch.int32, device=state.device)
    lengths = torch.tensor([committed_length], dtype=torch.int32, device=state.device)
    lease = None
    content_version = int(warm.content_version[owner])
    pool_generation = int(state.pool.req_generation[owner])
    ws.guard.acquire()
    try:
        if state.device.type == "npu":
            torch.npu.current_stream().wait_event(publication_event)
        state.refresh_mapping_epoch(mapping_epoch)
        resident = state.external_resident
        resident.prepare(
            layer_id, owners, state.generations[owners], state.mapping_epoch_input
        )
        plan = warm.planner.plan(
            layer_id, physical, ends, owners, mapping_epoch=state.mapping_epoch_input
        )
        resident.record_plan(layer_id, owners, plan.physical_keys, plan.slots)
        lease = resident.begin_execution(
            layer_id,
            owners,
            physical_keys=plan.physical_keys,
            slots=plan.slots,
            cacheable_mask=plan.cacheable_mask,
        )
        expected = expected_identity(resident, lease, owners, state.shared_instance)
        ws.guard.retained = [lease, expected, plan, full_kv, binding, physical, table]
        native_ops().external_plan_fill_only(
            ws.payload,
            full_kv,
            region.device_ptr,
            table,
            lengths,
            owners,
            plan.union_keys,
            plan.union_slots,
            plan.union_count,
            plan.union_miss,
            plan.union_owner,
            plan.union_cacheable,
            expected,
            warm.ready,
            warm.status,
            warm.completion,
            warm.counters,
            capacity,
        )
        complete_shared(
            resident,
            lease,
            owners,
            expected,
            warm.completion,
            warm.status,
            warm.ready,
            plan.union_count,
        )
        check_shared_residents((resident,))
        # Outside-capture checked completion synchronizes device metadata. A
        # concurrent generation/remap cannot publish a stale marker afterward.
        if (
            binding.mapping_epoch != mapping_epoch
            or int(state.pool.req_generation[owner]) != pool_generation
            or int(warm.content_version[owner]) != content_version
        ):
            warm.invalidate(owners)
            raise RuntimeError("warmup identity changed before marker publication")
        if not bool((warm.completion == expected).all()) or int(warm.status[0]) != 0:
            raise RuntimeError("warmup exact completion failed")
        warm.marker[owner] = torch.stack(
            (
                owners[0],
                expected[0, 4],
                expected[0, 5],
                lengths[0].long(),
                expected[0, 8],
                expected[0, 6],
            )
        )
        warm.stamp[owner] = torch.stack(
            (
                expected[0, 4],
                expected[0, 5],
                lengths[0].long(),
                warm.content_version[owner],
                expected[0, 6],
            )
        )
        if state.device.type == "npu":
            event = torch.npu.Event()
            event.record(torch.npu.current_stream())
            binding.record_reader(event)
        ws.guard.release()
        ws.guard.retained = []
        return True
    except Exception as original_error:
        # Publication is transactional through final reader/fence bookkeeping.
        # Revoke on the owning main stream before draining even when the
        # native fill itself succeeded. Failed drains still quarantine it.
        revocation_errors = []
        for record in (warm.marker, warm.stamp):
            try:
                record[owner].fill_(-1)
            except Exception as error:
                revocation_errors.append(error)
        # Even failed marker writes cannot bypass either producer drain.
        # Warmup is eager: after the first drain the lease terminal tensor is
        # safe to inspect, distinguishing a checked-complete lease from one
        # still requiring cancellation.
        try:
            drain_shared(ws, binding)
            if lease is not None and bool((lease.active & ~lease.terminal).any()):
                state.external_resident.cancel_execution(
                    layer_id, lease.row_request_slots, execution_mask=lease
                )
            drain_shared(ws, binding)
            if revocation_errors:
                raise RuntimeError(
                    "warmup marker revocation failed after drained cleanup"
                ) from revocation_errors[0]
            ws.guard.drained()
            ws.guard.retained = []
        except Exception as cleanup_error:
            failure = RuntimeError(
                f"warmup failed: {original_error}; cleanup failed: {cleanup_error}"
            )
            failure.cleanup_errors = (*revocation_errors, cleanup_error)
            ws.guard.fail(failure)
            binding.mark_reader_failure(failure)
            raise failure from original_error
        raise
