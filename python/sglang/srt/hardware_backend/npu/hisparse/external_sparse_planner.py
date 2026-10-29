"""Request-owned resident state and execution leases for external_plan."""

from dataclasses import dataclass

import torch


def align_up(value: int, alignment: int) -> int:
    """Return ``value`` rounded up to a positive page/alignment boundary."""

    if value < 0 or alignment <= 0:
        raise ValueError("capacity and alignment must be non-negative/positive")
    return (value + alignment - 1) // alignment * alignment


def expanded_resident_capacity(
    configured_capacity: int,
    max_routes: int,
    expanded_topk: int,
    page_size: int,
) -> int:
    """Compute the external_plan resident capacity without LIM assumptions."""

    if configured_capacity < 0 or max_routes <= 0 or expanded_topk <= 0:
        raise ValueError("invalid external_plan resident geometry")
    capacity = align_up(max(configured_capacity, max_routes * expanded_topk), page_size)
    if capacity <= 0 or capacity > 32767:
        raise ValueError(
            "external_plan resident capacity must fit the native int16 ABI"
        )
    return capacity


@dataclass
class ExternalExecutionLease:
    """One begin/terminal pair, with snapshots recomputed by each graph replay.

    The per-layer serial never resets on retirement. The device terminal mask
    makes repeated complete/cancel harmless, including failures after publish.
    """

    resident: object
    layer_id: int
    active: torch.Tensor
    generation: torch.Tensor
    mapping_epoch: torch.Tensor
    planner_version: torch.Tensor
    serial: torch.Tensor
    terminal: torch.Tensor
    row_request_slots: torch.Tensor
    physical_keys: torch.Tensor | None = None
    slots: torch.Tensor | None = None
    cacheable_mask: torch.Tensor | None = None


class ExternalPlanResidentState:
    """Request/layer resident identity and validity owner for external_plan.

    Slot allocation, payload validity and reusable identity remain separate.
    The fused planner and prefix warmup share this lifecycle owner.
    """

    RELEASE_ACTIVE = 0
    RELEASE_PENDING = 1
    RELEASED = 2

    def __init__(
        self,
        device,
        request_capacity: int,
        capacity: int,
        *,
        max_routes: int,
        expanded_topk: int,
        page_size: int,
    ):
        if request_capacity <= 0:
            raise ValueError("request capacity must be positive")
        expected = expanded_resident_capacity(
            capacity, max_routes, expanded_topk, page_size
        )
        if expected != capacity:
            raise ValueError(
                "external_plan capacity must already include route expansion"
            )
        self.device = torch.device(device)
        self.request_capacity = request_capacity
        self.capacity = capacity
        self.max_routes = max_routes
        self.expanded_topk = expanded_topk
        self.page_size = page_size
        self.layers = {}
        self._mapping_epoch = torch.full(
            (request_capacity,), -1, dtype=torch.int64, device=self.device
        )
        self._generation = torch.full(
            (request_capacity,), -1, dtype=torch.int64, device=self.device
        )
        self._planner_version = torch.zeros(
            (request_capacity,), dtype=torch.int64, device=self.device
        )
        self._release_state = torch.zeros(
            (request_capacity,), dtype=torch.int8, device=self.device
        )
        # Device-side lifecycle violations remain observable without forcing a
        # host read (which is illegal during graph capture).
        self.lifecycle_error = torch.zeros((), dtype=torch.int32, device=self.device)
        self.blocked_requests = torch.zeros(
            request_capacity, dtype=torch.bool, device=self.device
        )

    def ensure_layer(self, layer_id: int):
        state = self.layers.get(layer_id)
        if state is None:
            shape = (self.request_capacity, self.capacity)
            state = {
                "identity": torch.full(
                    shape, -1, dtype=torch.int64, device=self.device
                ),
                "slot_allocated": torch.zeros(
                    shape, dtype=torch.bool, device=self.device
                ),
                "payload_valid": torch.zeros(
                    shape, dtype=torch.bool, device=self.device
                ),
                "reusable": torch.zeros(shape, dtype=torch.bool, device=self.device),
                "in_flight": torch.zeros(
                    (self.request_capacity,), dtype=torch.int32, device=self.device
                ),
                "execution_serial": torch.zeros(
                    self.request_capacity, dtype=torch.int64, device=self.device
                ),
            }
            self.layers[layer_id] = state
        return state

    @property
    def generation(self):
        return self._generation

    @property
    def mapping_epoch(self):
        return self._mapping_epoch

    @property
    def planner_version(self):
        return self._planner_version

    @property
    def release_state(self):
        return self._release_state

    def _epoch_tensor(self, mapping_epoch):
        epoch = torch.as_tensor(mapping_epoch, dtype=torch.int64, device=self.device)
        if epoch.numel() != 1:
            raise ValueError("mapping epoch must be a scalar")
        return epoch.reshape(())

    def _error(self, invalid, code=1):
        self.lifecycle_error.copy_(
            torch.maximum(self.lifecycle_error, invalid.any().to(torch.int32) * code)
        )

    def prepare(self, layer_id, request_slots, generations, mapping_epoch):
        """Adopt identities only for unique owners, preserving live executions."""
        self.ensure_layer(layer_id)
        slots = request_slots.to(device=self.device, dtype=torch.long).reshape(-1)
        generations = generations.to(device=self.device, dtype=torch.int64).reshape(-1)
        if slots.numel() != generations.numel():
            raise ValueError("resident request and generation counts must match")
        if self.device.type == "cpu":
            if slots.numel() and ((slots < 0) | (slots >= self.request_capacity)).any():
                raise ValueError("resident request slot is outside the request pool")
            safe = slots.clamp(min=0, max=self.request_capacity - 1)
            busy = torch.zeros_like(slots, dtype=torch.bool)
            for layer in self.layers.values():
                busy |= layer["in_flight"][safe] > 0
            if ((slots > 0) & busy & (self._generation[safe] != generations)).any():
                raise RuntimeError(
                    "cannot reuse an external_plan request slot while FSA work is in flight"
                )
        return self._prepare_device(slots, generations, mapping_epoch)

    def _device_owner_mask(self, slots):
        valid = (slots > 0) & (slots < self.request_capacity)
        safe = slots.clamp(min=0, max=self.request_capacity - 1)
        counts = torch.zeros(
            self.request_capacity, dtype=torch.int32, device=self.device
        )
        counts.scatter_add_(0, safe, valid.to(torch.int32))
        return counts > 0

    def _prepare_device(self, slots, generations, mapping_epoch):
        safe = slots.clamp(min=0, max=self.request_capacity - 1)
        valid = (slots > 0) & (slots < self.request_capacity)
        counts = torch.zeros(
            self.request_capacity, dtype=torch.int32, device=self.device
        )
        counts.scatter_add_(0, safe, valid.to(torch.int32))
        unique = valid & (counts[safe] == 1) & (generations >= 0)
        self._error(
            (valid & ~unique) | (slots < 0) | (slots >= self.request_capacity), 2
        )
        active = self._device_owner_mask(slots.masked_fill(~unique, 0))
        current = self._generation.clone()
        self._device_scatter(
            current,
            safe.masked_fill(~unique, self.request_capacity),
            generations,
            invalid_value=-1,
        )
        epoch = self._epoch_tensor(mapping_epoch)
        changed_generation = active & (current != self._generation)
        changed_mapping = active & (self._mapping_epoch != epoch)
        busy = torch.zeros_like(active)
        for layer in self.layers.values():
            busy |= layer["in_flight"] > 0
        blocked = changed_generation & busy
        self._error(blocked)
        self.blocked_requests.copy_(torch.where(active, blocked, self.blocked_requests))
        active &= ~blocked
        changed_generation &= ~blocked
        changed_mapping &= ~blocked
        for layer in self.layers.values():
            layer["identity"].masked_fill_(changed_generation[:, None], -1)
            layer["slot_allocated"].masked_fill_(changed_generation[:, None], False)
            layer["payload_valid"].masked_fill_(changed_generation[:, None], False)
            layer["reusable"].masked_fill_(
                (changed_generation | changed_mapping)[:, None], False
            )
        self._generation.copy_(torch.where(active, current, self._generation))
        self._mapping_epoch.copy_(torch.where(active, epoch, self._mapping_epoch))
        self._planner_version.add_(
            changed_generation.to(torch.int64) + changed_mapping.to(torch.int64)
        )
        self._release_state.masked_fill_(changed_generation, self.RELEASE_ACTIVE)
        return self._planner_version[safe]

    def _plan_tensors(self, row_request_slots, physical_keys, slots):
        owners = row_request_slots.to(device=self.device, dtype=torch.long).reshape(-1)
        keys = physical_keys.to(device=self.device, dtype=torch.int64)
        destinations = slots.to(device=self.device, dtype=torch.long)
        if (
            keys.ndim != 2
            or keys.shape != destinations.shape
            or keys.shape[0] != owners.numel()
        ):
            raise ValueError("resident plan rows and request owners disagree")
        return owners, keys, destinations

    def _device_destinations(self, owners, keys, destinations, active=None):
        safe_owner = owners.clamp(min=0, max=self.request_capacity - 1)
        valid = (
            (owners[:, None] > 0)
            & (owners[:, None] < self.request_capacity)
            & (keys >= 0)
            & (keys < torch.iinfo(torch.int64).max)
            & (destinations >= 0)
            & (destinations < self.capacity)
            & ~self.blocked_requests[safe_owner, None]
        )
        if active is not None:
            valid &= active[safe_owner, None]
        padding = self.request_capacity * self.capacity
        index = safe_owner[:, None] * self.capacity + destinations.clamp(
            0, self.capacity - 1
        )
        return index.masked_fill(~valid, padding).reshape(-1), valid

    def _device_scatter(self, target, index, values, *, invalid_value):
        padded = torch.cat((target.reshape(-1), target.new_full((1,), invalid_value)))
        padded.scatter_(0, index, values.reshape(-1))
        target.copy_(padded[:-1].view_as(target))

    def record_plan(self, layer_id, row_request_slots, physical_keys, slots):
        """Reserve destinations before begin; a live execution owns its slots."""
        state = self.ensure_layer(layer_id)
        owners, keys, destinations = self._plan_tensors(
            row_request_slots, physical_keys, slots
        )
        requested = self._device_owner_mask(owners)
        busy = requested & (state["in_flight"] > 0)
        self._error(busy)
        active = (state["in_flight"] == 0) & (
            self._release_state == self.RELEASE_ACTIVE
        )
        index, valid = self._device_destinations(owners, keys, destinations, active)
        safe_owner = owners.clamp(min=0, max=self.request_capacity - 1)
        safe_slot = destinations.clamp(min=0, max=self.capacity - 1)
        old = state["identity"][safe_owner[:, None], safe_slot]
        hit = valid & state["payload_valid"][safe_owner[:, None], safe_slot]
        hit &= state["reusable"][safe_owner[:, None], safe_slot] & (old == keys)
        self._device_scatter(state["slot_allocated"], index, valid, invalid_value=False)
        self._device_scatter(state["payload_valid"], index, hit, invalid_value=False)
        self._device_scatter(state["reusable"], index, hit, invalid_value=False)

    def begin_execution(
        self,
        layer_id,
        row_request_slots,
        *,
        physical_keys=None,
        slots=None,
        cacheable_mask=None,
    ):
        """Acquire a one-shot lease; no padding/blocked/empty row owns a reference.

        Plan tensors are mandatory: the lease snapshots immutable keys, slots,
        and reuse eligibility because planner workspace may be reused.
        """
        state = self.ensure_layer(layer_id)
        owners = row_request_slots.to(self.device, dtype=torch.long).reshape(-1).clone()
        keys = destinations = None
        if physical_keys is None or slots is None:
            raise ValueError("execution keys and destinations are required for a lease")
        if cacheable_mask is not None and (
            cacheable_mask.shape != physical_keys.shape
            or cacheable_mask.dtype != torch.bool
        ):
            raise ValueError("execution cacheability must match physical keys")
        if physical_keys is not None:
            owners, keys, destinations = self._plan_tensors(
                owners, physical_keys, slots
            )
            _, valid = self._device_destinations(owners, keys, destinations)
            owners = owners.masked_fill(~valid.any(dim=1), 0)
            keys, destinations = keys.clone(), destinations.clone()
        requested = self._device_owner_mask(owners)
        busy = requested & (state["in_flight"] > 0)
        self._error(busy)
        active = requested & ~self.blocked_requests & ~busy
        active &= self._release_state == self.RELEASE_ACTIVE
        # Construct every fallible snapshot before acquiring a device ref.
        # A clone/allocation failure cannot strand a live execution.
        lease = ExternalExecutionLease(
            self,
            layer_id,
            active,
            self._generation.clone(),
            self._mapping_epoch.clone(),
            self._planner_version.clone(),
            state["execution_serial"].clone() + active.to(torch.int64),
            torch.zeros_like(active),
            owners,
            keys,
            destinations,
            None if cacheable_mask is None else cacheable_mask.to(self.device).clone(),
        )
        state["execution_serial"].add_(active.to(torch.int64))
        state["in_flight"].add_(active.to(torch.int32))
        return lease

    def _consume_mask(self, layer_id, lease):
        if (
            not isinstance(lease, ExternalExecutionLease)
            or lease.resident is not self
            or lease.layer_id != layer_id
        ):
            raise ValueError("complete/cancel requires this layer's execution lease")
        state = self.ensure_layer(layer_id)
        return (
            lease.active
            & ~lease.terminal
            & (lease.serial == state["execution_serial"])
            & (state["in_flight"] > 0)
        )

    def _finish_execution(self, layer_id, lease, executing):
        self.layers[layer_id]["in_flight"].sub_(executing.to(torch.int32))
        lease.terminal.logical_or_(lease.active)
        self._finalize_releases(lease.row_request_slots)

    def _finalize_releases(self, slots):
        """Release a request only after every layer has drained its work."""
        pending = self._release_state == self.RELEASE_PENDING
        busy = torch.zeros_like(pending)
        for state in self.layers.values():
            busy |= state["in_flight"] > 0
        release = pending & ~busy
        for state in self.layers.values():
            state["identity"].masked_fill_(release[:, None], -1)
            state["slot_allocated"].masked_fill_(release[:, None], False)
            state["payload_valid"].masked_fill_(release[:, None], False)
            state["reusable"].masked_fill_(release[:, None], False)
        self._release_state.masked_fill_(release, self.RELEASED)

    def complete(
        self,
        layer_id,
        row_request_slots,
        physical_keys,
        slots,
        *,
        cacheable_mask=None,
        mapping_epoch=None,
        execution_mask=None,
        success_mask=None,
    ):
        """Publish only a current successful lease, then consume it exactly once.

        A stale epoch may drain its own reference, but may not write identity,
        validity, destinations, or a newer execution's reference. The caller
        orders this method after native work and records its Host reader fence.
        """
        lease = execution_mask
        executing = self._consume_mask(layer_id, lease)
        state = self.layers[layer_id]
        owners, keys, destinations = self._plan_tensors(
            row_request_slots, physical_keys, slots
        )
        if lease.physical_keys is not None:
            owners, keys, destinations = (
                lease.row_request_slots,
                lease.physical_keys,
                lease.slots,
            )
        cacheable_mask = lease.cacheable_mask
        publish = (
            executing
            & (lease.generation == self._generation)
            & (lease.mapping_epoch == self._mapping_epoch)
            & (lease.planner_version == self._planner_version)
            & (self._release_state == self.RELEASE_ACTIVE)
        )
        if success_mask is not None:
            publish &= success_mask
        if mapping_epoch is not None:
            publish &= lease.mapping_epoch == self._epoch_tensor(mapping_epoch)
        index, valid = self._device_destinations(owners, keys, destinations, publish)
        stable = valid.clone() if cacheable_mask is None else valid & cacheable_mask
        unstable = torch.zeros(
            self.request_capacity * self.capacity + 1,
            dtype=torch.int32,
            device=self.device,
        )
        unstable.scatter_add_(0, index, (valid & ~stable).reshape(-1).to(torch.int32))
        stable = valid & (unstable[index].view_as(keys) == 0)
        self._device_scatter(state["identity"], index, keys, invalid_value=-1)
        self._device_scatter(state["slot_allocated"], index, valid, invalid_value=False)
        self._device_scatter(state["payload_valid"], index, valid, invalid_value=False)
        self._device_scatter(state["reusable"], index, stable, invalid_value=False)
        self._finish_execution(layer_id, lease, executing)

    def cancel_execution(self, layer_id, row_request_slots, *, execution_mask=None):
        """Cancel after the caller drains native writes; never cancel twice."""
        lease = execution_mask
        executing = self._consume_mask(layer_id, lease)
        state = self.layers[layer_id]
        # Failed native work may have partially overwritten existing payload.
        state["payload_valid"].masked_fill_(executing[:, None], False)
        state["reusable"].masked_fill_(executing[:, None], False)
        self._finish_execution(layer_id, lease, executing)

    def retire(self, request_slots):
        """Prevent reuse immediately, then clear only after all layers drain."""
        slots = request_slots.to(self.device, dtype=torch.long).reshape(-1)
        retiring = self._device_owner_mask(slots)
        self._release_state.masked_fill_(retiring, self.RELEASE_PENDING)
        for state in self.layers.values():
            state["reusable"].masked_fill_(retiring[:, None], False)
        self._finalize_releases(slots)

    def reset(self):
        for state in self.layers.values():
            state["identity"].fill_(-1)
            state["slot_allocated"].zero_()
            state["payload_valid"].zero_()
            state["reusable"].zero_()
            state["in_flight"].zero_()
            # Keep execution identities monotonic even across shutdown/reset.
            state["execution_serial"].add_(1)
        self._release_state.fill_(self.RELEASED)
        self.lifecycle_error.zero_()
        self.blocked_requests.zero_()

    def validate_lifecycle(self):
        """Check asynchronous lifecycle errors outside graph capture/debug only."""
        npu = getattr(torch, "npu", None)
        is_capturing = getattr(npu, "is_current_stream_capturing", lambda: False)
        if self.device.type == "npu" and is_capturing():
            raise RuntimeError("lifecycle validation must run outside graph capture")
        if self.lifecycle_error.item():
            raise RuntimeError(
                "external_plan request generation changed while FSA work was in flight"
            )
