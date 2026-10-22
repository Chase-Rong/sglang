"""CPU oracle for pooled selection and score-driven request-level residency.

This oracle is not a serving fallback. It keeps model selection independent
from physical page placement and from the replacement policy.
"""

from dataclasses import dataclass

import torch


def pooled_selection(query, weights, keys, visible_lengths, *, topk=2048, pool_size=4):
    if (
        query.device.type != "cpu"
        or weights.device.type != "cpu"
        or keys.device.type != "cpu"
    ):
        raise ValueError("The LIM reference runs on CPU only")
    if query.ndim != 3 or keys.ndim != 2 or weights.shape != query.shape[:2]:
        raise ValueError(
            "Expected query[rows,heads,dim], weights[rows,heads], keys[pools,dim]"
        )
    if query.shape[-1] != keys.shape[-1] or topk % pool_size:
        raise ValueError("Incompatible pooled index geometry")
    if len(visible_lengths) != query.shape[0] or any(
        length < 0 for length in visible_lengths
    ):
        raise ValueError("Invalid per-query causal lengths")
    scores = torch.einsum("rhd,pd->rhp", query.float(), keys.float()).relu()
    scores = (scores * weights.float().unsqueeze(-1)).sum(1)
    selected = torch.full((query.shape[0], topk + pool_size - 1), -1, dtype=torch.int32)
    for row, length in enumerate(visible_lengths):
        available = length // pool_size
        if available > keys.shape[0]:
            raise ValueError("Pooled keys do not cover a query's visible prefix")
        if length <= topk:
            selected[row, :length] = torch.arange(length, dtype=torch.int32)
        else:
            indices = torch.argsort(
                scores[row, :available], descending=True, stable=True
            )[: topk // pool_size]
            tokens = (indices[:, None] * pool_size + torch.arange(pool_size)).flatten()
            selected[row, : tokens.numel()] = tokens.to(torch.int32)
            tail = torch.arange(available * pool_size, length, dtype=torch.int32)
            selected[row, topk : topk + tail.numel()] = tail
        scores[row, available:] = -torch.inf
    return selected, scores


@dataclass
class ReferencePlan:
    logical_ids: torch.Tensor
    physical_ids: torch.Tensor
    slots: torch.Tensor
    misses: torch.Tensor
    stable: torch.Tensor
    copy_sources: torch.Tensor
    copy_slots: torch.Tensor
    generation: int
    sequence_length: int
    active: bool = True


class ReferenceResident:
    def __init__(self, capacity, source_capacity, replacement_policy="score"):
        if capacity <= 0 or source_capacity <= 0:
            raise ValueError("Positive resident/source capacities are required")
        self.capacity = capacity
        self.source_capacity = source_capacity
        if replacement_policy not in ("score", "protected_hash"):
            raise ValueError("Unknown replacement policy")
        self.replacement_policy = replacement_policy
        self.sources = torch.full((capacity,), -1, dtype=torch.int64)
        self.physical = torch.full((capacity,), -1, dtype=torch.int64)
        self.generation = -1
        self.sequence_length = -1
        self.pending = None

    def plan(
        self,
        ids,
        table,
        lengths,
        generation,
        *,
        page_size=128,
        mutable_tail=12,
        scores=None,
        active=True,
    ):
        if self.pending is not None:
            raise RuntimeError(
                "Previous plan must complete before reusing resident slots"
            )
        if (
            ids.ndim != 2
            or len(lengths) != ids.shape[0]
            or page_size <= 0
            or mutable_tail < 0
        ):
            raise ValueError("Invalid plan geometry")
        if generation < 0 or not lengths or any(length < 0 for length in lengths):
            raise ValueError("Invalid owner generation or causal lengths")
        if max(lengths) > self.source_capacity:
            raise ValueError("Source capacity exceeded")
        if not active:
            invalid = torch.full_like(ids, -1, dtype=torch.int64)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            plan = ReferencePlan(
                invalid,
                invalid.clone(),
                invalid.clone(),
                mask,
                mask.clone(),
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64),
                generation,
                0,
                active=False,
            )
            self.pending = plan
            return plan
        if generation != self.generation or max(lengths) < self.sequence_length:
            self.sources.fill_(-1)
            self.physical.fill_(-1)
        logical = ids.long().clone()
        physical = torch.full_like(logical, -1)
        stable = torch.zeros_like(logical, dtype=torch.bool)
        for row, length in enumerate(lengths):
            valid = (logical[row] >= 0) & (logical[row] < length)
            for column in valid.nonzero().flatten().tolist():
                source = int(logical[row, column])
                page = source // page_size
                if page < len(table) and int(table[page]) >= 0:
                    physical[row, column] = (
                        int(table[page]) * page_size + source % page_size
                    )
                    stable[row, column] = source < max(0, length - mutable_tail)
            logical[row, physical[row] < 0] = -1
        union = sorted(set(logical[logical >= 0].tolist()))
        if len(union) > self.capacity:
            raise ValueError("Selected query union exceeds the resident capacity")
        resident = {
            int(source): slot for slot, source in enumerate(self.sources) if source >= 0
        }
        missing = [source for source in union if source not in resident]
        protected = set(union)

        def priority(slot):
            source = int(self.sources[slot])
            if source < 0:
                return (-torch.inf, slot)
            if scores is None:
                return (0.0, slot)
            pool = source // 4
            if pool >= scores.shape[1]:
                return (-torch.inf, slot)
            values = scores[:, pool]
            finite = values[torch.isfinite(values)]
            return (float(finite.sum()) if finite.numel() else -torch.inf, slot)

        victims = sorted(
            (
                slot
                for slot, source in enumerate(self.sources)
                if int(source) not in protected
            ),
            key=priority,
        )
        if self.replacement_policy == "protected_hash":
            available = set(victims)
            empty = [
                slot
                for slot in range(self.capacity)
                if slot in available and self.sources[slot] < 0
            ]
            available.difference_update(empty)
            seed = max(lengths)
            seed = ((seed ^ (seed >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
            seed = ((seed ^ (seed >> 15)) * 0x846CA68B) & 0xFFFFFFFF
            seed ^= seed >> 16
            chunks = (self.capacity + 2047) // 2048
            first = seed % chunks
            ordered = []
            for visit in range(chunks):
                begin = ((first + visit) % chunks) * 2048
                ordered.extend(
                    slot
                    for slot in range(begin, min(begin + 2048, self.capacity))
                    if slot in available
                )
            victims = empty + ordered
        for source, slot in zip(missing, victims):
            previous = int(self.sources[slot])
            resident.pop(previous, None)
            resident[source] = slot
            self.sources[slot] = source
            self.physical[slot] = -1
        slots = torch.full_like(logical, -1)
        misses = torch.zeros_like(logical, dtype=torch.bool)
        copies = {}
        for row in range(ids.shape[0]):
            for column in range(ids.shape[1]):
                source = int(logical[row, column])
                if source < 0:
                    continue
                slot = resident[source]
                slots[row, column] = slot
                miss = (
                    self.physical[slot] != physical[row, column]
                    or not stable[row, column]
                )
                misses[row, column] = miss
                if miss:
                    copies[slot] = int(physical[row, column])
        plan = ReferencePlan(
            logical,
            physical,
            slots,
            misses,
            stable,
            torch.tensor(list(copies.values()), dtype=torch.int64),
            torch.tensor(list(copies), dtype=torch.int64),
            generation,
            max(lengths),
        )
        self.pending = plan
        return plan

    def commit(self, plan):
        if self.pending is not plan:
            raise RuntimeError("Cannot commit a foreign or already committed plan")
        if not plan.active:
            self.pending = None
            return
        for slot in plan.slots.unique().tolist():
            if slot < 0:
                continue
            selected = plan.slots == slot
            self.physical[slot] = (
                plan.physical_ids[selected][0] if plan.stable[selected].all() else -1
            )
        self.generation = plan.generation
        self.sequence_length = plan.sequence_length
        self.pending = None

    def abort(self):
        self.sources.fill_(-1)
        self.physical.fill_(-1)
        self.generation = -1
        self.sequence_length = -1
        self.pending = None


def copy_attention(query, host_kv, resident_kv, plan, scale):
    resident_kv[plan.copy_slots] = host_kv[plan.copy_sources]
    output = torch.zeros_like(query)
    for row in range(query.shape[0]):
        selected = plan.slots[row] >= 0
        if selected.any():
            keys = resident_kv[plan.slots[row, selected]].float()
            probabilities = torch.softmax(query[row].float() @ keys.T * scale, dim=-1)
            output[row] = (probabilities @ keys).to(output.dtype)
    return output
