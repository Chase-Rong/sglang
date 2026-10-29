"""Scalar CPU oracle for ABI3 tests only; never imported by production code."""


def hash32(value):
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = value * 0x7FEB352D & 0xFFFFFFFF
    value ^= value >> 15
    value = value * 0x846CA68B & 0xFFFFFFFF
    return value ^ (value >> 16)


def key_hash(value):
    return hash32((value & 0xFFFFFFFF) ^ hash32(value >> 32))


def reference(keys, stable, identity, allocated, payload, reusable, *, length):
    """Valid single-request oracle, union in first physical occurrence order."""
    capacity = len(identity)
    flat = [key for row in keys for key in row]
    stable_flat = [value for row in stable for value in row]
    first = {}
    cacheable = {}
    for occurrence, key in enumerate(flat):
        if key < 0:
            continue
        first.setdefault(key, occurrence)
        cacheable[key] = cacheable.get(key, True) and stable_flat[occurrence]
    union = list(first)
    after_reusable = list(reusable)
    for slot, key in enumerate(identity):
        if key in cacheable and not cacheable[key]:
            after_reusable[slot] = False
    hits = {}
    for slot, key in enumerate(identity):
        if (
            key in first
            and cacheable[key]
            and allocated[slot]
            and payload[slot]
            and after_reusable[slot]
        ):
            hits.setdefault(key, slot)
    protected = set(hits.values())
    empty = [s for s in range(capacity) if s not in protected and not allocated[s]]
    start = hash32(length) % ((capacity + 2047) // 2048) * 2048
    occupied = [
        (start + i) % capacity
        for i in range(capacity)
        if (start + i) % capacity not in protected and allocated[(start + i) % capacity]
    ]
    victims = iter(empty + occupied)
    slots = [hits[key] if key in hits else next(victims) for key in union]
    lookup = {key: u for u, key in enumerate(union)}
    return dict(
        keys=union,
        slots=slots,
        owner=[first[key] for key in union],
        miss=[key not in hits for key in union],
        cacheable=[cacheable[key] for key in union],
        row_union=[lookup.get(key, -1) for key in flat],
        reusable=after_reusable,
    )
