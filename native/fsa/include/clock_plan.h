#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace sglang::npu_kernel {

inline size_t ClockBucketCount(int64_t capacity) noexcept
{
    size_t count = 1;
    while (count < static_cast<size_t>(capacity) * 2) count <<= 1;
    return count;
}

inline size_t ClockWorkspaceSize(int64_t capacity) noexcept
{
    return ClockBucketCount(capacity) + 3 * static_cast<size_t>(capacity) + 1;
}

inline size_t ClockBucket(int64_t key, size_t mask) noexcept
{
    // Mix all physical-key bits; page-aligned keys must not share one bucket.
    uint64_t value = static_cast<uint64_t>(key);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return static_cast<size_t>(value ^ (value >> 31)) & mask;
}

// Workspace belongs to the resident table, shared by eager and every graph.
// Calls on the same table must be stream ordered, as for the legacy planner.
template<typename Index>
inline void PlanClockRow(const int64_t* keys, int64_t width,
                         int64_t owner, int64_t length, int64_t* resident,
                         int64_t* ages, int64_t* state, int64_t* slots,
                         int64_t capacity, Index* workspace,
                         uint8_t* protected_slots) noexcept
{
    static_assert(std::is_same_v<Index, int16_t> || std::is_same_v<Index, int64_t>,
                  "CLOCK workspace indices must be signed int16 or int64");
    const size_t buckets = ClockBucketCount(capacity);
    auto* heads = workspace;
    auto* next = heads + buckets;
    auto* prev = next + capacity;
    auto* recent = prev + capacity;
    auto& hand = recent[capacity];
    if (state[0] != owner || length < state[1] ||
        state[2] >= std::numeric_limits<int64_t>::max() / 2) {
        std::fill(resident, resident + capacity, -1);
        std::fill(ages, ages + capacity, 0);
        std::fill(heads, recent, -1);
        std::fill(recent, recent + capacity, 0);
        hand = 0;
        state[2] = 0;
    }
    state[0] = owner;
    state[1] = length;
    if (owner < 0 || length <= 0) {
        for (int64_t i = 0; i < width; ++i) slots[i] = i;
        return;
    }
    const int64_t epoch = ++state[2];
    std::fill(protected_slots, protected_slots + capacity, 0);
    std::fill(slots, slots + width, -1);
    // Protect every hit before selecting victims, including hits appearing
    // after misses. Duplicate keys use distinct slots to avoid device races.
    for (int64_t i = 0; i < width; ++i) {
        if (keys[i] < 0) continue;
        for (auto slot = heads[ClockBucket(keys[i], buckets - 1)];
             slot >= 0; slot = next[slot]) {
            if (resident[slot] == keys[i] && !protected_slots[slot]) {
                slots[i] = slot;
                protected_slots[slot] = 1;
                recent[slot] = 1;
                ages[slot] = epoch;
                break;
            }
        }
    }
    for (int64_t i = 0; i < width; ++i) {
        if (slots[i] >= 0 || keys[i] < 0) continue;
        int64_t slot;
        for (;;) {
            slot = hand;
            hand = (hand + 1 == capacity) ? 0 : hand + 1;
            if (protected_slots[slot]) continue;
            if (recent[slot]) { recent[slot] = 0; continue; }
            break;
        }
        if (resident[slot] >= 0) {
            if (prev[slot] >= 0) next[prev[slot]] = next[slot];
            else heads[ClockBucket(resident[slot], buckets - 1)] = next[slot];
            if (next[slot] >= 0) prev[next[slot]] = prev[slot];
        }
        const auto bucket = ClockBucket(keys[i], buckets - 1);
        next[slot] = heads[bucket];
        prev[slot] = -1;
        if (next[slot] >= 0) prev[next[slot]] = slot;
        heads[bucket] = slot;
        resident[slot] = keys[i];
        ages[slot] = epoch;
        recent[slot] = 1;
        protected_slots[slot] = 1;
        slots[i] = slot;
    }
    // Invalid positions still need distinct legal placeholders. They must not
    // invalidate resident entries: the device will not write KV for them.
    int64_t available = 0;
    for (int64_t i = 0; i < width; ++i) {
        if (slots[i] >= 0) continue;
        while (protected_slots[available]) ++available;
        slots[i] = available;
        protected_slots[available++] = 1;
    }
}

} // namespace sglang::npu_kernel
