#pragma once

#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"

namespace SglLim {

__simt_callee__ inline uint32_t PrefixCounts(__ubuf__ uint32_t *scratch, uint32_t count)
{
    uint32_t lane = threadIdx.x;
    scratch[lane] = count;
    asc_syncthreads();
    for (uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        uint32_t previous = lane >= offset ? scratch[lane - offset] : 0;
        asc_syncthreads();
        scratch[lane] += previous;
        asc_syncthreads();
    }
    return lane == 0 ? 0 : scratch[lane - 1];
}

__simt_callee__ inline uint32_t LowerBoundUnion(
    __ubuf__ const int32_t *tokens, uint32_t begin, uint32_t end, int32_t token)
{
    while (begin < end) {
        uint32_t middle = begin + (end - begin) / 2;
        if (tokens[middle] < token) begin = middle + 1;
        else end = middle;
    }
    return begin;
}

__simt_vf__ __launch_bounds__(256) inline void CompactRowPlan(
    __ubuf__ const int32_t *tokens, __ubuf__ int16_t *plans,
    __ubuf__ int16_t *misses, __ubuf__ int16_t *hits,
    __ubuf__ uint32_t *scratch, int32_t length, uint32_t tail, bool ready, __ubuf__ int32_t *slots)
{
    uint32_t lane = threadIdx.x;
    uint32_t begin = lane * 9;
    uint32_t end = begin + 9 < 2051 ? begin + 9 : 2051;
    uint32_t count = 0;
    uint32_t unmapped = 0;
    uint32_t mutableMask = 0;
    for (uint32_t column = begin; column < end; ++column) {
        count += plans[column] < 0;
        unmapped |= tokens[column] >= 0 && plans[column] == 0;
        int32_t offset = tokens[column] - length + static_cast<int32_t>(tail);
        if (tokens[column] >= 0 && offset >= 0 && offset < 32) mutableMask |= 1U << offset;
    }
    scratch[256 + lane] = unmapped;
    scratch[512 + lane] = mutableMask;
    uint32_t prefix = PrefixCounts(scratch, count);
    for (uint32_t offset = 128; offset > 0; offset >>= 1) {
        if (lane < offset) {
            scratch[256 + lane] |= scratch[256 + lane + offset];
            scratch[512 + lane] |= scratch[512 + lane + offset];
        }
        asc_syncthreads();
    }
    for (uint32_t column = begin; column < end; ++column) {
        if (plans[column] < 0) misses[prefix++] = column;
        else if (column < 2048) hits[column - prefix] = column;
        if (slots != nullptr) {
            int32_t plan = plans[column];
            slots[column] = plan == 0 ? -1 : (plan > 0 ? plan - 1 : -plan - 1);
        }
    }
    if (lane == 0) {
        uint32_t total = scratch[255];
        uint32_t tailMisses = (plans[2048] < 0) + (plans[2049] < 0) + (plans[2050] < 0);
        plans[2071] = ready && scratch[256] == 0 ? static_cast<int16_t>(total) : -1;
        plans[2072] = static_cast<int16_t>(scratch[512] & 0xffffU);
        plans[2073] = static_cast<int16_t>(scratch[512] >> 16);
        plans[2074] = total - tailMisses;
    }
}

__simt_vf__ __launch_bounds__(256) inline void ResetResident(
    __gm__ int32_t *mapping, __gm__ int32_t *sources, __gm__ int64_t *physical,
    uint64_t mapBase, uint64_t residentBase, uint32_t sourceCount, uint32_t capacity)
{
    for (uint32_t slot = threadIdx.x; slot < capacity; slot += blockDim.x) {
        int32_t previous = sources[residentBase + slot];
        if (previous >= 0 && previous < sourceCount) mapping[mapBase + previous] = -1;
        sources[residentBase + slot] = -1;
        physical[residentBase + slot] = -1;
    }
    asc_threadfence();
}

__simt_vf__ __launch_bounds__(256) inline void PrepareUnionSort(
    __ubuf__ const int32_t *tokens, __ubuf__ float *keys, __ubuf__ uint32_t *indices,
    uint32_t base, uint32_t valid, uint32_t padded)
{
    for (uint32_t index = threadIdx.x; index < padded; index += blockDim.x) {
        int32_t token = index < valid ? tokens[base + index] : -1;
        keys[index] = token >= 0 ? -static_cast<float>(token) : -1.0e20F;
        indices[index] = static_cast<uint32_t>(token);
    }
}

__simt_vf__ __launch_bounds__(256) inline void UnpackUnionSort(
    __ubuf__ const int32_t *pairs, __ubuf__ int32_t *tokens, uint32_t base, uint32_t valid)
{
    for (uint32_t index = threadIdx.x; index < valid; index += blockDim.x)
        tokens[base + index] = pairs[index * 2 + 1];
}

__simt_vf__ __launch_bounds__(256) inline void BuildUnionPrefixes(
    __ubuf__ const int32_t *tokens, __ubuf__ uint16_t *prefixes,
    __ubuf__ uint32_t *scratch, uint32_t count)
{
    uint32_t lane = threadIdx.x;
    uint32_t begin = lane * 64;
    uint32_t end = begin + 64 < count ? begin + 64 : count;
    uint32_t chunk = begin / 1024;
    uint32_t unique = 0;
    for (uint32_t index = begin; index < end; ++index) {
        int32_t token = tokens[index];
        bool first = index == chunk * 1024 || tokens[index - 1] != token;
        for (uint32_t previous = 0; first && previous < chunk; ++previous) {
            uint32_t limit = (previous + 1) * 1024;
            uint32_t found = LowerBoundUnion(tokens, previous * 1024, limit, token);
            if (found < limit && tokens[found] == token) first = false;
        }
        unique += first;
        prefixes[index] = unique;
    }
    scratch[lane] = unique;
    asc_syncthreads();
    for (uint32_t offset = 1; offset < 16; offset <<= 1) {
        uint32_t previous = lane % 16 >= offset ? scratch[lane - offset] : 0;
        asc_syncthreads();
        scratch[lane] += previous;
        asc_syncthreads();
    }
    uint32_t prefix = lane % 16 == 0 ? 0 : scratch[lane - 1];
    for (uint32_t index = begin; index < end; ++index) prefixes[index] += prefix;
}

__simt_vf__ __launch_bounds__(256) inline void MergeUniqueUnion(
    __ubuf__ const int32_t *tokens, __ubuf__ const uint16_t *prefixes,
    __ubuf__ int32_t *output, uint32_t count, uint32_t capacity)
{
    uint32_t chunks = (count + 1023) / 1024;
    for (uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        uint32_t previous = index % 1024 == 0 ? 0 : prefixes[index - 1];
        if (prefixes[index] == previous) continue;
        int32_t token = tokens[index];
        uint32_t rank = 0;
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            uint32_t begin = chunk * 1024;
            uint32_t end = begin + 1024 < count ? begin + 1024 : count;
            uint32_t found = LowerBoundUnion(tokens, begin, end, token);
            if (found > begin) rank += prefixes[found - 1];
        }
        if (rank < capacity) output[rank] = token;
    }
}

__simt_vf__ __launch_bounds__(256) inline void CopyUnion(
    __ubuf__ const int32_t *input, __ubuf__ int32_t *output, uint32_t count)
{
    for (uint32_t index = threadIdx.x; index < count; index += blockDim.x) output[index] = input[index];
}

__simt_vf__ __launch_bounds__(256) inline void RankNewSlots(
    __ubuf__ const int32_t *tokens, __ubuf__ uint16_t *ranks,
    __ubuf__ uint32_t *scratch, __gm__ const int32_t *mapping,
    uint64_t mapBase, uint32_t count, uint32_t capacity)
{
    uint32_t span = (count + blockDim.x - 1) / blockDim.x;
    uint32_t begin = threadIdx.x * span;
    uint32_t end = begin + span < count ? begin + span : count;
    uint32_t missing = 0;
    for (uint32_t index = begin; index < end; ++index) {
        int32_t slot = mapping[mapBase + tokens[index]];
        bool absent = slot < 0 || slot >= capacity;
        missing += absent;
        ranks[index] = absent ? missing : 0;
    }
    uint32_t prefix = PrefixCounts(scratch, missing);
    for (uint32_t index = begin; index < end; ++index)
        if (ranks[index] != 0) ranks[index] += prefix;
}

__simt_vf__ __launch_bounds__(256) inline void SelectVictims(
    __ubuf__ const int32_t *protectedSlots, __ubuf__ int32_t *victims,
    __ubuf__ uint32_t *scratch, __gm__ const int32_t *sources,
    uint64_t residentBase, uint32_t capacity, uint32_t missing, uint32_t firstSlot)
{
    uint32_t span = (capacity + blockDim.x - 1) / blockDim.x;
    uint32_t begin = threadIdx.x * span;
    uint32_t end = begin + span < capacity ? begin + span : capacity;
    uint32_t empty = 0;
    for (uint32_t slot = begin; slot < end; ++slot)
        empty += protectedSlots[slot] != 0 && sources[residentBase + slot] < 0;
    uint32_t prefix = PrefixCounts(scratch, empty);
    uint32_t totalEmpty = scratch[255];
    for (uint32_t slot = begin; slot < end; ++slot) {
        if (protectedSlots[slot] == 0 || sources[residentBase + slot] >= 0) continue;
        if (prefix < missing) victims[prefix] = slot;
        ++prefix;
    }
    asc_syncthreads();
    if (totalEmpty >= missing) return;
    uint32_t eligible = 0;
    for (uint32_t index = begin; index < end; ++index) {
        uint32_t slot = (firstSlot + index) % capacity;
        eligible += protectedSlots[slot] != 0 && sources[residentBase + slot] >= 0;
    }
    prefix = PrefixCounts(scratch, eligible) + totalEmpty;
    for (uint32_t index = begin; index < end; ++index) {
        uint32_t slot = (firstSlot + index) % capacity;
        if (protectedSlots[slot] == 0 || sources[residentBase + slot] < 0) continue;
        if (prefix < missing) victims[prefix] = slot;
        ++prefix;
    }
    asc_syncthreads();
    if (threadIdx.x == 0) scratch[255] += totalEmpty;
}

__simt_vf__ __launch_bounds__(256) inline void ApplyResidentUpdates(
    __ubuf__ const int32_t *tokens, __ubuf__ const uint16_t *ranks,
    __ubuf__ const int32_t *victims, __ubuf__ int32_t *slots,
    __gm__ int32_t *mapping, __gm__ int32_t *sources, __gm__ int64_t *physical,
    uint64_t mapBase, uint64_t residentBase, uint32_t count, uint32_t sourceCount)
{
    for (uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        int32_t token = tokens[index];
        uint32_t rank = ranks[index];
        int32_t slot = rank == 0 ? mapping[mapBase + token] : victims[rank - 1];
        slots[index] = slot;
        if (rank == 0) continue;
        int32_t previous = sources[residentBase + slot];
        if (previous >= 0 && previous < sourceCount) mapping[mapBase + previous] = -1;
        mapping[mapBase + token] = slot;
        sources[residentBase + slot] = token;
        physical[residentBase + slot] = -1;
    }
    asc_threadfence();
}

__simt_callee__ inline uint64_t PreparedEntry(
    __gm__ const int16_t *membership, uint32_t start, uint32_t end, uint32_t index)
{
    uint32_t row = start;
    for (; row < end; ++row) {
        uint32_t count = membership[static_cast<uint64_t>(row) * 16400 + 16391];
        if (index < count) break;
        index -= count;
    }
    uint32_t column = membership[static_cast<uint64_t>(row) * 16400 + index];
    return static_cast<uint64_t>(row) * 2051 + column;
}

__simt_vf__ __launch_bounds__(256) inline void CollectPreparedUnion(
    __gm__ const int16_t *membership, __gm__ const int32_t *ids,
    __gm__ const int32_t *slots, __ubuf__ uint32_t *owners,
    __ubuf__ int32_t *output, __ubuf__ uint32_t *scratch,
    uint32_t start, uint32_t end, uint32_t total, uint32_t capacity)
{
    for (uint32_t slot = threadIdx.x; slot < capacity; slot += blockDim.x) owners[slot] = total;
    asc_syncthreads();
    uint32_t span = (total + blockDim.x - 1) / blockDim.x;
    uint32_t begin = threadIdx.x * span;
    uint32_t limit = begin + span < total ? begin + span : total;
    for (uint32_t index = begin; index < limit; ++index) {
        uint64_t entry = PreparedEntry(membership, start, end, index);
        asc_atomic_min(owners + slots[entry], index);
    }
    asc_syncthreads();
    uint32_t count = 0;
    for (uint32_t index = begin; index < limit; ++index) {
        uint64_t entry = PreparedEntry(membership, start, end, index);
        count += owners[slots[entry]] == index;
    }
    uint32_t prefix = PrefixCounts(scratch, count);
    for (uint32_t index = begin; index < limit; ++index) {
        uint64_t entry = PreparedEntry(membership, start, end, index);
        if (owners[slots[entry]] == index) output[prefix++] = ids[entry];
    }
}

__simt_vf__ __launch_bounds__(256) inline void PublishMissUnion(
    __ubuf__ const int32_t *tokens, __gm__ const int32_t *mapping,
    __gm__ const int32_t *table, __gm__ int32_t *outputIds,
    __gm__ int32_t *outputSlots, __gm__ int64_t *outputPhysical,
    uint64_t mapBase, uint64_t tableBase, uint64_t outputBase, uint32_t count)
{
    for (uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        int32_t token = tokens[index];
        int32_t page = table[tableBase + token / 128];
        outputIds[outputBase + index] = token;
        outputSlots[outputBase + index] = mapping[mapBase + token];
        outputPhysical[outputBase + index] = static_cast<int64_t>(page) * 128 + token % 128;
    }
    asc_threadfence();
}

}
