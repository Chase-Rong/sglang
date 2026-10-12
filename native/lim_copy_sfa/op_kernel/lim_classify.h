#pragma once

#include "simt_api/asc_simt.h"
#include "lim_postprocess.h"

namespace SglLim {
__simt_vf__ __launch_bounds__(256) inline void ClassifyPooledSelection(
    __ubuf__ const uint32_t *pools, __ubuf__ int32_t *tokens,
    __ubuf__ int32_t *slots, __ubuf__ int16_t *plans,
    __gm__ const int32_t *table, __gm__ const int32_t *mapping,
    __gm__ const int64_t *physical, uint64_t tableBase, uint64_t mapBase,
    uint64_t residentBase, uint64_t hostSize, uint32_t sourceCount,
    uint32_t capacity, uint32_t columns, int32_t length, uint32_t tail, bool active,
    __ubuf__ const int32_t *prefetchedSlots)
{
    for (uint32_t column = threadIdx.x; column < 2051; column += blockDim.x) {
        int32_t token = -1;
        if (active) {
            if (length <= 2048) token = column;
            else if (column >= 2048) token = length / 4 * 4 + column - 2048;
            else {
                int32_t pool = static_cast<int32_t>(pools[column / 4]);
                if (pool >= 0 && pool < length / 4) token = pool * 4 + column % 4;
            }
        }
        int64_t address = -1;
        if (token >= 0 && token < length && token < sourceCount && token / 128 < columns) {
            int32_t page = table[tableBase + token / 128];
            if (page >= 0) address = static_cast<int64_t>(page) * 128 + token % 128;
            if (address >= hostSize) address = -1;
        }
        int32_t slot = address >= 0 ?
            (prefetchedSlots != nullptr && column < 2048 && length > 2048 ?
                prefetchedSlots[column] : mapping[mapBase + token]) : -1;
        bool valid = slot >= 0 && slot < capacity;
        bool hit = valid && physical[residentBase + slot] == address && token < length - static_cast<int32_t>(tail);
        tokens[column] = address >= 0 ? token : -1;
        slots[column] = valid ? slot : -1;
        plans[column] = valid ? (hit ? slot + 1 : -slot - 1) : 0;
    }
}

__simt_vf__ __launch_bounds__(256) inline void ClassifyManagedRows(
    __gm__ const int32_t *ids, __gm__ const int32_t *table,
    __gm__ const int32_t *mapping, __gm__ const int64_t *physical, __gm__ const int32_t *lengths,
    __ubuf__ int32_t *tokens, __ubuf__ int32_t *slots, __ubuf__ int16_t *plans,
    __ubuf__ const int32_t *unionTokens, __ubuf__ const int32_t *unionSlots,
    uint64_t tableBase, uint64_t mapBase, uint64_t residentBase, uint64_t hostSize,
    uint32_t sourceCount, uint32_t capacity, uint32_t columns, uint32_t start, uint32_t end,
    uint32_t tail, uint32_t unique, bool pooled, bool active, bool repair)
{
    for (uint32_t index = threadIdx.x; index < (end - start) * 2051; index += blockDim.x) {
        uint32_t route = index / 2051;
        uint32_t row = start + route;
        uint32_t column = index % 2051;
        int32_t length = lengths[row];
        int32_t token = -1;
        if (active) {
            if (!pooled) token = ids[static_cast<uint64_t>(row) * 2051 + column];
            else if (length <= 2048) token = column;
            else if (column >= 2048) token = length / 4 * 4 + column - 2048;
            else {
                int32_t pool = ids[static_cast<uint64_t>(row) * 512 + column / 4];
                if (pool >= 0 && pool < length / 4) token = pool * 4 + column % 4;
            }
        }
        int64_t address = -1;
        if (token >= 0 && token < length && token < sourceCount && token / 128 < columns) {
            int32_t page = table[tableBase + token / 128];
            if (page >= 0) address = static_cast<int64_t>(page) * 128 + token % 128;
            if (address >= hostSize) address = -1;
        }
        int32_t slot = -1;
        if (address >= 0) {
            uint32_t found = LowerBoundUnion(unionTokens, 0, unique, token);
            slot = found < unique && unionTokens[found] == token ? unionSlots[found] : mapping[mapBase + token];
        }
        bool valid = slot >= 0 && slot < capacity;
        bool hit = valid && physical[residentBase + slot] == address && token < length - static_cast<int32_t>(tail);
        uint32_t output = route * 2080 + column;
        tokens[output] = address >= 0 && (!repair || valid) ? token : -1;
        if (slots != nullptr) slots[output] = valid ? slot : -1;
        plans[output] = valid ? (hit ? slot + 1 : -slot - 1) : 0;
    }
}

__simt_vf__ __launch_bounds__(256) inline void CollectRowMisses(
    __ubuf__ const int32_t *tokens, __ubuf__ const int32_t *slots,
    __ubuf__ const int16_t *plans, __ubuf__ int32_t *output,
    __ubuf__ uint32_t *protectedSlots, __ubuf__ uint32_t *scratch, uint32_t outputBase)
{
    uint32_t begin = threadIdx.x * 9;
    uint32_t end = begin + 9 < 2051 ? begin + 9 : 2051;
    uint32_t count = 0;
    for (uint32_t column = begin; column < end; ++column) {
        if (slots[column] >= 0) asc_atomic_min(protectedSlots + slots[column], 0U);
        count += tokens[column] >= 0 && plans[column] <= 0;
    }
    uint32_t prefix = PrefixCounts(scratch, count) + outputBase;
    for (uint32_t column = begin; column < end; ++column)
        if (tokens[column] >= 0 && plans[column] <= 0) output[prefix++] = tokens[column];
}

__simt_vf__ __launch_bounds__(256) inline void CollectPreparedRowMisses(
    __gm__ const int32_t *tokens, __gm__ const int32_t *slots,
    __gm__ const int16_t *plans, __ubuf__ int32_t *output,
    __ubuf__ uint32_t *protectedSlots, __ubuf__ uint32_t *scratch,
    uint32_t row, uint32_t outputBase)
{
    uint32_t begin = threadIdx.x * 9;
    uint32_t end = begin + 9 < 2051 ? begin + 9 : 2051;
    uint64_t tokenBase = static_cast<uint64_t>(row) * 2051;
    uint64_t planBase = static_cast<uint64_t>(row) * 16400 + 14333;
    uint32_t count = 0;
    for (uint32_t column = begin; column < end; ++column) {
        int32_t slot = slots[tokenBase + column];
        if (slot >= 0) asc_atomic_min(protectedSlots + slot, 0U);
        count += tokens[tokenBase + column] >= 0 && plans[planBase + column] <= 0;
    }
    uint32_t prefix = PrefixCounts(scratch, count) + outputBase;
    for (uint32_t column = begin; column < end; ++column)
        if (tokens[tokenBase + column] >= 0 && plans[planBase + column] <= 0)
            output[prefix++] = tokens[tokenBase + column];
}

__simt_vf__ __launch_bounds__(256) inline void RepairPreparedRows(
    __gm__ const int32_t *selected, __gm__ const int16_t *preparedPlans,
    __gm__ const int32_t *mapping, __ubuf__ int32_t *tokens,
    __ubuf__ int16_t *plans, uint64_t mapBase,
    uint32_t start, uint32_t end, uint32_t capacity)
{
    for (uint32_t index = threadIdx.x; index < (end - start) * 2051; index += blockDim.x) {
        uint32_t route = index / 2051;
        uint32_t row = start + route;
        uint32_t column = index % 2051;
        int32_t token = selected[static_cast<uint64_t>(row) * 2051 + column];
        int16_t plan = preparedPlans[static_cast<uint64_t>(row) * 16400 + 14333 + column];
        if (token >= 0 && plan <= 0) {
            int32_t slot = mapping[mapBase + token];
            if (slot >= 0 && slot < capacity) plan = -slot - 1;
            else {
                token = -1;
                plan = 0;
            }
        }
        tokens[route * 2080 + column] = token;
        plans[route * 2080 + column] = plan;
    }
}
}
