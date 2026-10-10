#pragma once

#include "simt_api/asc_simt.h"

namespace SglLim {
__simt_vf__ __launch_bounds__(256) inline void ClassifyPooledSelection(
    __ubuf__ const uint32_t *pools, __ubuf__ int32_t *tokens,
    __ubuf__ int32_t *slots, __ubuf__ int16_t *plans,
    __gm__ const int32_t *table, __gm__ const int32_t *mapping,
    __gm__ const int64_t *physical, uint64_t tableBase, uint64_t mapBase,
    uint64_t residentBase, uint64_t hostSize, uint32_t sourceCount,
    uint32_t capacity, uint32_t columns, int32_t length, uint32_t tail, bool active)
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
        int32_t slot = address >= 0 ? mapping[mapBase + token] : -1;
        bool valid = slot >= 0 && slot < capacity;
        bool hit = valid && physical[residentBase + slot] == address && token < length - static_cast<int32_t>(tail);
        tokens[column] = address >= 0 ? token : -1;
        slots[column] = valid ? slot : -1;
        plans[column] = valid ? (hit ? slot + 1 : -slot - 1) : 0;
    }
}
}
