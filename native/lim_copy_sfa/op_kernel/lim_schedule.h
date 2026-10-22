#pragma once

#include <cstdint>

#ifdef __CCE_AICORE__
#define SGL_LIM_INLINE __aicore__ inline
#else
#define SGL_LIM_INLINE constexpr
#endif

namespace SglLimSchedule {
SGL_LIM_INLINE uint32_t HashVictimScanSeed(uint32_t length)
{
    length ^= length >> 16U;
    length *= 0x7feb352dU;
    length ^= length >> 15U;
    length *= 0x846ca68bU;
    return length ^ (length >> 16U);
}

SGL_LIM_INLINE bool NeedsUnionCopy(int64_t count)
{
    return count > 512;
}

SGL_LIM_INLINE uint32_t GatherIndex(uint32_t batch, uint32_t index)
{
    if (index >= 2048) return index;
    return ((3 - batch % 3 + 4 - index / 512) % 4) * 512 + index % 512;
}
}

#undef SGL_LIM_INLINE
