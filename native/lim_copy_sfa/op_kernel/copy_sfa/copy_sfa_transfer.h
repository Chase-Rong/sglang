#pragma once

#include "kernel_operator.h"

namespace SglCopy {
using namespace AscendC;

__aicore__ inline uint32_t BatchWork(GM_ADDR requests, GM_ADDR state,
    uint32_t batch, uint32_t requestCapacity)
{
    GlobalTensor<int64_t> requestsGm, stateGm;
    requestsGm.SetGlobalBuffer((__gm__ int64_t *)requests);
    stateGm.SetGlobalBuffer((__gm__ int64_t *)state);
    uint32_t work = 0;
    for (uint32_t request = 0; request < batch; ++request) {
        int64_t owner = requestsGm.GetValue(request);
        if (owner <= 0 || owner >= requestCapacity) continue;
        if (stateGm.GetValue(owner * 16 + 2) != 0) work |= 1;
        if (stateGm.GetValue(owner * 16 + 3) > 0) work |= 2;
    }
    return work;
}

__aicore__ inline void Fill(GM_ADDR host, GM_ADDR resident, GM_ADDR requests,
    GM_ADDR ends, GM_ADDR unionSlots, GM_ADDR unionPhysical, GM_ADDR state,
    GM_ADDR membership, uint32_t batch, uint32_t capacity, uint32_t unionWidth,
    uint32_t requestCapacity, TPipe &pipe)
{
    if ASCEND_IS_AIV {
        GlobalTensor<bfloat16_t> hostGm, residentGm;
        GlobalTensor<int64_t> requestsGm, keysGm, stateGm;
        GlobalTensor<int32_t> endsGm, slotsGm;
        GlobalTensor<int16_t> planGm;
        hostGm.SetGlobalBuffer((__gm__ bfloat16_t *)host);
        residentGm.SetGlobalBuffer((__gm__ bfloat16_t *)resident);
        requestsGm.SetGlobalBuffer((__gm__ int64_t *)requests);
        keysGm.SetGlobalBuffer((__gm__ int64_t *)unionPhysical);
        stateGm.SetGlobalBuffer((__gm__ int64_t *)state);
        endsGm.SetGlobalBuffer((__gm__ int32_t *)ends);
        slotsGm.SetGlobalBuffer((__gm__ int32_t *)unionSlots);
        planGm.SetGlobalBuffer((__gm__ int16_t *)membership);
        TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> queue;
        pipe.InitBuffer(queue, 2, 1024);
        uint64_t pendingDestination = 0;
        bool pending = false;
        uint32_t coreCount = GetBlockNum() * 2;
        uint32_t flatBase = 0;
        for (uint32_t request = 0; request < batch; ++request) {
            int64_t owner = requestsGm.GetValue(request);
            if (owner <= 0 || owner >= requestCapacity || stateGm.GetValue(owner * 16 + 2) == 0) continue;
            int64_t count = stateGm.GetValue(owner * 16 + 3);
            if (count < 0 || count > unionWidth) continue;
            uint32_t first = (GetBlockIdx() + coreCount - flatBase % coreCount) % coreCount;
            for (uint32_t index = first; index < count; index += coreCount) {
                uint64_t entry = static_cast<uint64_t>(request) * unionWidth + index;
                int32_t slot = slotsGm.GetValue(entry);
                int64_t key = keysGm.GetValue(entry);
                if (slot < 0 || slot >= capacity || key < 0) continue;
                auto incoming = queue.AllocTensor<bfloat16_t>();
                DataCopy(incoming, hostGm[key * 512], 512);
                queue.EnQue(incoming);
                if (pending) {
                    auto outgoing = queue.DeQue<bfloat16_t>();
                    DataCopy(residentGm[pendingDestination], outgoing, 512);
                    queue.FreeTensor(outgoing);
                }
                pendingDestination = (owner * capacity + slot) * 512;
                pending = true;
            }
            flatBase += count;
            int32_t start = request == 0 ? 0 : endsGm.GetValue(request - 1);
            int32_t end = endsGm.GetValue(request);
            for (int32_t row = start + GetBlockIdx(); row < end; row += coreCount) {
                for (uint32_t column = 0; column < 2051; ++column) {
                    uint64_t offset = static_cast<uint64_t>(row) * 16400 + 14333 + column;
                    int16_t value = planGm.GetValue(offset);
                    if (value < 0) planGm.SetValue(offset, -value);
                }
            }
        }
        if (pending) {
            auto outgoing = queue.DeQue<bfloat16_t>();
            DataCopy(residentGm[pendingDestination], outgoing, 512);
            queue.FreeTensor(outgoing);
        }
        DataCacheCleanAndInvalid<int16_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(planGm);
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void Commit(GM_ADDR ends, GM_ADDR lengths, GM_ADDR requests,
    GM_ADDR unionSources, GM_ADDR unionSlots, GM_ADDR unionPhysical, GM_ADDR physical,
    GM_ADDR state, GM_ADDR membership, uint32_t batch, uint32_t capacity,
    uint32_t unionWidth, uint32_t mutableTail, uint32_t requestCapacity, TPipe &pipe)
{
    if ASCEND_IS_AIV {
        GlobalTensor<int32_t> endsGm, lengthsGm, sourcesGm, slotsGm;
        GlobalTensor<int64_t> requestsGm, keysGm, physicalGm, stateGm;
        GlobalTensor<uint16_t> planGm;
        endsGm.SetGlobalBuffer((__gm__ int32_t *)ends);
        lengthsGm.SetGlobalBuffer((__gm__ int32_t *)lengths);
        sourcesGm.SetGlobalBuffer((__gm__ int32_t *)unionSources);
        slotsGm.SetGlobalBuffer((__gm__ int32_t *)unionSlots);
        requestsGm.SetGlobalBuffer((__gm__ int64_t *)requests);
        keysGm.SetGlobalBuffer((__gm__ int64_t *)unionPhysical);
        physicalGm.SetGlobalBuffer((__gm__ int64_t *)physical);
        stateGm.SetGlobalBuffer((__gm__ int64_t *)state);
        planGm.SetGlobalBuffer((__gm__ uint16_t *)membership);
        uint32_t coreCount = GetBlockNum() * 2;
        uint32_t flatBase = 0;
        TBuf<TPosition::VECCALC> publicationBuffer;
        pipe.InitBuffer(publicationBuffer, 32 * 32);
        auto publication = publicationBuffer.Get<int64_t>();
        uint64_t destinations[32];
        for (uint32_t request = 0; request < batch; ++request) {
            int64_t owner = requestsGm.GetValue(request);
            if (owner <= 0 || owner >= requestCapacity) continue;
            int64_t count = stateGm.GetValue(owner * 16 + 3);
            if (count < 0 || count > unionWidth) continue;
            uint32_t first = (GetBlockIdx() + coreCount - flatBase % coreCount) % coreCount;
            flatBase += count;
            if (first >= count) continue;
            int32_t start = request == 0 ? 0 : endsGm.GetValue(request - 1);
            int32_t end = endsGm.GetValue(request);
            uint32_t presence[6];
            int32_t visible[6];
            for (int32_t row = start; row < end; ++row) {
                uint64_t base = static_cast<uint64_t>(row) * 16400 + 16392;
                presence[row - start] = planGm.GetValue(base) | (static_cast<uint32_t>(planGm.GetValue(base + 1)) << 16);
                visible[row - start] = lengthsGm.GetValue(row);
            }
            uint32_t index = first;
            while (index < count) {
                uint32_t published = 0;
                for (; index < count && published < 32; index += coreCount) {
                    uint64_t entry = static_cast<uint64_t>(request) * unionWidth + index;
                    int32_t slot = slotsGm.GetValue(entry);
                    if (slot < 0 || slot >= capacity) continue;
                    int32_t source = sourcesGm.GetValue(entry);
                    bool reusable = true;
                    for (int32_t row = 0; row < end - start && reusable; ++row) {
                        int32_t tail = source - visible[row] + mutableTail;
                        if (tail >= 0 && tail < mutableTail && (presence[row] & (1U << tail))) reusable = false;
                    }
                    destinations[published] = owner * capacity + slot;
                    publication.SetValue(published * 4, reusable ? keysGm.GetValue(entry) : -1);
                    ++published;
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                for (uint32_t entry = 0; entry < published; ++entry)
                    DataCopyPad(physicalGm[destinations[entry]], publication[entry * 4],
                        DataCopyExtParams{1, 8, 0, 0, 0});
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            }
        }
    }
}
}
