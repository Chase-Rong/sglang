#pragma once

#include "kernel_operator.h"
#include "lim_classify.h"

namespace SglLim {
using namespace AscendC;

struct EmptyRowPlan {
    __aicore__ inline void BindScratch(LocalTensor<int32_t>) {}
    __aicore__ inline void operator()(uint32_t, LocalTensor<uint32_t>) {}
};

class ResidentRowPlan {
public:
    __aicore__ inline void Init(GM_ADDR ends, GM_ADDR lengths, GM_ADDR table,
        GM_ADDR requests, GM_ADDR generations, GM_ADDR mapping, GM_ADDR physical,
        GM_ADDR state, GM_ADDR outputIds, GM_ADDR outputSlots, GM_ADDR membership,
        uint32_t batch, uint32_t requestCapacity, uint32_t sourceCapacity,
        uint32_t capacity, uint32_t columns, uint64_t hostTokens, uint32_t mutableTail)
    {
        endsGm.SetGlobalBuffer((__gm__ int32_t *)ends);
        lengthsGm.SetGlobalBuffer((__gm__ int32_t *)lengths);
        tableGm.SetGlobalBuffer((__gm__ int32_t *)table);
        requestsGm.SetGlobalBuffer((__gm__ int64_t *)requests);
        generationsGm.SetGlobalBuffer((__gm__ int64_t *)generations);
        mappingGm.SetGlobalBuffer((__gm__ int32_t *)mapping);
        physicalGm.SetGlobalBuffer((__gm__ int64_t *)physical);
        stateGm.SetGlobalBuffer((__gm__ int64_t *)state);
        outputIdsGm.SetGlobalBuffer((__gm__ int32_t *)outputIds);
        outputSlotsGm.SetGlobalBuffer((__gm__ int32_t *)outputSlots);
        membershipGm.SetGlobalBuffer((__gm__ int16_t *)membership);
        batchCount = batch;
        requestCount = requestCapacity;
        sourceCount = sourceCapacity;
        residentCount = capacity;
        tableColumns = columns;
        hostSize = hostTokens;
        tailSize = mutableTail;
    }

    __aicore__ inline void BindScratch(LocalTensor<int32_t> scratch)
    {
        tokenLocal = scratch;
        slotLocal = tokenLocal[2080];
        planLocal = tokenLocal[4160].ReinterpretCast<int16_t>();
        missLocal = tokenLocal[6240].ReinterpretCast<int16_t>();
        hitLocal = tokenLocal[8320].ReinterpretCast<int16_t>();
    }

    __aicore__ inline void operator()(uint32_t row, LocalTensor<uint32_t> pools)
    {
        uint32_t request = 0;
        while (request < batchCount && endsGm.GetValue(request) <= row) ++request;
        if (request == batchCount) return;
        int64_t owner = requestsGm.GetValue(request);
        bool active = owner > 0 && owner < requestCount;
        int32_t length = lengthsGm.GetValue(row);
        int32_t firstRow = request == 0 ? 0 : endsGm.GetValue(request - 1);
        int32_t maxLength = 0;
        for (int32_t current = firstRow; current < endsGm.GetValue(request); ++current) {
            int32_t currentLength = lengthsGm.GetValue(current);
            maxLength = maxLength > currentLength ? maxLength : currentLength;
        }
        bool ready = active && stateGm.GetValue(owner * 16) == generationsGm.GetValue(owner) &&
            maxLength >= stateGm.GetValue(owner * 16 + 1);
        int32_t misses = 0;
        uint32_t mutablePresence = 0;
        uint32_t bodyMisses = 0;
        uint32_t bodyHits = 0;
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
        asc_vf_call<ClassifyPooledSelection>(dim3(256),
            (__ubuf__ const uint32_t *)pools.GetPhyAddr(), (__ubuf__ int32_t *)tokenLocal.GetPhyAddr(),
            (__ubuf__ int32_t *)slotLocal.GetPhyAddr(), (__ubuf__ int16_t *)planLocal.GetPhyAddr(),
            (__gm__ const int32_t *)tableGm.GetPhyAddr(), (__gm__ const int32_t *)mappingGm.GetPhyAddr(),
            (__gm__ const int64_t *)physicalGm.GetPhyAddr(), static_cast<uint64_t>(request) * tableColumns,
            static_cast<uint64_t>(active ? owner : 0) * sourceCount,
            static_cast<uint64_t>(active ? owner : 0) * residentCount, hostSize, sourceCount,
            residentCount, tableColumns, length, tailSize, active);
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        for (uint32_t column = 0; column < 2051; ++column) {
            int32_t token = tokenLocal.GetValue(column);
            int16_t plan = planLocal.GetValue(column);
            if (token >= 0 && plan == 0) ready = false;
            int32_t tailOffset = token - length + static_cast<int32_t>(tailSize);
            if (token >= 0 && tailOffset >= 0 && tailOffset < 32)
                mutablePresence |= 1U << tailOffset;
            if (plan < 0) missLocal.SetValue(misses++, column);
            if (column < 2048 && plan < 0) ++bodyMisses;
        }
        if (bodyMisses > 0 && bodyMisses < 2048)
            for (uint32_t column = 0; column < 2048; ++column)
                if (planLocal.GetValue(column) >= 0) hitLocal.SetValue(bodyHits++, column);
        planLocal.SetValue(2064, 0x5A4D);
        planLocal.SetValue(2065, 0x5A45);
        planLocal.SetValue(2066, 2051);
        planLocal.SetValue(2067, 14333);
        planLocal.SetValue(2068, active ? owner : 0);
        planLocal.SetValue(2069, 0x5A44);
        planLocal.SetValue(2070, residentCount);
        planLocal.SetValue(2071, ready ? misses : -1);
        planLocal.SetValue(2072, static_cast<int16_t>(mutablePresence & 0xFFFFU));
        planLocal.SetValue(2073, static_cast<int16_t>(mutablePresence >> 16));
        planLocal.SetValue(2074, bodyMisses);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID7);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID7);
        DataCopyPad(outputIdsGm[static_cast<uint64_t>(row) * 2051], tokenLocal, {1, 2051 * 4, 0, 0});
        DataCopyPad(outputSlotsGm[static_cast<uint64_t>(row) * 2051], slotLocal, {1, 2051 * 4, 0, 0});
        DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 14333], planLocal, {1, 2051 * 2, 0, 0});
        DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 16384], planLocal[2064], {1, 22, 0, 0});
        if (bodyMisses > 0 && bodyMisses < 2048)
            DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 4096], missLocal,
                DataCopyExtParams{1, bodyMisses * 2, 0, 0, 0});
        if (bodyHits > 0)
            DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 4096 + bodyMisses], hitLocal,
                DataCopyExtParams{1, bodyHits * 2, 0, 0, 0});
        if (ready && misses > 0)
            DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400], missLocal,
                DataCopyExtParams{1, static_cast<uint32_t>(misses * 2), 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
    }

private:
    GlobalTensor<int32_t> endsGm, lengthsGm, tableGm, mappingGm, outputIdsGm, outputSlotsGm;
    GlobalTensor<int64_t> requestsGm, generationsGm, physicalGm, stateGm;
    GlobalTensor<int16_t> membershipGm;
    LocalTensor<int32_t> tokenLocal, slotLocal;
    LocalTensor<int16_t> planLocal, missLocal, hitLocal;
    uint32_t batchCount, requestCount, sourceCount, residentCount, tableColumns, tailSize;
    uint64_t hostSize;
};
}
