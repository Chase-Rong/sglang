#pragma once

#include "kernel_operator.h"
#include "lim_classify.h"
#include "lim_schedule.h"

namespace SglLim {
using namespace AscendC;

class ResidentManager {
public:
    __aicore__ inline void Init(GM_ADDR ids, GM_ADDR scores, GM_ADDR ends, GM_ADDR lengths,
        GM_ADDR table, GM_ADDR requests, GM_ADDR generations, GM_ADDR mapping, GM_ADDR sources,
        GM_ADDR physical, GM_ADDR state, GM_ADDR outputIds, GM_ADDR outputSlots, GM_ADDR membership,
        GM_ADDR unionSources, GM_ADDR unionSlots, GM_ADDR unionPhysical, uint32_t rows, uint32_t batch,
        uint32_t requestCapacity, uint32_t sourceCapacity, uint32_t residentCapacity, uint32_t columns,
        uint32_t unionWidth, uint32_t scoreColumns, uint64_t hostTokens, uint32_t mutableTail,
        bool pooledInput, bool encodedScores, TPipe &pipe, bool preparedRows = false)
    {
        idsGm.SetGlobalBuffer((__gm__ int32_t *)ids);
        scoresGm.SetGlobalBuffer((__gm__ uint32_t *)scores);
        endsGm.SetGlobalBuffer((__gm__ int32_t *)ends);
        lengthsGm.SetGlobalBuffer((__gm__ int32_t *)lengths);
        tableGm.SetGlobalBuffer((__gm__ int32_t *)table);
        requestsGm.SetGlobalBuffer((__gm__ int64_t *)requests);
        generationsGm.SetGlobalBuffer((__gm__ int64_t *)generations);
        mappingGm.SetGlobalBuffer((__gm__ int32_t *)mapping);
        sourcesGm.SetGlobalBuffer((__gm__ int32_t *)sources);
        physicalGm.SetGlobalBuffer((__gm__ int64_t *)physical);
        stateGm.SetGlobalBuffer((__gm__ int64_t *)state);
        outputIdsGm.SetGlobalBuffer((__gm__ int32_t *)outputIds);
        outputSlotsGm.SetGlobalBuffer((__gm__ int32_t *)outputSlots);
        membershipGm.SetGlobalBuffer((__gm__ int16_t *)membership);
        unionSourcesGm.SetGlobalBuffer((__gm__ int32_t *)unionSources);
        unionSlotsGm.SetGlobalBuffer((__gm__ int32_t *)unionSlots);
        unionPhysicalGm.SetGlobalBuffer((__gm__ int64_t *)unionPhysical);
        rowCount = rows;
        batchCount = batch;
        requestCount = requestCapacity;
        sourceCount = sourceCapacity;
        capacity = residentCapacity;
        tableColumns = columns;
        unionSize = unionWidth;
        scoreStride = scoreColumns;
        hostSize = hostTokens;
        tailSize = mutableTail;
        pooled = pooledInput;
        encoded = encodedScores;
        prepared = preparedRows;
        pipe.InitBuffer(unionBuffer, unionSize * 4);
        pipe.InitBuffer(residentBuffer, (capacity < 5120 ? 5120 : capacity) * 4);
        pipe.InitBuffer(heapBuffer, capacity * 4);
        pipe.InitBuffer(rowBuffer, 2080 * 16);
        unionLocal = unionBuffer.Get<int32_t>();
        residentLocal = residentBuffer.Get<int32_t>();
        heapLocal = heapBuffer.Get<int32_t>();
        victimLocal = residentLocal;
        rowIds = rowBuffer.Get<int32_t>();
        rowSlots = rowIds[2080];
        rowPlan = rowIds[4160].ReinterpretCast<int16_t>();
    }

    __aicore__ inline void Run()
    {
        for (uint32_t request = GetBlockIdx(); request < batchCount; request += GetBlockNum() * 2) {
            ProcessRequest(request);
        }
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(mappingGm);
    }

private:
    __aicore__ inline void PublishUnion(uint32_t request, uint32_t unique, uint64_t mapBase)
    {
        int64_t owner = requestsGm.GetValue(request);
        if (owner > 0 && owner < requestCount) stateGm.SetValue(owner * 16 + 3, unique);
        if (unique == 0) return;
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
        asc_vf_call<PublishMissUnion>(dim3(256), (__ubuf__ const int32_t *)unionLocal.GetPhyAddr(),
            (__gm__ const int32_t *)mappingGm.GetPhyAddr(), (__gm__ const int32_t *)tableGm.GetPhyAddr(),
            (__gm__ int32_t *)unionSourcesGm.GetPhyAddr(), (__gm__ int32_t *)unionSlotsGm.GetPhyAddr(),
            (__gm__ int64_t *)unionPhysicalGm.GetPhyAddr(), mapBase,
            static_cast<uint64_t>(request) * tableColumns, static_cast<uint64_t>(request) * unionSize, unique);
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
    }

    __aicore__ inline bool FinishPrepared(uint32_t request, int32_t start, int32_t end,
        int64_t owner, int32_t maxLength)
    {
        if (!prepared || owner <= 0 || owner >= requestCount) return false;
        uint32_t totalMisses = 0;
        for (int32_t row = start; row < end; ++row) {
            int16_t count = membershipGm.GetValue(static_cast<uint64_t>(row) * 16400 + 16391);
            if (count < 0) return false;
            totalMisses += count;
        }
        stateGm.SetValue(owner * 16 + 1, maxLength);
        stateGm.SetValue(owner * 16 + 2, 0);
        uint32_t unique = 0;
        if (totalMisses > 0) {
            asc_vf_call<CollectPreparedUnion>(dim3(256),
                (__gm__ const int16_t *)membershipGm.GetPhyAddr(), (__gm__ const int32_t *)outputIdsGm.GetPhyAddr(),
                (__gm__ const int32_t *)outputSlotsGm.GetPhyAddr(), (__ubuf__ uint32_t *)residentLocal.GetPhyAddr(),
                (__ubuf__ int32_t *)unionLocal.GetPhyAddr(), (__ubuf__ uint32_t *)rowIds.GetPhyAddr(),
                static_cast<uint32_t>(start), static_cast<uint32_t>(end), totalMisses, capacity);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
            unique = rowIds.GetValue(255);
        }
        PublishUnion(request, unique, owner * sourceCount);
        return true;
    }

    __aicore__ inline void ClassifyRow(uint32_t request, uint32_t row, int64_t owner,
        uint32_t unique, bool repair)
    {
        bool active = owner > 0 && owner < requestCount;
        SetFlag<HardEvent::S_V>(EVENT_ID7);
        WaitFlag<HardEvent::S_V>(EVENT_ID7);
        asc_vf_call<ClassifyManagedRows>(dim3(256), (__gm__ const int32_t *)idsGm.GetPhyAddr(),
            (__gm__ const int32_t *)tableGm.GetPhyAddr(), (__gm__ const int32_t *)mappingGm.GetPhyAddr(),
            (__gm__ const int64_t *)physicalGm.GetPhyAddr(), (__gm__ const int32_t *)lengthsGm.GetPhyAddr(),
            (__ubuf__ int32_t *)rowIds.GetPhyAddr(),
            (__ubuf__ int32_t *)rowSlots.GetPhyAddr(), (__ubuf__ int16_t *)rowPlan.GetPhyAddr(),
            (__ubuf__ const int32_t *)unionLocal.GetPhyAddr(), (__ubuf__ const int32_t *)heapLocal.GetPhyAddr(),
            static_cast<uint64_t>(request) * tableColumns, static_cast<uint64_t>(active ? owner : 0) * sourceCount,
            static_cast<uint64_t>(active ? owner : 0) * capacity, hostSize, sourceCount, capacity, tableColumns,
            row, row + 1, tailSize, unique, pooled, active, repair);
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
    }

    __aicore__ inline uint32_t SortUnion(uint32_t count)
    {
        if (count == 0) return 0;
        auto keys = rowIds.ReinterpretCast<float>();
        auto indices = rowIds[1024].ReinterpretCast<uint32_t>();
        auto pairs = rowIds[2048].ReinterpretCast<float>();
        auto scratch = rowIds[4096].ReinterpretCast<float>();
        uint32_t chunks = (count + 1023) / 1024;
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            uint32_t base = chunk * 1024;
            uint32_t valid = count - base < 1024 ? count - base : 1024;
            uint32_t padded = (valid + 31) / 32 * 32;
            SetFlag<HardEvent::S_V>(EVENT_ID7);
            WaitFlag<HardEvent::S_V>(EVENT_ID7);
            asc_vf_call<PrepareUnionSort>(dim3(256), (__ubuf__ const int32_t *)unionLocal.GetPhyAddr(),
                (__ubuf__ float *)keys.GetPhyAddr(), (__ubuf__ uint32_t *)indices.GetPhyAddr(), base, valid, padded);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
            Sort<float, true>(pairs, keys, indices, scratch, padded / 32);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
            asc_vf_call<UnpackUnionSort>(dim3(256), (__ubuf__ const int32_t *)pairs.GetPhyAddr(),
                (__ubuf__ int32_t *)unionLocal.GetPhyAddr(), base, valid);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
        }
        auto prefixes = rowIds.ReinterpretCast<uint16_t>();
        auto counts = rowIds[8192 - 256].ReinterpretCast<uint32_t>();
        asc_vf_call<BuildUnionPrefixes>(dim3(256), (__ubuf__ const int32_t *)unionLocal.GetPhyAddr(),
            (__ubuf__ uint16_t *)prefixes.GetPhyAddr(), (__ubuf__ uint32_t *)counts.GetPhyAddr(), count);
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        uint32_t unique = 0;
        for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
            uint32_t limit = (chunk + 1) * 1024 < count ? (chunk + 1) * 1024 : count;
            unique += prefixes.GetValue(limit - 1);
        }
        if (unique > capacity) return capacity + 1;
        asc_vf_call<MergeUniqueUnion>(dim3(256), (__ubuf__ const int32_t *)unionLocal.GetPhyAddr(),
            (__ubuf__ const uint16_t *)prefixes.GetPhyAddr(), (__ubuf__ int32_t *)residentLocal.GetPhyAddr(),
            count, capacity);
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        asc_vf_call<CopyUnion>(dim3(256), (__ubuf__ const int32_t *)residentLocal.GetPhyAddr(),
            (__ubuf__ int32_t *)unionLocal.GetPhyAddr(), unique);
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        return unique;
    }

    __aicore__ inline void CopyRow(uint32_t row, LocalTensor<int32_t> ids, LocalTensor<int32_t> slots,
        LocalTensor<int16_t> plans, LocalTensor<int16_t> misses, LocalTensor<int16_t> hits)
    {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID7);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID7);
        DataCopyPad(outputIdsGm[static_cast<uint64_t>(row) * 2051], ids, {1, 2051 * 4, 0, 0});
        DataCopyPad(outputSlotsGm[static_cast<uint64_t>(row) * 2051], slots, {1, 2051 * 4, 0, 0});
        DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 16384 - 2051], plans, {1, 2051 * 2, 0, 0});
        DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 16384], plans[2064], {1, 22, 0, 0});
        uint32_t count = static_cast<uint16_t>(plans.GetValue(2074));
        if (count > 0 && count < 2048)
            DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 4096],
                misses, DataCopyExtParams{1, count * 2, 0, 0, 0});
        if (count > 0 && count < 2048)
            DataCopyPad(membershipGm[static_cast<uint64_t>(row) * 16400 + 4096 + count],
                hits, DataCopyExtParams{1, (2048 - count) * 2, 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID7);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID7);
    }

    __aicore__ inline void ProcessRequest(uint32_t request)
    {
        int32_t start = request == 0 ? 0 : endsGm.GetValue(request - 1);
        int32_t end = endsGm.GetValue(request);
        int64_t owner = requestsGm.GetValue(request);
        if (start < 0 || end < start || end > rowCount || end - start > 6 || (end - start) * 2051 > unionSize) return;
        bool active = owner > 0 && owner < requestCount;
        uint64_t residentBase = active ? owner * capacity : 0;
        uint64_t mapBase = active ? owner * sourceCount : 0;
        uint32_t total = 0;
        int32_t maxLength = 0;
        for (int32_t row = start; row < end; ++row) {
            int32_t length = lengthsGm.GetValue(row);
            maxLength = maxLength > length ? maxLength : length;
        }
        if (FinishPrepared(request, start, end, owner, maxLength)) return;
        bool reset = active && (stateGm.GetValue(owner * 16) != generationsGm.GetValue(owner) ||
            maxLength < stateGm.GetValue(owner * 16 + 1));
        if (active) {
            stateGm.SetValue(owner * 16 + 2, reset);
            if (reset) {
                asc_vf_call<ResetResident>(dim3(256), (__gm__ int32_t *)mappingGm.GetPhyAddr(),
                    (__gm__ int32_t *)sourcesGm.GetPhyAddr(), (__gm__ int64_t *)physicalGm.GetPhyAddr(),
                    mapBase, residentBase, sourceCount, capacity);
                SetFlag<HardEvent::V_S>(EVENT_ID7);
                WaitFlag<HardEvent::V_S>(EVENT_ID7);
            }
        }
        Duplicate(heapLocal, static_cast<int32_t>(1), capacity);
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        for (int32_t row = start; row < end && active; ++row) {
            ClassifyRow(request, row, owner, 0, false);
            asc_vf_call<CollectRowMisses>(dim3(256), (__ubuf__ const int32_t *)rowIds.GetPhyAddr(),
                (__ubuf__ const int32_t *)rowSlots.GetPhyAddr(), (__ubuf__ const int16_t *)rowPlan.GetPhyAddr(),
                (__ubuf__ int32_t *)unionLocal.GetPhyAddr(), (__ubuf__ uint32_t *)heapLocal.GetPhyAddr(),
                (__ubuf__ uint32_t *)rowPlan[2080].GetPhyAddr(), total);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
            total += rowPlan[2080].ReinterpretCast<uint32_t>().GetValue(255);
        }
        uint32_t unique = SortUnion(total);
        if (unique > capacity) return;
        if (unique > 0) {
            auto ranks = rowIds.ReinterpretCast<uint16_t>();
            auto counts = rowIds[8192 - 256].ReinterpretCast<uint32_t>();
            asc_vf_call<RankNewSlots>(dim3(256), (__ubuf__ const int32_t *)unionLocal.GetPhyAddr(),
                (__ubuf__ uint16_t *)ranks.GetPhyAddr(), (__ubuf__ uint32_t *)counts.GetPhyAddr(),
                (__gm__ const int32_t *)mappingGm.GetPhyAddr(), mapBase, unique, capacity);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
            uint32_t missing = counts.GetValue(255);
            if (missing > 0) {
                uint32_t chunks = (capacity + 2047) / 2048;
                uint32_t firstChunk = SglLimSchedule::HashVictimScanSeed(maxLength) % chunks;
                asc_vf_call<SelectVictims>(dim3(256), (__ubuf__ const int32_t *)heapLocal.GetPhyAddr(),
                    (__ubuf__ int32_t *)victimLocal.GetPhyAddr(), (__ubuf__ uint32_t *)counts.GetPhyAddr(),
                    (__gm__ const int32_t *)sourcesGm.GetPhyAddr(), residentBase, capacity, missing, firstChunk * 2048);
                SetFlag<HardEvent::V_S>(EVENT_ID7);
                WaitFlag<HardEvent::V_S>(EVENT_ID7);
                if (counts.GetValue(255) < missing) return;
            }
            asc_vf_call<ApplyResidentUpdates>(dim3(256), (__ubuf__ const int32_t *)unionLocal.GetPhyAddr(),
                (__ubuf__ const uint16_t *)ranks.GetPhyAddr(), (__ubuf__ const int32_t *)victimLocal.GetPhyAddr(),
                (__ubuf__ int32_t *)heapLocal.GetPhyAddr(), (__gm__ int32_t *)mappingGm.GetPhyAddr(),
                (__gm__ int32_t *)sourcesGm.GetPhyAddr(), (__gm__ int64_t *)physicalGm.GetPhyAddr(),
                mapBase, residentBase, unique, sourceCount);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
        }
        if (active) {
            stateGm.SetValue(owner * 16, generationsGm.GetValue(owner));
            stateGm.SetValue(owner * 16 + 1, maxLength);
        }
        PublishUnion(request, unique, mapBase);
        if (unique > 0) {
            asc_vf_call<CopyUnion>(dim3(256), (__ubuf__ const int32_t *)unionLocal.GetPhyAddr(),
                (__ubuf__ int32_t *)residentLocal.GetPhyAddr(), unique);
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
        }
        asc_vf_call<ClassifyManagedRows>(dim3(256), (__gm__ const int32_t *)idsGm.GetPhyAddr(),
            (__gm__ const int32_t *)tableGm.GetPhyAddr(), (__gm__ const int32_t *)mappingGm.GetPhyAddr(),
            (__gm__ const int64_t *)physicalGm.GetPhyAddr(), (__gm__ const int32_t *)lengthsGm.GetPhyAddr(),
            (__ubuf__ int32_t *)unionLocal.GetPhyAddr(), (__ubuf__ int32_t *)nullptr,
            (__ubuf__ int16_t *)rowIds.GetPhyAddr(), (__ubuf__ const int32_t *)residentLocal.GetPhyAddr(),
            (__ubuf__ const int32_t *)heapLocal.GetPhyAddr(), static_cast<uint64_t>(request) * tableColumns,
            mapBase, residentBase, hostSize, sourceCount, capacity, tableColumns,
            static_cast<uint32_t>(start), static_cast<uint32_t>(end), tailSize, unique, pooled, active, true);
        SetFlag<HardEvent::V_S>(EVENT_ID7);
        WaitFlag<HardEvent::V_S>(EVENT_ID7);
        for (int32_t row = start; row < end; ++row) {
            auto tokens = unionLocal[(row - start) * 2080];
            auto plans = rowIds.ReinterpretCast<int16_t>()[(row - start) * 2080];
            auto missOrder = residentLocal[2080].ReinterpretCast<int16_t>();
            auto hitOrder = residentLocal[3120].ReinterpretCast<int16_t>();
            asc_vf_call<CompactRowPlan>(dim3(256), (__ubuf__ const int32_t *)tokens.GetPhyAddr(),
                (__ubuf__ int16_t *)plans.GetPhyAddr(), (__ubuf__ int16_t *)missOrder.GetPhyAddr(),
                (__ubuf__ int16_t *)hitOrder.GetPhyAddr(), (__ubuf__ uint32_t *)residentLocal[4160].GetPhyAddr(),
                lengthsGm.GetValue(row), tailSize, false, (__ubuf__ int32_t *)residentLocal.GetPhyAddr());
            SetFlag<HardEvent::V_S>(EVENT_ID7);
            WaitFlag<HardEvent::V_S>(EVENT_ID7);
            plans.SetValue(2064, 0x5A4D);
            plans.SetValue(2065, 0x5A45);
            plans.SetValue(2066, 2051);
            plans.SetValue(2067, 16384 - 2051);
            plans.SetValue(2068, active ? owner : 0);
            plans.SetValue(2069, 0x5A44);
            plans.SetValue(2070, capacity);
            plans.SetValue(2071, 0);
            CopyRow(row, tokens, residentLocal, plans, missOrder, hitOrder);
        }
    }

    GlobalTensor<int32_t> idsGm, endsGm, lengthsGm, tableGm, mappingGm, sourcesGm, outputIdsGm, outputSlotsGm;
    GlobalTensor<int32_t> unionSourcesGm, unionSlotsGm;
    GlobalTensor<uint32_t> scoresGm;
    GlobalTensor<int64_t> requestsGm, generationsGm, physicalGm, stateGm, unionPhysicalGm;
    GlobalTensor<int16_t> membershipGm;
    TBuf<TPosition::VECCALC> unionBuffer, residentBuffer, heapBuffer, rowBuffer;
    LocalTensor<int32_t> unionLocal, residentLocal, heapLocal, rowIds, rowSlots;
    LocalTensor<int16_t> rowPlan;
    LocalTensor<int32_t> victimLocal;
    uint32_t rowCount, batchCount, requestCount, sourceCount, capacity, tableColumns, unionSize, scoreStride, tailSize;
    uint64_t hostSize;
    bool pooled, encoded, prepared;
};
}
