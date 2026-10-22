#pragma once

#include "kernel_operator.h"
#include "lim_row_plan.h"

namespace SglLim {
__simt_vf__ __launch_bounds__(256) inline void CheckSharedPoolPages(
    __gm__ const int32_t *table, __gm__ const int32_t *ends, __gm__ int32_t *grouped,
    uint32_t batch, uint32_t columns)
{
    if (threadIdx.x == 0) grouped[0] = 1;
    asc_threadfence();
    asc_syncthreads();
    bool shared = true;
    uint32_t firstRow = 0;
    for (uint32_t request = 0; request < batch; ++request) {
        uint32_t endRow = ends[request];
        for (uint32_t row = firstRow + 1; row < endRow; ++row) {
            for (uint32_t column = threadIdx.x; column < columns; column += blockDim.x) {
                shared = shared && table[static_cast<uint64_t>(row) * columns + column] ==
                    table[static_cast<uint64_t>(firstRow) * columns + column];
            }
        }
        firstRow = endRow;
    }
    if (!shared) asc_atomic_min(grouped, 0);
}

__simt_vf__ __launch_bounds__(256) inline void GatherPooledSlotPayload(
    __ubuf__ const uint32_t *selected, __ubuf__ const int32_t *prefetched,
    __ubuf__ const int32_t *previous, __ubuf__ int32_t *output,
    __gm__ const int32_t *mapping, uint64_t mapBase, uint32_t sourceCount,
    uint32_t poolBase, uint32_t previousPools, uint32_t prefetchedTokens, uint32_t validPools)
{
    for (uint32_t tokenColumn = threadIdx.x; tokenColumn < 2048; tokenColumn += blockDim.x) {
        uint32_t selectedPool = selected[tokenColumn / 4];
        uint32_t lane = tokenColumn % 4;
        int32_t slot = -1;
        if (selectedPool < previousPools) {
            slot = previous[selectedPool * 4 + lane];
        } else {
            uint32_t relativePool = selectedPool - previousPools;
            uint64_t token = (static_cast<uint64_t>(poolBase) + relativePool) * 4 + lane;
            if (relativePool < validPools && token < sourceCount) {
                uint32_t localToken = relativePool * 4 + lane;
                slot = localToken < prefetchedTokens ? prefetched[localToken] : mapping[mapBase + token];
            }
        }
        output[tokenColumn] = slot;
    }
}

class PooledSlotPayload {
public:
    __aicore__ inline void Init(GM_ADDR ends, GM_ADDR lengths, GM_ADDR requests, GM_ADDR mapping,
        uint32_t batch, uint32_t requestCapacity, uint32_t sourceCapacity)
    {
        enabled = mapping != nullptr;
        if (!enabled) return;
        endsGm.SetGlobalBuffer((__gm__ int32_t *)ends);
        lengthsGm.SetGlobalBuffer((__gm__ int32_t *)lengths);
        requestsGm.SetGlobalBuffer((__gm__ int64_t *)requests);
        mappingGm.SetGlobalBuffer((__gm__ int32_t *)mapping);
        batchCount = batch;
        requestCount = requestCapacity;
        sourceCount = sourceCapacity;
    }

    __aicore__ inline void BeginRow(uint32_t row, LocalTensor<int32_t> scratch, uint32_t scratchCount)
    {
        active = false;
        produced = false;
        if (!enabled || scratchCount < 16384) return;
        uint32_t request = 0;
        while (request < batchCount && endsGm.GetValue(request) <= row) ++request;
        if (request == batchCount) return;
        int64_t owner = requestsGm.GetValue(request);
        int32_t length = lengthsGm.GetValue(row);
        if (owner <= 0 || owner >= requestCount || length <= 2048) return;
        active = true;
        mapBase = static_cast<uint64_t>(owner) * sourceCount;
        poolCount = static_cast<uint32_t>(length) / 4;
        prefetchCapacity = scratchCount - 4096;
        prefetchLocal = scratch;
        payloadLocal[0] = scratch[prefetchCapacity];
        payloadLocal[1] = scratch[prefetchCapacity + 2048];
    }

    __aicore__ inline void Prefetch(uint32_t loop, uint32_t trunk)
    {
        if (!active) return;
        poolBase = loop * trunk;
        validPools = poolBase < poolCount ? poolCount - poolBase : 0;
        if (validPools > trunk) validPools = trunk;
        uint64_t tokenBase = static_cast<uint64_t>(poolBase) * 4;
        uint32_t available = tokenBase < sourceCount ? sourceCount - tokenBase : 0;
        prefetchTokens = validPools * 4;
        if (prefetchTokens > available) prefetchTokens = available;
        if (prefetchTokens > prefetchCapacity) prefetchTokens = prefetchCapacity;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
        if (prefetchTokens > 0) {
            DataCopyPad(prefetchLocal, mappingGm[mapBase + tokenBase],
                DataCopyExtParams{1, prefetchTokens * 4U, 0, 0, 0},
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
    }

    __aicore__ inline void Finish(LocalTensor<uint32_t> selected, uint32_t loop)
    {
        if (!active) return;
        SetFlag<HardEvent::MTE2_V>(EVENT_ID6);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
        PipeBarrier<PIPE_V>();
        outputBank = loop % 2;
        asc_vf_call<GatherPooledSlotPayload>(dim3(256),
            (__ubuf__ const uint32_t *)selected.GetPhyAddr(),
            (__ubuf__ const int32_t *)prefetchLocal.GetPhyAddr(),
            (__ubuf__ const int32_t *)payloadLocal[1 - outputBank].GetPhyAddr(),
            (__ubuf__ int32_t *)payloadLocal[outputBank].GetPhyAddr(),
            (__gm__ const int32_t *)mappingGm.GetPhyAddr(), mapBase, sourceCount,
            poolBase, loop == 0 ? 0U : 512U, prefetchTokens, validPools);
        SetFlag<HardEvent::V_S>(EVENT_ID6);
        WaitFlag<HardEvent::V_S>(EVENT_ID6);
        produced = true;
    }

    __aicore__ inline __ubuf__ const int32_t *SelectedSlots()
    {
        return produced ? (__ubuf__ const int32_t *)payloadLocal[outputBank].GetPhyAddr() : nullptr;
    }

private:
    GlobalTensor<int32_t> endsGm, lengthsGm, mappingGm;
    GlobalTensor<int64_t> requestsGm;
    LocalTensor<int32_t> prefetchLocal, payloadLocal[2];
    uint64_t mapBase = 0;
    uint32_t batchCount = 0, requestCount = 0, sourceCount = 0, poolCount = 0;
    uint32_t prefetchCapacity = 0, prefetchTokens = 0, poolBase = 0, validPools = 0, outputBank = 0;
    bool enabled = false, active = false, produced = false;
};

}

struct LITilingData {
    uint32_t usedCoreNum;
    uint32_t bSize;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t sparseMode;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t sparseCount;
    bool returnValue;
    bool groupedQueries;
    uint32_t totalRows;
};

#include "arch35/lightning_indexer_kernel.h"

template <typename RowPlan>
__aicore__ inline void RunSglIndexer(GM_ADDR query, GM_ADDR key, GM_ADDR weights,
    GM_ADDR queryEnds, GM_ADDR poolLengths, GM_ADDR table, GM_ADDR indices, GM_ADDR scores,
    GM_ADDR workspace, uint32_t rows, uint32_t heads, uint32_t pools,
    uint32_t columns, AscendC::TPipe &pipe, RowPlan &rowPlan,
    GM_ADDR ends = nullptr, GM_ADDR lengths = nullptr, GM_ADDR requests = nullptr,
    GM_ADDR mapping = nullptr, uint32_t batch = 0, uint32_t requestCapacity = 0, uint32_t sourceCapacity = 0,
    bool groupedQueries = false)
{
    LITilingData tiling{static_cast<uint32_t>(AscendC::GetBlockNum()),
        groupedQueries ? batch : rows, heads, groupedQueries ? 4U : 1U,
        pools, groupedQueries ? 0U : 3U, 128, columns, 512, false, groupedQueries, rows};
    using IndexerType = LICommon::LIType<bfloat16_t, bfloat16_t, int32_t, true,
        LICommon::LI_LAYOUT::TND, LICommon::LI_LAYOUT::PA_BSND, false>;
    LIKernel::LightningIndexerKernel<IndexerType> indexer;
    indexer.Init(query, key, weights, groupedQueries ? ends : queryEnds, poolLengths, table, indices,
        nullptr, workspace, &tiling, &pipe);
    if ASCEND_IS_AIV {
        indexer.vectorService.SetExportedScores(scores);
        indexer.vectorService.slotPayload.Init(ends, lengths, requests, mapping,
            batch, requestCapacity, sourceCapacity);
    }
    indexer.Process();
    AscendC::SyncAll<false>();
    indexer.FinishTopK(scores, rowPlan);
}
