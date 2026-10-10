#include "kernel_operator.h"
#if __CCE_AICORE__ == 310
#include "lim_indexer_body.h"
#include "lim_resident.h"
#endif

extern "C" __global__ __aicore__ void sgl_lim_indexer_manage(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR indexEnds, GM_ADDR poolLengths,
    GM_ADDR indexTable, GM_ADDR indices, GM_ADDR scores, GM_ADDR workspace,
    GM_ADDR ends, GM_ADDR lengths, GM_ADDR table, GM_ADDR requests, GM_ADDR generations,
    GM_ADDR mapping, GM_ADDR sources, GM_ADDR physical, GM_ADDR state,
    GM_ADDR outputIds, GM_ADDR outputSlots, GM_ADDR membership,
    GM_ADDR unionSources, GM_ADDR unionSlots, GM_ADDR unionPhysical,
    uint32_t rows, uint32_t heads, uint32_t pools, uint32_t indexColumns,
    uint32_t batch, uint32_t requestCapacity, uint32_t sourceCapacity,
    uint32_t capacity, uint32_t columns, uint32_t unionWidth, uint64_t hostTokens, uint32_t mutableTail)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if __CCE_AICORE__ == 310
    AscendC::TPipe pipe;
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            AscendC::GlobalTensor<int32_t> queryEndsGm, poolLengthsGm, lengthsGm;
            queryEndsGm.SetGlobalBuffer((__gm__ int32_t *)indexEnds);
            poolLengthsGm.SetGlobalBuffer((__gm__ int32_t *)poolLengths);
            lengthsGm.SetGlobalBuffer((__gm__ int32_t *)lengths);
            for (uint32_t row = 0; row < rows; ++row) {
                int32_t length = lengthsGm.GetValue(row);
                queryEndsGm.SetValue(row, row + 1);
                poolLengthsGm.SetValue(row, length > 0 ? length / 4 : 0);
            }
            AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                AscendC::DcciDst::CACHELINE_OUT>(poolLengthsGm);
            if (rows > batch) {
                asc_vf_call<SglLim::CheckSharedPoolPages>(dim3(256),
                    (__gm__ const int32_t *)indexTable, (__gm__ const int32_t *)ends,
                    (__gm__ int32_t *)workspace, batch, indexColumns);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
            }
        }
    }
    AscendC::SyncAll<false>();
    bool groupedQueries = false;
    if (rows > batch) {
        AscendC::GlobalTensor<int32_t> groupedGm;
        groupedGm.SetGlobalBuffer((__gm__ int32_t *)workspace);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(groupedGm);
        groupedQueries = groupedGm.GetValue(0) != 0;
    }
    SglLim::ResidentRowPlan rowPlan;
    if ASCEND_IS_AIV {
        rowPlan.Init(ends, lengths, table, requests, generations, mapping, physical, state,
            outputIds, outputSlots, membership, batch, requestCapacity, sourceCapacity,
            capacity, columns, hostTokens, mutableTail);
    }
    RunSglIndexer(query, key, weights, indexEnds, poolLengths, indexTable, indices,
        scores, workspace, rows, heads, pools, indexColumns, pipe, rowPlan,
        ends, lengths, requests, mapping, batch, requestCapacity, sourceCapacity, groupedQueries);
    AscendC::SyncAll<false>();
    pipe.Reset();
    if ASCEND_IS_AIV {
        {
            SglLim::ResidentManager manager;
            manager.Init(indices, scores, ends, lengths, table, requests, generations,
                mapping, sources, physical, state, outputIds, outputSlots, membership,
                unionSources, unionSlots, unionPhysical, rows, batch, requestCapacity,
                sourceCapacity, capacity, columns, unionWidth, pools, hostTokens, mutableTail,
                true, true, pipe, true);
            manager.Run();
        }
    }
#endif
}
