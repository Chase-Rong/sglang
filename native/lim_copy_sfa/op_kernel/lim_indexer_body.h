#pragma once

#include "kernel_operator.h"
#include "lim_row_plan.h"

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
};

#include "arch35/lightning_indexer_kernel.h"

template <typename RowPlan>
__aicore__ inline void RunSglIndexer(GM_ADDR query, GM_ADDR key, GM_ADDR weights,
    GM_ADDR queryEnds, GM_ADDR poolLengths, GM_ADDR table, GM_ADDR indices, GM_ADDR scores,
    GM_ADDR workspace, uint32_t rows, uint32_t heads, uint32_t pools,
    uint32_t columns, AscendC::TPipe &pipe, RowPlan &rowPlan)
{
    LITilingData tiling{static_cast<uint32_t>(AscendC::GetBlockNum()),
        rows, heads, 1, pools, 3, 128, columns, 512, false};
    using IndexerType = LICommon::LIType<bfloat16_t, bfloat16_t, int32_t, true,
        LICommon::LI_LAYOUT::TND, LICommon::LI_LAYOUT::PA_BSND, false>;
    LIKernel::LightningIndexerKernel<IndexerType> indexer;
    indexer.Init(query, key, weights, queryEnds, poolLengths, table, indices,
        nullptr, workspace, &tiling, &pipe);
    if ASCEND_IS_AIV {
        indexer.vectorService.SetExportedScores(scores);
    }
    indexer.Process();
    AscendC::SyncAll<false>();
    indexer.FinishTopK(scores, rowPlan);
}
