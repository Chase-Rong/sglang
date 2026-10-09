#include "kernel_operator.h"

#if __CCE_AICORE__ == 310
#include "lim_indexer_body.h"
#endif

extern "C" __global__ __aicore__ void sgl_lim_indexer(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR queryEnds,
    GM_ADDR poolLengths, GM_ADDR table, GM_ADDR indices, GM_ADDR scores,
    GM_ADDR workspace, uint32_t rows, uint32_t heads, uint32_t pools,
    uint32_t columns)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if __CCE_AICORE__ == 310
    AscendC::TPipe pipe;
    SglLim::EmptyRowPlan rowPlan;
    RunSglIndexer(query, key, weights, queryEnds, poolLengths, table, indices,
        scores, workspace, rows, heads, pools, columns, pipe, rowPlan);
#endif
}
