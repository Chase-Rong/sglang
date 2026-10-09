#include "kernel_operator.h"
#if __CCE_AICORE__ == 310
#include "lim_resident.h"
#endif

extern "C" __global__ __aicore__ void sgl_lim_manage(
    GM_ADDR ids, GM_ADDR scores, GM_ADDR ends, GM_ADDR lengths, GM_ADDR table,
    GM_ADDR requests, GM_ADDR generations, GM_ADDR mapping, GM_ADDR sources, GM_ADDR physical,
    GM_ADDR state, GM_ADDR outputIds, GM_ADDR outputSlots, GM_ADDR membership,
    GM_ADDR unionSources, GM_ADDR unionSlots, GM_ADDR unionPhysical,
    uint32_t rows, uint32_t batch, uint32_t requestCapacity, uint32_t sourceCapacity,
    uint32_t capacity, uint32_t columns, uint32_t unionWidth, uint32_t scoreColumns,
    uint64_t hostTokens, uint32_t mutableTail, bool pooledInput, bool encodedScores)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if __CCE_AICORE__ == 310
    if ASCEND_IS_AIV {
        {
            AscendC::TPipe pipe;
            SglLim::ResidentManager manager;
            manager.Init(ids, scores, ends, lengths, table, requests, generations, mapping, sources,
                physical, state, outputIds, outputSlots, membership, unionSources, unionSlots,
                unionPhysical, rows, batch, requestCapacity, sourceCapacity, capacity, columns,
                unionWidth, scoreColumns, hostTokens, mutableTail, pooledInput, encodedScores, pipe);
            manager.Run();
        }
    }
#endif
}
