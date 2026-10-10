#include "kernel_operator.h"
#if __CCE_AICORE__ == 310
#include "copy_sfa/copy_sfa_tiling.h"
#include "copy_sfa/arch35/copy_sparse_attention_kernel_mla.h"

using namespace AscendC;

#include "copy_sfa/copy_sfa_transfer.h"
#endif

extern "C" __global__ __aicore__ void sgl_lim_copy_sfa(
    GM_ADDR query, GM_ADDR host, GM_ADDR resident, GM_ADDR ids, GM_ADDR ends,
    GM_ADDR lengths, GM_ADDR kvLengths, GM_ADDR table, GM_ADDR requests,
    GM_ADDR membership, GM_ADDR unionSources, GM_ADDR unionSlots, GM_ADDR unionPhysical,
    GM_ADDR physical, GM_ADDR state, GM_ADDR output, GM_ADDR workspace,
    uint32_t rows, uint32_t batch, uint32_t heads, uint32_t capacity,
    uint32_t columns, uint32_t unionWidth, uint32_t mutableTail,
    uint32_t requestCapacity, float scale)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if __CCE_AICORE__ == 310
    TPipe pipe;
    uint32_t work = SglCopy::BatchWork(requests, state, batch, requestCapacity);
    if (work & 1) {
        SglCopy::Fill(host, resident, requests, ends, unionSlots, unionPhysical, state,
            membership, batch, capacity, unionWidth, requestCapacity, pipe);
        SyncAll<false>();
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
    }
    CopySparseAttentionTilingDataMla tiling{};
    tiling.baseParams.batchSize = batch;
    tiling.baseParams.seqSize = columns * 128;
    tiling.baseParams.qSeqSize = rows;
    tiling.baseParams.blockSize = 128;
    tiling.baseParams.maxBlockNumPerBatch = columns;
    tiling.baseParams.scaleValue = scale;
    tiling.baseParams.nNumOfQInOneGroup = heads;
    tiling.baseParams.actualLenDimsQ = batch;
    tiling.baseParams.actualLenDimsKV = batch;
    tiling.baseParams.outputLayout = 1;
    tiling.baseParams.sparseMode = 3;
    tiling.baseParams.sparseBlockSize = 1;
    tiling.baseParams.sparseBlockCount = 2051;
    tiling.baseParams.selectionMembershipStride = 16400;
    tiling.baseParams.directKvL1 = (work & 3) == 2;
    tiling.singleCoreParams.usedCoreNum = GetBlockNum();
    using Cube = typename std::conditional<g_coreType == AIC,
        CopySfaBase::CopySparseAttentionMatmulService<bfloat16_t, bfloat16_t, float, bfloat16_t,
            false, true, CopySparseAttentionLayoutArch35::TND,
            CopySparseAttentionLayoutArch35::PA_BSND,
            CopySparseAttentionTemplateModeArch35::V_TEMPLATE_MODE, false>,
        CopySfaBase::CopySparseAttentionMatmulServiceDummy<bfloat16_t, bfloat16_t, float, bfloat16_t,
            false, true, CopySparseAttentionLayoutArch35::TND,
            CopySparseAttentionLayoutArch35::PA_BSND,
            CopySparseAttentionTemplateModeArch35::V_TEMPLATE_MODE, false>>::type;
    using Vector = typename std::conditional<g_coreType == AIC,
        CopySfaBase::CopySparseAttentionVectorServiceDummy<bfloat16_t, bfloat16_t, float, bfloat16_t,
            false, true, CopySparseAttentionLayoutArch35::TND,
            CopySparseAttentionLayoutArch35::PA_BSND,
            CopySparseAttentionTemplateModeArch35::V_TEMPLATE_MODE, false>,
        CopySfaBase::CopySparseAttentionVectorService<bfloat16_t, bfloat16_t, float, bfloat16_t,
            false, true, CopySparseAttentionLayoutArch35::TND,
            CopySparseAttentionLayoutArch35::PA_BSND,
            CopySparseAttentionTemplateModeArch35::V_TEMPLATE_MODE, false>>::type;
    CopySfaBase::CopySparseAttentionKernelMla<Cube, Vector> attention;
    attention.Init(query, host, host, ids, ends, kvLengths, table, nullptr, nullptr,
        nullptr, resident, nullptr, nullptr, membership, output,
        workspace + GetBlockNum() * 3 * 128 * 576 * 2, workspace,
        &tiling, nullptr, &pipe);
    attention.Process();
    if (work & 2) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<false>();
        if ASCEND_IS_AIV {
            pipe.Reset();
        }
        SglCopy::Commit(ends, lengths, requests, unionSources, unionSlots, unionPhysical,
            physical, state, membership, batch, capacity, unionWidth, mutableTail, requestCapacity, pipe);
    }
#endif
}
