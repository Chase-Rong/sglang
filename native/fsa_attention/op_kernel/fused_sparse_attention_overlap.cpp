#include "kernel_operator.h"
#include "fused_sparse_attention_overlap_template_tiling_key.h"
#include "arch35/fused_sparse_attention_overlap_kernel_mla.h"

using namespace AscendC;

#if defined(__DAV_C310_CUBE__)
#define FSA_OVERLAP_SELECTION_UPDATE_OP_IMPL(templateClass, tilingdataClass, ...)                 \
    do {                                                                                          \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,              \
            BaseApi::FusedSparseAttentionOverlapMatmulService<__VA_ARGS__>,                       \
            BaseApi::FusedSparseAttentionOverlapMatmulServiceDummy<__VA_ARGS__>>::type;           \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,               \
            BaseApi::FusedSparseAttentionOverlapVectorServiceDummy<__VA_ARGS__>,                  \
            BaseApi::FusedSparseAttentionOverlapVectorService<__VA_ARGS__>>::type;                \
        templateClass<CubeBlockType, VecBlockType> op;                                            \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                     \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,      \
            blocktable, queryRope, keyRope, selectionKRope, selectionKvCache,                     \
            selectionKvBlockTable, selectionKvBlockStatus, selectionMembershipMap,               \
            attentionOut, selectionKvActualSeq, user, nullptr, tiling, &tPipe);                   \
        op.Process();                                                                             \
    } while (0)
#else
#define FSA_OVERLAP_SELECTION_UPDATE_OP_IMPL(templateClass, tilingdataClass, ...)                 \
    do {                                                                                          \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,              \
            BaseApi::FusedSparseAttentionOverlapMatmulService<__VA_ARGS__>,                       \
            BaseApi::FusedSparseAttentionOverlapMatmulServiceDummy<__VA_ARGS__>>::type;           \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,               \
            BaseApi::FusedSparseAttentionOverlapVectorServiceDummy<__VA_ARGS__>,                  \
            BaseApi::FusedSparseAttentionOverlapVectorService<__VA_ARGS__>>::type;                \
        templateClass<CubeBlockType, VecBlockType> op;                                            \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                     \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in;                          \
        op.Init(query, key, value, sparseIndices, actualSeqLengthsQuery, actualSeqLengthsKV,      \
            blocktable, queryRope, keyRope, selectionKRope, selectionKvCache,                     \
            selectionKvBlockTable, selectionKvBlockStatus, selectionMembershipMap,               \
            attentionOut, selectionKvActualSeq, user, tiling_data, tiling, &tPipe);               \
        op.Process();                                                                             \
    } while (0)
#endif

template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int IS_SPLIT_G>
__global__ __aicore__ void fused_sparse_attention_overlap(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *actualSeqLengthsQuery, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *selectionKRope, __gm__ uint8_t *selectionKvCache,
    __gm__ uint8_t *selectionKvBlockTable, __gm__ uint8_t *selectionKvBlockStatus,
    __gm__ uint8_t *selectionMembershipMap,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *selectionKvActualSeq,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    constexpr bool pageAttention = KV_LAYOUT_T == FUSED_SPARSE_ATTENTION_OVERLAP_LAYOUT_PA_BSND;
    constexpr auto layout = static_cast<FusedSparseAttentionOverlapLayoutArch35>(LAYOUT_T);
    constexpr auto kvLayout = static_cast<FusedSparseAttentionOverlapLayoutArch35>(KV_LAYOUT_T);
    constexpr auto templateMode = TEMPLATE_MODE == C_TEMPLATE ?
        FusedSparseAttentionOverlapTemplateModeArch35::C_TEMPLATE_MODE :
        FusedSparseAttentionOverlapTemplateModeArch35::V_TEMPLATE_MODE;
    if constexpr (ORIG_DTYPE_QUERY == DT_BF16 && ORIG_DTYPE_KEY == DT_BF16 &&
                  ORIG_DTYPE_ATTENTION_OUT == DT_BF16) {
        FSA_OVERLAP_SELECTION_UPDATE_OP_IMPL(BaseApi::FusedSparseAttentionOverlapKernelMla,
            FusedSparseAttentionOverlapTilingDataMla, bfloat16_t, bfloat16_t, float, bfloat16_t,
            FLASH_DECODE, pageAttention, layout, kvLayout, templateMode, IS_SPLIT_G);
    } else {
        FSA_OVERLAP_SELECTION_UPDATE_OP_IMPL(BaseApi::FusedSparseAttentionOverlapKernelMla,
            FusedSparseAttentionOverlapTilingDataMla, half, half, float, half,
            FLASH_DECODE, pageAttention, layout, kvLayout, templateMode, IS_SPLIT_G);
    }
}
