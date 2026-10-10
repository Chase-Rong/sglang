/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file copy_sparse_attention_service_vector_mla.h
 * \brief
 */
#ifndef COPY_SPARSE_ATTENTION_SERVICE_VECTOR_MLA_ARCH35_H
#define COPY_SPARSE_ATTENTION_SERVICE_VECTOR_MLA_ARCH35_H

#include "util_regbase.h"
#include "copy_sparse_attention_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "../common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"

#include "../common/arch35/vf/vf_flashupdate_new.h"

#include "../common/buffers_policy.h"
#include "../common/buffer_manager.h"
#include "../common/buffer.h"
#include "../../lim_schedule.h"
#include "copy_sparse_attention_layout.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;

namespace CopySfaBase {

TEMPLATES_DEF
class CopySparseAttentionVectorService {
public:
    // Buffer sizes in bytes
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================Compile-time base-block constants================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr uint32_t dVTemplateTypeInput = 576;
    static constexpr float R0 = 1.0f;

    // ==================== Functions ======================
    __aicore__ inline CopySparseAttentionVectorService() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const CopySparseAttentionTilingDataMla *__restrict tiling,
                                        CVSharedParams &sharedParams, uint8_t subBlockIdx,
                                        __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            tilingData = tiling;
            if (actualSeqLengthsQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ);
            }
            if (actualSeqLengths != nullptr) {
                actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengths);
            }
            this->InitCubeVecSharedParams(sharedParams, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // Initialize LocalTensor objects.
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    // Initialize attentionOut in GM.
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                            __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *selectionKRope,
                                            __gm__ uint8_t *selectionKvCache,
                                            __gm__ uint8_t *selectionKvBlockTable,
                                            __gm__ uint8_t *selectionKvBlockStatus,
                                            __gm__ uint8_t *selectionMembershipMap,
                                            __gm__ uint8_t *selectionKvActualSeq);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    __aicore__ inline void CompleteMissPrefetch();
    __aicore__ inline void SubmitMissPrefetch(const RunInfo &runInfo, ConstInfo &constInfo);

private:
    __aicore__ inline void PrepareMissOwners(uint32_t batch, uint32_t query);
    __aicore__ inline bool OwnsMissSlot(uint32_t slot)
    {
        return (earlierMissSlots[slot / 32] & (1U << (slot % 32))) == 0;
    }
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx,
        int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo,
        ConstInfo &constInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb,
        int64_t startRow, int64_t keyOffset, ConstInfo &constInfo);
    __aicore__ inline int64_t GetSelectionRow(const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline int64_t GetSelectionTokenOffset(int64_t selectionRow, int32_t selectionSlot);
    __aicore__ inline void CopyInSingleSelectionKv(LocalTensor<KV_T> kvInUb,
        int64_t startRow, int64_t selectionTokenOffset, ConstInfo &constInfo);
    __aicore__ inline bool PrepareExternalSelectionPlan(const RunInfo &runInfo,
        ConstInfo &constInfo, LocalTensor<int16_t> selectionPlanLocal, int64_t &logicalSelectionRow);
    __aicore__ inline uint32_t CopyInKvWithExternalPlan(LocalTensor<KV_T> kvInUb,
        LocalTensor<int16_t> compactPlanLocal, int64_t startRow, int64_t token0Idx,
        int64_t token1Idx, int16_t planValue0, int16_t planValue1,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutSelectionMisses(LocalTensor<KV_T> kvInUb,
        LocalTensor<int16_t> compactPlanLocal, int64_t dealRow,
        int64_t selectionRow, ConstInfo &constInfo);
    __aicore__ inline void SignalKvReady(
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm);
    __aicore__ inline int64_t CopySparseKvBatch(
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        LocalTensor<int16_t> selectionPlanLocal, LocalTensor<int16_t> compactPlanLocal,
        int64_t &s2, uint32_t pingPong, const RunInfo &runInfo, ConstInfo &constInfo);
    template <bool HIT_ONLY>
    __aicore__ inline bool CopyPlannedKvWindow(LocalTensor<KV_T> destination,
        LocalTensor<int16_t> selectionPlanLocal, LocalTensor<int16_t> compactPlanLocal,
        int64_t start, int64_t count, const RunInfo &runInfo, ConstInfo &constInfo);
    /* VEC2_RES_T is the current type of bmm2ResUb. No cast is needed when VEC2_RES_T == Q_T.
       Invalid-row cases currently require a cast by default. */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo &runInfo,
        ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
        int64_t vec2S1Idx, int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void CopyFALseToGm(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams &sharedParams, uint8_t subBlockIdx);
    __aicore__ inline void GetExtremeValue(T &negativeScalar);

    TPipe *tPipe;
    const CopySparseAttentionTilingDataMla *__restrict tilingData;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<KV_T> selectionKRopeGm;
    GlobalTensor<KV_T> selectionKvCacheGm;
    GlobalTensor<int32_t> selectionKvBlockTableGm;
    GlobalTensor<int32_t> selectionKvBlockStatusGm;
    GlobalTensor<int16_t> selectionMembershipMapGm;
    GlobalTensor<int32_t> selectionKvActualSeqGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    LocalTensor<float> lseUb;

    TBuf<> commonTBuf; // Shared reusable space
    TBuf<> selectionPlanTBuf;
    TBuf<> rowMetadataTBuf;
    TBuf<> missPrefetchTBuf;
    TBuf<> layoutTBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2]; // Two entries allow ping-pong buffering
    TBuf<> stage0OutBuf[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVAttnOutId; // MTE3_V event ID for synchronization while Vector2 copies attentionOut
    TEventID vToMte3AttnOutId; // V_MTE3 event ID for synchronization while Vector2 copies attentionOut
    TEventID mte3ToVLseOutId; // MTE3_V event ID for synchronization while Vector1 copies LSE
    TEventID vToMte3LseOutId; // V_MTE3 event ID for synchronization while Vector1 copies LSE
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> dequantScaleBuff;
    TBuf<> lseBuf;

    TEventID mte2ToV;
    TEventID mte2ToSPlan;
    TEventID mte2ToMte3[2];
    TEventID vToMte2Invalid;
    TEventID vToSInvalid;
    TEventID sToVInvalid;
    TEventID mte3ToMte2[2];
    TEventID prefetchReadEvent;
    TEventID prefetchWriteEvent;
    TEventID layoutReadEvent[2];

    bool isSinks = false;
    T negativeFloatScalar;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t selectionKvBlockSize = 0;
    int64_t selectionMaxBlockNum = 0;
    int64_t selectionMembershipStride = 0;
    int64_t selectionDataRow = -1;
    int64_t selectionDirectRowStride = 0;
    int64_t selectionPlanOffset = 0;
    bool externalSelectionPlanActive = false;
    bool selectionWindowHasMiss = true;
    bool selectionChunkAllHit = false;
    int64_t cachedSelectionRow = -1;
    uint32_t selectionBodyMisses = 0;
    uint32_t gatherRows = 16;
    uint32_t layoutRows = 16;
    bool directKvL1 = false;
    bool selectionOrderActive = false;
    int64_t selectionChunkStart = 0;
    uint32_t prefetchedUntil = 0;
    uint32_t pendingPrefetchRows = 0;
    int64_t pendingPrefetchDestinations[4];
    uint32_t pendingPrefetchIndices[4];
    GlobalTensor<KV_T> prefetchedKvGm;
    uint32_t earlierMissSlots[388] = {};
    int32_t ownerBatch = -1;
    int32_t ownerQuery = -1;

    static constexpr int16_t SELECTION_EXTERNAL_PLAN_READY_MARKER = 0x5A45;
    static constexpr int16_t SELECTION_DIRECT_LAYOUT_MARKER = 0x5A44;
    static constexpr int32_t SELECTION_MEMBERSHIP_CONTROL_OFFSET = 16384;
    static constexpr int32_t SELECTION_MEMBERSHIP_STORAGE_COUNT = 16400;
    static constexpr int32_t SELECTION_PLAN_LOCAL_OFFSET = 16;
    static constexpr int32_t SELECTION_COMPACT_PLAN_LOCAL_OFFSET = 144;
    static constexpr int32_t ROW_METADATA_ELEMENTS = 2080;
    static constexpr int32_t ROW_METADATA_BYTES =
        ROW_METADATA_ELEMENTS * (sizeof(int32_t) + sizeof(int16_t));
    static_assert(ROW_METADATA_ELEMENTS >= 2051 && ROW_METADATA_BYTES % 32 == 0);

    int64_t sparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;
};

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
CopySparseAttentionVectorService<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx,
    int64_t s2IdxInBase, const RunInfo &runInfo, ConstInfo &constInfo)
{
    auto rowIds = rowMetadataTBuf.template Get<int32_t>();
    auto order = selectionPlanTBuf.template Get<uint16_t>()[256];
    int64_t tokenLimit = constInfo.sparseMode == 3 ?
        runInfo.actualS2Size - runInfo.actualS1Size + runInfo.s1oIdx + 1 : runInfo.actualS2Size;
    int64_t column0 = selectionOrderActive ? order.GetValue(s2IdxInBase) : selectionChunkStart + s2IdxInBase;
    int64_t column1 = selectionOrderActive && s2IdxInBase + 1 < runInfo.s2RealSize ?
        order.GetValue(s2IdxInBase + 1) : selectionChunkStart + s2IdxInBase + 1;
    token0Idx = s2IdxInBase < runInfo.s2RealSize ? rowIds.GetValue(column0) : -1;
    token1Idx = s2IdxInBase + 1 < runInfo.s2RealSize ? rowIds.GetValue(column1) : -1;
    token0Idx = token0Idx >= 0 && token0Idx < tokenLimit ? token0Idx + runInfo.s2StartIdx : -1;
    token1Idx = token1Idx >= 0 && token1Idx < tokenLimit ? token1Idx + runInfo.s2StartIdx : -1;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
int64_t CopySparseAttentionVectorService<TEMPLATE_ARGS>::GetkeyOffset(
    int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) *
            static_cast<int64_t>(blockSize) +
            blkTableOffset; // BlockNum, BlockSize, N(1), D
    } else {
        if constexpr (LAYOUT_T == CopySparseAttentionLayoutArch35::BSND) {
            realkeyOffset = (runInfo.boIdx * constInfo.s2Size + s2Idx); // BSN(1)D
        } else if constexpr (LAYOUT_T == CopySparseAttentionLayoutArch35::TND) {
            int64_t batchKvStart = (runInfo.boIdx == 0) ? 0 : actualSeqLengthsKVGm.GetValue(runInfo.boIdx - 1);
            realkeyOffset = (batchKvStart + s2Idx);
        }
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline int64_t
CopySparseAttentionVectorService<TEMPLATE_ARGS>::GetSelectionRow(
    const RunInfo &runInfo, const ConstInfo &constInfo)
{
    int64_t queryTokenOffset = 0;
    if constexpr (LAYOUT_T == CopySparseAttentionLayoutArch35::TND) {
        int64_t actualSeqQPrefixSum = runInfo.boIdx == 0 ?
            0 : cuSeqlensQGm.GetValue(runInfo.boIdx - 1);
        queryTokenOffset = actualSeqQPrefixSum + runInfo.s1oIdx;
    } else {
        queryTokenOffset = runInfo.boIdx * constInfo.s1Size + runInfo.s1oIdx;
    }
    return queryTokenOffset * constInfo.n2Size + runInfo.n2oIdx;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline int64_t
CopySparseAttentionVectorService<TEMPLATE_ARGS>::GetSelectionTokenOffset(
    int64_t selectionRow, int32_t selectionSlot)
{
    if (selectionRow < 0 || selectionSlot < 0) {
        return -1;
    }
    if (selectionDirectRowStride > 0) {
        if (selectionSlot >= selectionDirectRowStride) {
            return -1;
        }
        return selectionRow * selectionDirectRowStride + selectionSlot;
    }
    if (selectionKvBlockSize <= 0 || selectionMaxBlockNum <= 0) {
        return -1;
    }
    int64_t blockTableIndex = selectionSlot / selectionKvBlockSize;
    if (blockTableIndex >= selectionMaxBlockNum) {
        return -1;
    }
    int32_t blockNumber = selectionKvBlockTableGm.GetValue(
        selectionRow * selectionMaxBlockNum + blockTableIndex);
    if (blockNumber < 0) {
        return -1;
    }
    return static_cast<int64_t>(blockNumber) * selectionKvBlockSize +
        selectionSlot % selectionKvBlockSize;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
CopySparseAttentionVectorService<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
    int64_t keyOffset, ConstInfo &constInfo)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;

    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // Only COMBINE mode is currently supported.
    uint32_t combineBytes = 512 * sizeof(KV_T);
    intriParams.blockLen = combineBytes;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * 576], keyGm[keyOffset * 512], intriParams, padParams);

}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
CopySparseAttentionVectorService<TEMPLATE_ARGS>::CopyInSingleSelectionKv(
    LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t selectionTokenOffset,
    ConstInfo &constInfo)
{
    if (selectionTokenOffset < 0) {
        return;
    }
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    copyParams.blockLen = constInfo.dSizeNope * sizeof(KV_T);
    DataCopyPad(kvInUb[startRow * 576],
        selectionKvCacheGm[selectionTokenOffset * constInfo.dSizeNope],
        copyParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline bool
CopySparseAttentionVectorService<TEMPLATE_ARGS>::PrepareExternalSelectionPlan(
    const RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<int16_t> selectionPlanLocal,
    int64_t &logicalSelectionRow)
{
    logicalSelectionRow = GetSelectionRow(runInfo, constInfo);
    if (logicalSelectionRow < 0 || selectionMembershipStride < SELECTION_MEMBERSHIP_STORAGE_COUNT) return false;
    auto rowIds = rowMetadataTBuf.template Get<int32_t>();
    auto rowPlan = rowIds[ROW_METADATA_ELEMENTS].template ReinterpretCast<int16_t>();
    auto order = selectionPlanTBuf.template Get<uint16_t>()[256];
    int64_t membershipBase = logicalSelectionRow * selectionMembershipStride;
    if (cachedSelectionRow != logicalSelectionRow) {
        auto control = selectionPlanTBuf.template Get<int16_t>();
        DataCopyPad(control, selectionMembershipMapGm[membershipBase + 16384],
            DataCopyExtParams{1, 22, 0, 0, 0}, DataCopyPadExtParams<int16_t>{false, 0, 0, 0});
        DataCopyPad(rowPlan, selectionMembershipMapGm[membershipBase + 14333],
            DataCopyExtParams{1, 4102, 0, 0, 0}, DataCopyPadExtParams<int16_t>{false, 0, 0, 0});
        DataCopyPad(rowIds, sparseIndicesGm[logicalSelectionRow * 2051],
            DataCopyExtParams{1, 8204, 0, 0, 0}, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(mte2ToSPlan);
        WaitFlag<HardEvent::MTE2_S>(mte2ToSPlan);
        if (control.GetValue(1) != SELECTION_EXTERNAL_PLAN_READY_MARKER ||
            control.GetValue(5) != SELECTION_DIRECT_LAYOUT_MARKER || control.GetValue(2) != 2051) return false;
        selectionDataRow = control.GetValue(4);
        selectionDirectRowStride = control.GetValue(6);
        selectionBodyMisses = static_cast<uint16_t>(control.GetValue(10));
        prefetchedUntil = 0;
        cachedSelectionRow = logicalSelectionRow;
        bool needsOwners = selectionBodyMisses > 0;
        for (uint32_t column = 2048; column < 2051; ++column)
            needsOwners |= rowPlan.GetValue(column) < 0;
        if (needsOwners) PrepareMissOwners(runInfo.boIdx, logicalSelectionRow);
        selectionKvActualSeqGm.SetValue(logicalSelectionRow, 2051);
    }
    int64_t sourceStart = runInfo.s2LoopCount * 128;
    bool reordered = runInfo.s2LoopLimit >= 16 && selectionBodyMisses > 0 && selectionBodyMisses < 2048;
    selectionOrderActive = reordered && runInfo.s2LoopCount < 16;
    if (selectionOrderActive) {
        sourceStart = SglLimSchedule::GatherIndex(runInfo.boIdx, sourceStart);
        DataCopyPad(order.template ReinterpretCast<int16_t>(), selectionMembershipMapGm[membershipBase + 4096 + sourceStart],
            DataCopyExtParams{1, static_cast<uint32_t>(runInfo.s2RealSize * 2), 0, 0, 0},
            DataCopyPadExtParams<int16_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(mte2ToSPlan);
        WaitFlag<HardEvent::MTE2_S>(mte2ToSPlan);
    }
    selectionChunkStart = sourceStart;
    selectionChunkAllHit = true;
    if (selectionOrderActive) {
        for (int64_t index = 0; index < runInfo.s2RealSize; ++index) {
            int16_t value = rowPlan.GetValue(order.GetValue(index));
            selectionPlanLocal.SetValue(index, value);
            if (value < 0) selectionChunkAllHit = false;
        }
    } else {
        SetFlag<HardEvent::S_V>(sToVInvalid);
        WaitFlag<HardEvent::S_V>(sToVInvalid);
        DataCopy(selectionPlanLocal, rowPlan[sourceStart], Align16Func(runInfo.s2RealSize));
        SetFlag<HardEvent::V_S>(vToSInvalid);
        WaitFlag<HardEvent::V_S>(vToSInvalid);
        if (sourceStart < 2048) {
            selectionChunkAllHit = selectionBodyMisses == 0 ||
                (selectionBodyMisses == 2048 && selectionPlanLocal.GetValue(0) >= 0);
        } else {
            for (int64_t index = 0; index < runInfo.s2RealSize; ++index)
                if (selectionPlanLocal.GetValue(index) < 0) selectionChunkAllHit = false;
        }
    }
    externalSelectionPlanActive = true;
    return true;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline uint32_t
CopySparseAttentionVectorService<TEMPLATE_ARGS>::CopyInKvWithExternalPlan(
    LocalTensor<KV_T> kvInUb, LocalTensor<int16_t> compactPlanLocal, int64_t startRow,
    int64_t token0Idx, int64_t token1Idx, int16_t planValue0, int16_t planValue1,
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    uint32_t copiedRows = 0;
    if (token0Idx < 0 || token1Idx < 0) {
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);
        if (token0Idx < 0) {
            Duplicate(kvInUb[startRow * 576], static_cast<KV_T>(0), 576);
            planValue0 = 0;
        }
        if (token1Idx < 0) {
            Duplicate(kvInUb[(startRow + 1) * 576], static_cast<KV_T>(0), 576);
            planValue1 = 0;
        }
        SetFlag<HardEvent::V_MTE2>(vToMte2Invalid);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Invalid);
    }
    if (token0Idx >= 0) {
        int64_t selectionTokenOffset = planValue0 > 0 ? GetSelectionTokenOffset(
            selectionDataRow, static_cast<int32_t>(planValue0) - 1) : -1;
        if (selectionTokenOffset >= 0) {
            CopyInSingleSelectionKv(kvInUb, startRow, selectionTokenOffset, constInfo);
        } else {
            CopyInSingleKv(kvInUb, startRow, GetkeyOffset(token0Idx, runInfo, constInfo), constInfo);
            if (planValue0 > 0) {
                planValue0 = -planValue0;
            }
        }
    }
    compactPlanLocal.SetValue(startRow + copiedRows, planValue0);
    copiedRows++;
    if (token1Idx >= 0) {
        int64_t selectionTokenOffset = planValue1 > 0 ? GetSelectionTokenOffset(
            selectionDataRow, static_cast<int32_t>(planValue1) - 1) : -1;
        if (selectionTokenOffset >= 0) {
            CopyInSingleSelectionKv(kvInUb, startRow + copiedRows, selectionTokenOffset, constInfo);
        } else {
            CopyInSingleKv(kvInUb, startRow + copiedRows,
                GetkeyOffset(token1Idx, runInfo, constInfo), constInfo);
            if (planValue1 > 0) {
                planValue1 = -planValue1;
            }
        }
    }
    compactPlanLocal.SetValue(startRow + copiedRows, planValue1);
    copiedRows++;
    return copiedRows;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
CopySparseAttentionVectorService<TEMPLATE_ARGS>::CopyOutSelectionMisses(
    LocalTensor<KV_T> kvInUb, LocalTensor<int16_t> compactPlanLocal,
    int64_t dealRow, int64_t selectionRow, ConstInfo &constInfo)
{
    if (!externalSelectionPlanActive || selectionRow < 0 || !selectionWindowHasMiss) {
        return;
    }
    DataCopyExtParams kvParams;
    kvParams.blockCount = 1;
    kvParams.blockLen = constInfo.dSizeNope * sizeof(KV_T);
    kvParams.srcStride = 0;
    kvParams.dstStride = 0;
    for (int64_t row = 0; row < dealRow; ++row) {
        int16_t planValue = compactPlanLocal.GetValue(row);
        if (planValue >= 0) {
            continue;
        }
        int32_t destinationSlot = -static_cast<int32_t>(planValue) - 1;
        int64_t selectionTokenOffset = GetSelectionTokenOffset(selectionRow, destinationSlot);
        if (selectionTokenOffset < 0 || !OwnsMissSlot(destinationSlot)) {
            continue;
        }
        kvParams.blockCount = 1;
        kvParams.srcStride = 0;
        kvParams.dstStride = 0;
        if (row + 1 < dealRow) {
            int16_t nextPlan = compactPlanLocal.GetValue(row + 1);
            int32_t nextSlot = -static_cast<int32_t>(nextPlan) - 1;
            if (nextPlan < 0 && nextSlot > destinationSlot && OwnsMissSlot(nextSlot)) {
                kvParams.blockCount = 2;
                kvParams.srcStride = (576 - constInfo.dSizeNope) * sizeof(KV_T) / 32;
                kvParams.dstStride = (nextSlot - destinationSlot - 1) * constInfo.dSizeNope * sizeof(KV_T);
            }
        }
        DataCopyPad(selectionKvCacheGm[selectionTokenOffset * constInfo.dSizeNope],
            kvInUb[row * 576], kvParams);
        row += kvParams.blockCount - 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
uint32_t CopySparseAttentionVectorService<TEMPLATE_ARGS>::CopyInKvSparse(
    LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx,
    int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t keyOffset0 = GetkeyOffset(token0Idx, runInfo, constInfo);
    int64_t keyOffset1 = GetkeyOffset(token1Idx, runInfo, constInfo);
    if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
        return 0;
    }
    int64_t blkTableSrcStride =
        ((keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) :
        (keyOffset1 - keyOffset0)) - constInfo.sparseBlockSize);
    int64_t keySrcStride = blkTableSrcStride * constInfo.dSizeNope * sizeof(KV_T);
    if (unlikely(keyOffset1 < 0)) {
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
    } else if (keySrcStride >= INT32_MAX || keySrcStride < 0 || constInfo.sparseBlockSize > 1) {
        // For exceptional cases such as stride overflow, negative stride, or excessive S2 length, restore two copy instructions.
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
        CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, constInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.dSizeNope *sizeof(KV_T);
        intriParams.dstStride = (576 - constInfo.dSizeNope) * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
        if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
            keyOffset = keyOffset1;
        }
        DataCopyPad(kvInUb[startRow * 576], keyGm[keyOffset * constInfo.dSizeNope],
                    intriParams, padParams); // combineDimAlign

    }
    return (keyOffset0 > -1) + (keyOffset1 > -1);
}

// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_2 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_3 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_4 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename Q_T, typename KV_T>
__simd_vf__ void CastScaleImpl(__ubuf__ float* ubDstAddr, __ubuf__ int8_t* ubSrcAddr, uint32_t dealRowCount)
{
    MicroAPI::RegTensor<fp8_e8m0_t> vScale0;
    MicroAPI::RegTensor<fp8_e8m0_t> vScale1;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res0;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res1;
    MicroAPI::RegTensor<float> vScalefp32Res0;
    MicroAPI::RegTensor<float> vScalefp32Res1;
    __ubuf__ int8_t* ubScaleSrcAddrTemp = ubSrcAddr;
    __ubuf__ float* ubDstAddrTmp = ubDstAddr;
    MicroAPI::MaskReg bf16TypeMaskAll = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        // load scale
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t>&)vScale0, ubScaleSrcAddrTemp, 640);

        MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8_1>(vScalebf16Res0, vScale0, bf16TypeMaskAll);
        MicroAPI::Cast<float, bfloat16_t, castTraitFp8_1>(vScalefp32Res0, vScalebf16Res0, fp32MaskAll);

        MicroAPI::StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vScalefp32Res0, 64, bf16TypeMaskAll);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CopySparseAttentionVectorService<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> kvOutUb,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(v0ResGmTensor[s2StartIdx * 576], kvOutUb, dealRow * 576);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CopySparseAttentionVectorService<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = constInfo.aivIdx >> 1U;
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        uint32_t v0S2SizeSecondCore = runInfo.s2RealSize - v0S2SizeFirstCore;
        if (aicIdx % 2U == 0) {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = 0;
            } else {
                sparseCalSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = CeilDiv(v0S2SizeFirstCore, 2);
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeSecondCore, 2);
                sparseS2Start = v0S2SizeFirstCore;
            } else {
                sparseCalSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2);
                sparseS2Start = v0S2SizeFirstCore + CeilDiv(v0S2SizeSecondCore, 2);
            }
        }
        sparseS2End = sparseS2Start + sparseCalSize;
    } else {
        int64_t s2PerVecLoop = 2LL;
        int64_t vecNum = 2LL;
        int64_t s2Loops = CeilDiv(CeilDiv(runInfo.s2RealSize, vecNum), s2PerVecLoop);
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize);
        sparseS2End = GetSubBlockIdx() == 0 ? Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize) : runInfo.s2RealSize;
        sparseCalSize = sparseS2End - sparseS2Start;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void CopySparseAttentionVectorService<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos)
{
    blockSize = constInfo.oriBlockSize;
    maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;

    CalSparseCalSize(runInfo, constInfo);
    ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo, startPos);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
CopySparseAttentionVectorService<TEMPLATE_ARGS>::SignalKvReady(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm)
{
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<0, PIPE_MTE3>(15);
        CrossCoreWaitFlag<0, PIPE_MTE3>(15);
    }
    outputL1.SetCrossCore();
    if (IS_SPLIT_G || !directKvL1) v0ResGm.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void CopySparseAttentionVectorService<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos)
{
    CompleteMissPrefetch();
    auto plan = selectionPlanTBuf.template Get<int16_t>()[SELECTION_PLAN_LOCAL_OFFSET];
    int64_t logicalRow = -1;
    PrepareExternalSelectionPlan(runInfo, constInfo, plan, logicalRow);
    if (sparseCalSize == 0) {
        SignalKvReady(outputL1, v0ResGm);
        return;
    }
    int64_t source = sparseS2Start;
    int64_t pendingStart = 0;
    int64_t pendingRows = 0;
    bool pendingMiss = false;
    uint32_t pendingBank = 0;
    uint32_t nextBank = 0;
    while (source < sparseS2End) {
        int64_t begin = source;
        auto compact = selectionPlanTBuf.template Get<int16_t>()[SELECTION_COMPACT_PLAN_LOCAL_OFFSET + nextBank * 32];
        int64_t rows = CopySparseKvBatch(v0ResGm, plan, compact, source, nextBank, runInfo, constInfo);
        bool hasMiss = selectionWindowHasMiss;
        if (pendingRows > 0) {
            auto data = stage0OutBuf[pendingBank].template Get<Q_T>();
            auto pendingPlan = selectionPlanTBuf.template Get<int16_t>()[SELECTION_COMPACT_PLAN_LOCAL_OFFSET + pendingBank * 32];
            WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pendingBank]);
            if (IS_SPLIT_G || !directKvL1)
                CopyOutKvUb2Gm(v0ResGm, data, pendingRows, pendingStart, runInfo, constInfo);
            else {
                WaitFlag<HardEvent::MTE2_V>(layoutReadEvent[pendingBank]);
                CopyBf16Nope512UbToL1(outputL1.template GetTensor<Q_T>(), data,
                    layoutTBuf.template Get<Q_T>(), pendingRows, pendingStart, runInfo.s2RealSize, layoutRows);
            }
            selectionWindowHasMiss = pendingMiss;
            CopyOutSelectionMisses(data, pendingPlan, pendingRows, selectionDataRow, constInfo);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pendingBank]);
        }
        pendingStart = begin;
        pendingRows = rows;
        pendingMiss = hasMiss;
        pendingBank = nextBank;
        nextBank ^= 1;
    }
    auto data = stage0OutBuf[pendingBank].template Get<Q_T>();
    auto pendingPlan = selectionPlanTBuf.template Get<int16_t>()[SELECTION_COMPACT_PLAN_LOCAL_OFFSET + pendingBank * 32];
    WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pendingBank]);
    if (IS_SPLIT_G || !directKvL1)
        CopyOutKvUb2Gm(v0ResGm, data, pendingRows, pendingStart, runInfo, constInfo);
    else {
        WaitFlag<HardEvent::MTE2_V>(layoutReadEvent[pendingBank]);
        CopyBf16Nope512UbToL1(outputL1.template GetTensor<Q_T>(), data,
            layoutTBuf.template Get<Q_T>(), pendingRows, pendingStart, runInfo.s2RealSize, layoutRows);
    }
    if constexpr (!IS_SPLIT_G) SignalKvReady(outputL1, v0ResGm);
    selectionWindowHasMiss = pendingMiss;
    CopyOutSelectionMisses(data, pendingPlan, pendingRows, selectionDataRow, constInfo);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pendingBank]);
    if constexpr (IS_SPLIT_G) SignalKvReady(outputL1, v0ResGm);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline int64_t
CopySparseAttentionVectorService<TEMPLATE_ARGS>::CopySparseKvBatch(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    LocalTensor<int16_t> selectionPlanLocal, LocalTensor<int16_t> compactPlanLocal,
    int64_t &s2, uint32_t pingPong, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t dealRow = 0;
    selectionWindowHasMiss = true;
    LocalTensor<Q_T> stage0OutUb = this->stage0OutBuf[pingPong].template Get<Q_T>();
    WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
    int64_t windowSize = Min(static_cast<int64_t>(gatherRows), sparseS2End - s2);
    bool copied = false;
    if (externalSelectionPlanActive && selectionDirectRowStride > 0) {
        if (selectionChunkAllHit)
            copied = CopyPlannedKvWindow<true>(stage0OutUb, selectionPlanLocal, compactPlanLocal,
                s2, windowSize, runInfo, constInfo);
        else
            copied = CopyPlannedKvWindow<false>(stage0OutUb, selectionPlanLocal, compactPlanLocal,
                s2, windowSize, runInfo, constInfo);
    }
    if (copied) {
        dealRow = windowSize;
        s2 += windowSize;
    }
    while (dealRow < gatherRows && s2 < sparseS2End) {
        int64_t token0Idx;
        int64_t token1Idx;
        GetRealCmpS2Idx(token0Idx, token1Idx, s2, runInfo, constInfo);
        int16_t planValue0 = externalSelectionPlanActive ? selectionPlanLocal.GetValue(s2) : 0;
        int16_t planValue1 = externalSelectionPlanActive && s2 + 1 < runInfo.s2RealSize ?
            selectionPlanLocal.GetValue(s2 + 1) : 0;
        s2 += 2;
        dealRow += CopyInKvWithExternalPlan(stage0OutUb, compactPlanLocal, dealRow,
            token0Idx, token1Idx, planValue0, planValue1, runInfo, constInfo);
        if (s2 > sparseS2End) {
            --dealRow;
        }
    }
    SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
    if (!IS_SPLIT_G && directKvL1) SetFlag<HardEvent::MTE2_V>(layoutReadEvent[pingPong]);
    return dealRow;
}

TEMPLATES_DEF_NO_DEFAULT
template <bool HIT_ONLY>
__aicore__ inline bool
CopySparseAttentionVectorService<TEMPLATE_ARGS>::CopyPlannedKvWindow(
    LocalTensor<KV_T> destination, LocalTensor<int16_t> selectionPlanLocal,
    LocalTensor<int16_t> compactPlanLocal, int64_t start, int64_t count,
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    int16_t plans[32];
    for (int64_t row = 0; row < count; ++row) {
        plans[row] = selectionPlanLocal.GetValue(start + row);
        if (plans[row] == 0 || plans[row] > selectionDirectRowStride ||
            plans[row] < -selectionDirectRowStride) {
            return false;
        }
    }
    DataCopyExtParams copyParams;
    copyParams.blockLen = constInfo.dSizeNope * sizeof(KV_T);
    copyParams.dstStride = (576 - constInfo.dSizeNope) * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
    DataCopyPadExtParams<KV_T> padParams{false, 0, 0, 0};
    bool hasMiss = false;
    for (int64_t row = 0; row < count;) {
        bool fromSelection = HIT_ONLY || plans[row] > 0;
        uint32_t orderedIndex = selectionChunkStart + start + row;
        if (!fromSelection && selectionOrderActive && orderedIndex < prefetchedUntil &&
            (orderedIndex % 128) / 64 == constInfo.subBlockIdx) {
            DataCopy(destination[row * 576],
                prefetchedKvGm[(cachedSelectionRow * 512 + orderedIndex) * 512], 512);
            compactPlanLocal.SetValue(row, -plans[row]);
            ++row;
            continue;
        }
        int64_t runLength = 1;
        int64_t offset0;
        int64_t offset1 = -1;
        if (fromSelection) {
            offset0 = GetSelectionTokenOffset(selectionDataRow, plans[row] - 1);
            while (row + runLength < count && plans[row + runLength] == plans[row] + runLength) {
                ++runLength;
            }
            if (offset0 < 0 || plans[row] + runLength - 1 > selectionDirectRowStride) {
                return false;
            }
            if (runLength == 1 && row + 1 < count && plans[row + 1] > 0) {
                offset1 = GetSelectionTokenOffset(selectionDataRow, plans[row + 1] - 1);
            }
        } else {
            hasMiss = true;
            int64_t token0;
            int64_t token1;
            GetRealCmpS2Idx(token0, token1, start + row, runInfo, constInfo);
            if (token0 < 0) {
                return false;
            }
            offset0 = GetkeyOffset(token0, runInfo, constInfo);
            if (row + 1 < count && plans[row + 1] <= 0 && token1 >= 0) {
                offset1 = GetkeyOffset(token1, runInfo, constInfo);
            }
            if (offset0 < 0) {
                return false;
            }
        }
        copyParams.srcStride = 0;
        if (offset1 >= 0) {
            int64_t distance = offset1 - offset0;
            int64_t stride = ((distance < 0 ? -distance : distance) - 1) *
                constInfo.dSizeNope * sizeof(KV_T);
            if (stride >= 0 && stride < INT32_MAX) {
                if (distance < 0) {
                    offset0 = offset1;
                    int16_t savedPlan = plans[row];
                    plans[row] = plans[row + 1];
                    plans[row + 1] = savedPlan;
                }
                copyParams.srcStride = stride;
                runLength = 2;
            }
        }
        copyParams.blockCount = runLength;
        if (fromSelection) {
            DataCopyPad(destination[row * 576], selectionKvCacheGm[offset0 * constInfo.dSizeNope],
                copyParams, padParams);
        } else {
            DataCopyPad(destination[row * 576], keyGm[offset0 * constInfo.dSizeNope],
                copyParams, padParams);
        }
        for (int64_t offset = 0; offset < runLength; ++offset) {
            compactPlanLocal.SetValue(row + offset, plans[row + offset]);
        }
        row += runLength;
    }
    selectionWindowHasMiss = hasMiss;
    return true;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
CopySparseAttentionVectorService<TEMPLATE_ARGS>::PrepareMissOwners(uint32_t batch, uint32_t query)
{
    int32_t begin = batch == 0 ? 0 : cuSeqlensQGm.GetValue(batch - 1);
    if (ownerBatch != batch || ownerQuery < begin || ownerQuery > query) {
        for (uint32_t word = 0; word < 388; ++word) earlierMissSlots[word] = 0;
        ownerQuery = begin;
    }
    for (int32_t previous = ownerQuery; previous < query; ++previous) {
        uint64_t base = static_cast<uint64_t>(previous) * selectionMembershipStride;
        uint32_t misses = static_cast<uint16_t>(selectionMembershipMapGm.GetValue(base + 16394));
        for (uint32_t index = 0; index < misses + 3; ++index) {
            uint32_t column = index < misses ?
                (misses == 2048 ? index : static_cast<uint16_t>(selectionMembershipMapGm.GetValue(base + 4096 + index))) :
                2048 + index - misses;
            int16_t plan = selectionMembershipMapGm.GetValue(base + 14333 + column);
            if (plan >= 0) continue;
            uint32_t slot = -static_cast<int32_t>(plan) - 1;
            if (slot < selectionDirectRowStride) earlierMissSlots[slot / 32] |= 1U << (slot % 32);
        }
    }
    ownerBatch = batch;
    ownerQuery = query;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
CopySparseAttentionVectorService<TEMPLATE_ARGS>::SubmitMissPrefetch(const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (!externalSelectionPlanActive || selectionBodyMisses == 0 || selectionBodyMisses > 512 ||
        runInfo.s2LoopCount / 4 >= 3 - runInfo.boIdx % 3) return;
    auto rowIds = rowMetadataTBuf.template Get<int32_t>();
    auto rowPlan = rowIds[ROW_METADATA_ELEMENTS].template ReinterpretCast<int16_t>();
    auto order = missPrefetchTBuf.template Get<int16_t>();
    uint32_t count = selectionBodyMisses - prefetchedUntil;
    if (count == 0) return;
    if (count > 128) count = 128;
    DataCopyPad(order, selectionMembershipMapGm[cachedSelectionRow * selectionMembershipStride + 4096 + prefetchedUntil],
        DataCopyExtParams{1, count * 2, 0, 0, 0}, DataCopyPadExtParams<int16_t>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(mte2ToSPlan);
    WaitFlag<HardEvent::MTE2_S>(mte2ToSPlan);
    int32_t columns[4];
    uint32_t scan = 0;
    while (scan < count && pendingPrefetchRows < 4) {
        uint32_t index = prefetchedUntil + scan;
        uint32_t column = static_cast<uint16_t>(order.GetValue(scan++));
        if ((index % 128) / 64 != constInfo.subBlockIdx || rowPlan.GetValue(column) >= 0) continue;
        pendingPrefetchIndices[pendingPrefetchRows] = index;
        columns[pendingPrefetchRows++] = column;
    }
    prefetchedUntil += scan;
    auto incoming = missPrefetchTBuf.template Get<KV_T>();
    for (uint32_t index = 0; index < pendingPrefetchRows; ++index) {
        uint32_t column = columns[index];
        int32_t source = rowIds.GetValue(column);
        int32_t slot = -static_cast<int32_t>(rowPlan.GetValue(column)) - 1;
        pendingPrefetchDestinations[index] = OwnsMissSlot(slot) ?
            GetSelectionTokenOffset(selectionDataRow, slot) * 512 : -1;
        int64_t address = GetkeyOffset(source, runInfo, constInfo);
        DataCopy(incoming[index * 512], keyGm[address * 512], 512);
    }
    if (pendingPrefetchRows) SetFlag<HardEvent::MTE2_MTE3>(prefetchReadEvent);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
CopySparseAttentionVectorService<TEMPLATE_ARGS>::CompleteMissPrefetch()
{
    if (pendingPrefetchRows == 0) return;
    WaitFlag<HardEvent::MTE2_MTE3>(prefetchReadEvent);
    auto incoming = missPrefetchTBuf.template Get<KV_T>();
    for (uint32_t index = 0; index < pendingPrefetchRows; ++index) {
        DataCopy(prefetchedKvGm[(cachedSelectionRow * 512 + pendingPrefetchIndices[index]) * 512],
            incoming[index * 512], 512);
        if (pendingPrefetchDestinations[index] >= 0)
            DataCopy(selectionKvCacheGm[pendingPrefetchDestinations[index]], incoming[index * 512], 512);
    }
    SetFlag<HardEvent::MTE3_MTE2>(prefetchWriteEvent);
    WaitFlag<HardEvent::MTE3_MTE2>(prefetchWriteEvent);
    pendingPrefetchRows = 0;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void CopySparseAttentionVectorService<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
    SetFlag<HardEvent::V_S>(vToSInvalid);
    WaitFlag<HardEvent::V_S>(vToSInvalid);

    int64_t queryRow = runInfo.s1oIdx;
    if constexpr (LAYOUT_T == CopySparseAttentionLayoutArch35::TND) {
        queryRow += runInfo.boIdx == 0 ? 0 : cuSeqlensQGm.GetValue(runInfo.boIdx - 1);
    } else {
        queryRow += runInfo.boIdx * constInfo.s1Size;
    }
    int64_t indexBase = queryRow * constInfo.sparseBlockCount + runInfo.s2LoopCount * constInfo.s2BaseSize;
    int64_t tokenLimit = constInfo.sparseMode == 3 ?
        runInfo.actualS2Size - runInfo.actualS1Size + runInfo.s1oIdx + 1 : runInfo.actualS2Size;
    for (int64_t column = 0; column < runInfo.s2RealSize; ++column) {
        int32_t token = sparseIndicesGm.GetValue(indexBase + column);
        if (token < 0 || token >= tokenLimit) {
            for (int64_t row = 0; row < runInfo.halfMRealSize; ++row) {
                mmRes.SetValue(row * s2BaseSize + column, negativeFloatScalar);
            }
        }
    }
    SetFlag<HardEvent::S_V>(sToVInvalid);
    WaitFlag<HardEvent::S_V>(sToVInvalid);

    // When loopCount == 0 but sinks is provided, take the update branch, initialize maxUb from sinks,
    // and initialize sumUb to 1.0.
    if (runInfo.s2LoopCount == 0) { // The sink omits the first token, so add first-token information with N1 dimensions
        if (likely(runInfo.s2RealSize == 128)) { // Bucket for s2RealSize == 128; constant folding in VF reduces branches
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // Bucket for s2RealSize <= 64; constant folding in VF reduces branches
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize,
                runInfo.s2RealSize, // Actual number of valid elements to compute
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // Bucket for s2RealSize < 128; constant folding in VF reduces branches
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (likely(runInfo.s2RealSize == 128)) { // Bucket for s2RealSize == 128; constant folding in VF reduces branches
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // Bucket for s2RealSize <= 64; constant folding in VF reduces branches
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128) { // Bucket for s2RealSize < 128; constant folding in VF reduces branches
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();

    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>(s2BaseSize * constInfo.dSizeV);
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * \
            (BLOCK_BYTE / sizeof(Q_T)) * (runInfo.mRealSize - runInfo.halfMRealSize)],
            stage1CastTensor, {s2BaseSize / 16, (uint16_t)runInfo.halfMRealSize,
            (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
            (uint16_t)(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    if (runInfo.s2LoopCount != 0) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
    if (constInfo.returnSoftmaxLse && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        CopyFALseToGm(runInfo, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void CopySparseAttentionVectorService<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }

    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();

    WaitFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, expUb, expUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CopySparseAttentionVectorService<TEMPLATE_ARGS>::Bmm2DataCopyOut (RunInfo &runInfo, ConstInfo &constInfo,
    LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;

    attenOut.SetAddr(vec2ResUb.address_);
    Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3AttnOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3AttnOutId);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4; // Offset in 32-byte units; for BF16 this is 16 elements, hence shift right by 4
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CopySparseAttentionVectorService<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void CopySparseAttentionVectorService<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;
    // n2 = 1, n1 = gn2 = gSize
    if constexpr (LAYOUT_T == CopySparseAttentionLayoutArch35::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if constexpr (LAYOUT_T == CopySparseAttentionLayoutArch35::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }

    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (constInfo.aivIdx * singleCoreSize < totalOutputSize && singleInitOutputSize > 0) {
            matmul::InitOutput<OUTPUT_T>(
                this->attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
        }
    }

    if (constInfo.returnSoftmaxLse) {
        uint64_t totalReturnSoftmaxSize = 0;
        if constexpr (LAYOUT_T == CopySparseAttentionLayoutArch35::BSND) {
            totalReturnSoftmaxSize = constInfo.bSize *  constInfo.n2Size * constInfo.s1Size * constInfo.gSize;
        } else if constexpr (LAYOUT_T == CopySparseAttentionLayoutArch35::TND) {
            totalReturnSoftmaxSize = constInfo.n2Size * constInfo.s1Size * constInfo.gSize; // (N2,T1,G)
        }
        if (coreNum != 0 && totalReturnSoftmaxSize > 0) {
            uint64_t singleCoreSoftmaxSize = (totalReturnSoftmaxSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
            uint64_t tailSoftmaxSize = totalReturnSoftmaxSize - constInfo.aivIdx * singleCoreSoftmaxSize;
            uint64_t singleInitSoftmaxSize = tailSoftmaxSize < singleCoreSoftmaxSize ?
                                             tailSoftmaxSize : singleCoreSoftmaxSize;
            if (constInfo.aivIdx * singleCoreSoftmaxSize < totalReturnSoftmaxSize && singleInitSoftmaxSize > 0) {
                matmul::InitOutput<float>(this->softmaxSumGm[constInfo.aivIdx * singleCoreSoftmaxSize], singleInitSoftmaxSize, 0);
                matmul::InitOutput<float>(this->softmaxMaxGm[constInfo.aivIdx * singleCoreSoftmaxSize], singleInitSoftmaxSize, 0);
            }
        }
    }
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void CopySparseAttentionVectorService<TEMPLATE_ARGS>::CleanOutput(
    __gm__ uint8_t *attentionOut, ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo.needInit == 1) {
            InitOutputSingleCore(constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void CopySparseAttentionVectorService<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                                       __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
                                                       __gm__ uint8_t *blockTable,
                                                       __gm__ uint8_t *selectionKRope,
                                                       __gm__ uint8_t *selectionKvCache,
                                                       __gm__ uint8_t *selectionKvBlockTable,
                                                       __gm__ uint8_t *selectionKvBlockStatus,
                                                       __gm__ uint8_t *selectionMembershipMap,
                                                       __gm__ uint8_t *selectionKvActualSeq)
{
    keyGm.SetGlobalBuffer((__gm__ KV_T *)(key));
    if constexpr (isPa) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);;
    }
    sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    keyRopeGm.SetGlobalBuffer((__gm__ KV_T *)(keyRope));
    selectionKRopeGm.SetGlobalBuffer((__gm__ KV_T *)selectionKRope);
    selectionKvCacheGm.SetGlobalBuffer((__gm__ KV_T *)selectionKvCache);
    selectionKvBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)selectionKvBlockTable);
    selectionKvBlockStatusGm.SetGlobalBuffer((__gm__ int32_t *)selectionKvBlockStatus);
    selectionMembershipMapGm.SetGlobalBuffer((__gm__ int16_t *)selectionMembershipMap);
    selectionKvActualSeqGm.SetGlobalBuffer((__gm__ int32_t *)selectionKvActualSeq);
    prefetchedKvGm.SetGlobalBuffer((__gm__ KV_T *)(selectionKvActualSeq +
        (static_cast<uint64_t>(tilingData->baseParams.qSeqSize) * 4 + 31) / 32 * 32));
    selectionKvBlockSize = tilingData->baseParams.blockSize;
    selectionMaxBlockNum = tilingData->baseParams.selectionBlockTableStride;
    selectionMembershipStride = tilingData->baseParams.selectionMembershipStride;
    directKvL1 = tilingData->baseParams.directKvL1 != 0;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void CopySparseAttentionVectorService<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t softmaxBufSize = 256; // VF processes 256 bytes per operation
    tPipe->InitBuffer(softmaxSumBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], softmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void CopySparseAttentionVectorService<TEMPLATE_ARGS>::CopyFALseToGm(RunInfo &runInfo, ConstInfo &constInfo)
{
    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();

    size_t alignedSize = (sizeof(float) * runInfo.halfMRealSize + 31) / 32 * 32 / sizeof(float);

    int64_t lseOffset = runInfo.softmaxLseOffset;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = sizeof(float) * runInfo.halfMRealSize;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    // Copy softmaxMaxUb to GM.
    WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    DataCopy(lseUb, maxUb, alignedSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    DataCopyPad(this->softmaxMaxGm[lseOffset], lseUb, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);

    // Copy softmaxSumUb to GM.
    WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    DataCopy(lseUb, sumUb, alignedSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    DataCopyPad(this->softmaxSumGm[lseOffset], lseUb, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void CopySparseAttentionVectorService<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // Allocate 512 bytes for commonTBuf
    tPipe->InitBuffer(selectionPlanTBuf, 1024);
    tPipe->InitBuffer(rowMetadataTBuf, ROW_METADATA_BYTES);
    tPipe->InitBuffer(missPrefetchTBuf, 4 * 512 * sizeof(KV_T));
    layoutRows = constInfo.gSize <= 16 ? 32 : 16;
    if (!IS_SPLIT_G && directKvL1) tPipe->InitBuffer(layoutTBuf, (layoutRows + 1) * 512 * sizeof(Q_T));
    tPipe->InitBuffer(lseBuf, 512); // Allocate 512 bytes for lseBuf
    lseUb = this->lseBuf.template Get<float>();

    gatherRows = constInfo.gSize <= 32 ? 32 : 16;
    tPipe->InitBuffer(stage0OutBuf[0], 576 * gatherRows * sizeof(KV_T));
    tPipe->InitBuffer(stage0OutBuf[1], 576 * gatherRows * sizeof(KV_T));

    tPipe->InitBuffer(stage1OutQue[0], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutQue[1], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, CeilDiv(constInfo.gSize, CV_RATIO) * dTemplateAlign64 * sizeof(T));

    mte3ToVAttnOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVLseOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);

    vToMte3AttnOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3LseOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();

    mte2ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    mte2ToSPlan = GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>();
    mte3ToMte2[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    mte3ToMte2[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[0]);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[1]);
    mte2ToMte3[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    mte2ToMte3[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    prefetchReadEvent = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    prefetchWriteEvent = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    if (!IS_SPLIT_G && directKvL1) {
        layoutReadEvent[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        layoutReadEvent[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    }
    vToMte2Invalid = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    vToSInvalid = GetTPipePtr()->AllocEventID<HardEvent::V_S>();
    sToVInvalid = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void CopySparseAttentionVectorService<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams &sharedParams, uint8_t subBlockIdx)
{
    auto &sparseAttnSharedkvBaseParams = this->tilingData->baseParams;
    sharedParams.bSize = sparseAttnSharedkvBaseParams.batchSize;
    sharedParams.n2Size = 1;
    sharedParams.gSize = sparseAttnSharedkvBaseParams.nNumOfQInOneGroup;
    sharedParams.s1Size = sparseAttnSharedkvBaseParams.qSeqSize;
    sharedParams.s2Size = sparseAttnSharedkvBaseParams.seqSize;
    sharedParams.sparseBlockCount = sparseAttnSharedkvBaseParams.sparseBlockCount;
    sharedParams.cmpRatio = 1; // Use the sparse path without compression
    sharedParams.oriMaskMode = sparseAttnSharedkvBaseParams.sparseMode;
    sharedParams.oriWinLeft = -1;
    sharedParams.oriWinRight = 0;
    sharedParams.layoutType = sparseAttnSharedkvBaseParams.outputLayout;
    sharedParams.dSizeRope = 0;
    sharedParams.softmaxScale = sparseAttnSharedkvBaseParams.scaleValue;
    sharedParams.dSize = 512;
    sharedParams.dSizeVInput = 512;
    sharedParams.usedCoreNum = this->tilingData->singleCoreParams.usedCoreNum;

    // Used when the Cube side copies RoPE for page attention.
    if constexpr (isPa) {
        sharedParams.oriBlockSize = sparseAttnSharedkvBaseParams.blockSize;
        sharedParams.oriMaxBlockNumPerBatch = sparseAttnSharedkvBaseParams.maxBlockNumPerBatch;
    }

    // actQ uses TND; actKV supports any layout in page-attention scenarios.
    sharedParams.isActualSeqLengthsNull = sparseAttnSharedkvBaseParams.isActualLenDimsNull;
    sharedParams.isActualSeqLengthsKVNull = sparseAttnSharedkvBaseParams.isActualLenDimsKVNull;
    sharedParams.returnSoftmaxLse = sparseAttnSharedkvBaseParams.returnSoftmaxLse;
    sharedParams.needInit = 0;
    for (uint32_t bIdx = 0; bIdx < sharedParams.bSize; bIdx++) {
        int64_t s2Size;
        if constexpr (KV_LAYOUT_T == CopySparseAttentionLayoutArch35::TND) {
            s2Size = bIdx == 0 ? actualSeqLengthsKVGm.GetValue(bIdx) : \
            actualSeqLengthsKVGm.GetValue(bIdx) - actualSeqLengthsKVGm.GetValue(bIdx - 1);
        } else {
            if (sharedParams.isActualSeqLengthsKVNull) {
                s2Size = sharedParams.s2Size;
            } else {
                s2Size = actualSeqLengthsKVGm.GetValue(bIdx);
            }
        }
        int64_t s1Size;
        if constexpr (LAYOUT_T == CopySparseAttentionLayoutArch35::TND) {
            s1Size = bIdx == 0 ? cuSeqlensQGm.GetValue(bIdx) : \
            cuSeqlensQGm.GetValue(bIdx) - cuSeqlensQGm.GetValue(bIdx - 1);
        } else {
            if (sharedParams.isActualSeqLengthsNull) {
                s1Size = sharedParams.s1Size;
            } else {
                s1Size = cuSeqlensQGm.GetValue(bIdx);
            }
        }
        if (s1Size > s2Size || (LAYOUT_T == CopySparseAttentionLayoutArch35::BSND && s1Size < sharedParams.s1Size)) {
            sharedParams.needInit = 1;
            break;
        }
    }

    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // Copy starting at SSBUF address 0
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
#pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void CopySparseAttentionVectorService<TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float *)&tmp1);
}


TEMPLATES_DEF class CopySparseAttentionVectorServiceDummy {
public:
    __aicore__ inline CopySparseAttentionVectorServiceDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                            __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *selectionKRope,
                                            __gm__ uint8_t *selectionKvCache,
                                            __gm__ uint8_t *selectionKvBlockTable,
                                            __gm__ uint8_t *selectionKvBlockStatus,
                                            __gm__ uint8_t *selectionMembershipMap,
                                            __gm__ uint8_t *selectionKvActualSeq) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const CopySparseAttentionTilingDataMla *__restrict tiling,
        CVSharedParams &sharedParams, uint8_t subBlockIdx, __gm__ uint8_t *actualSeqLengthsQ,
        __gm__ uint8_t *actualSeqLengths) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}
};
}
#endif // COPY_SPARSE_ATTENTION_SERVICE_VECTOR_MLA_ARCH35_H
