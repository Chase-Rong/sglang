#pragma once

#include <stdint.h>

struct CopySparseAttentionBaseParamsMla {
    uint32_t batchSize;
    uint32_t seqSize;
    uint32_t qSeqSize;
    int64_t blockSize;
    uint32_t maxBlockNumPerBatch;
    float scaleValue;
    uint32_t nNumOfQInOneGroup;
    uint32_t actualLenDimsQ;
    uint32_t actualLenDimsKV;
    uint32_t outputLayout;
    uint32_t sparseMode;
    int64_t preTokens;
    int64_t nextTokens;
    uint32_t attentionMode;
    uint32_t returnSoftmaxLse;
    int64_t sparseBlockSize;
    uint32_t sparseBlockCount;
    uint32_t selectionBlockTableStride;
    uint32_t selectionStatusStride;
    uint32_t selectionMembershipStride;
    uint32_t isActualLenDimsNull;
    uint32_t isActualLenDimsKVNull;
};

struct CopySparseAttentionSingleCoreParamsMla {
    uint32_t usedCoreNum;
};

struct CopySparseAttentionSingleCoreTensorSizeMla {
    uint32_t mmResUbSize;
    uint32_t bmm2ResUbSize;
};

struct CopySparseAttentionSplitKVParamsMla {
    uint32_t s2;
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
};

struct CopySparseAttentionInnerSplitParams {
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
};

struct CopySparseAttentionTilingDataMla {
    CopySparseAttentionBaseParamsMla baseParams;
    CopySparseAttentionSplitKVParamsMla splitKVParams;
    CopySparseAttentionSingleCoreParamsMla singleCoreParams;
    CopySparseAttentionSingleCoreTensorSizeMla singleCoreTensorSize;
    CopySparseAttentionInnerSplitParams innerSplitParams;
};
