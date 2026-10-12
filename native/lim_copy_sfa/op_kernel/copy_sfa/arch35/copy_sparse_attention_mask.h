#pragma once

#include "kernel_operator.h"

namespace CopySfaBase {

__simd_vf__ inline void PrepareSparseTaskMask(
    __ubuf__ float *output, __ubuf__ int32_t *tokens,
    __ubuf__ uint16_t *order, __ubuf__ int16_t *rowPlan, __ubuf__ int16_t *chunkPlan,
    uint32_t sourceStart, uint32_t columns,
    int32_t tokenLimit, bool reordered, float negative)
{
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto all = CreateMask<int32_t, MaskPattern::ALL>();
    RegTensor<int32_t> selected;
    RegTensor<uint16_t> indices;
    RegTensor<float> zeros, invalid, maskValues;
    Duplicate(zeros, 0.0F);
    Duplicate(invalid, negative);
    MaskReg nonnegative, causal, valid;
    if (reordered) {
        auto all16 = CreateMask<int16_t, MaskPattern::ALL>();
        RegTensor<uint16_t> planOrder;
        RegTensor<int16_t> plans;
        LoadAlign(planOrder, order);
        Gather(plans, rowPlan, planOrder, all16);
        StoreAlign(chunkPlan, plans, all16);
    }
    for (uint16_t part = 0; part < static_cast<uint16_t>((columns + 63) / 64); ++part) {
        if (reordered) {
            LoadAlign<uint16_t, LoadDist::DIST_UNPACK_B16>(indices, order + part * 64);
            Gather(selected, tokens, reinterpret_cast<RegTensor<uint32_t> &>(indices), all);
        } else {
            LoadAlign(selected, tokens + sourceStart + part * 64);
        }
        Compares<int32_t, CMPMODE::GE>(nonnegative, selected, 0, all);
        Compares<int32_t, CMPMODE::LT>(causal, selected, tokenLimit, all);
        And(valid, nonnegative, causal, all);
        Select(maskValues, zeros, invalid, valid);
        StoreAlign(output + part * 64, maskValues, all);
    }
}

__simd_vf__ inline void ApplySparseTaskMask(
    __ubuf__ float *scores, __ubuf__ float *taskMask,
    uint32_t columns, uint32_t rows, uint32_t pitch, float negative)
{
    using namespace AscendC;
    using namespace AscendC::MicroAPI;
    auto all = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> values, maskValues, invalid;
    Duplicate(invalid, negative);
    MaskReg valid;
    for (uint16_t part = 0; part < static_cast<uint16_t>((columns + 63) / 64); ++part) {
        LoadAlign(maskValues, taskMask + part * 64);
        Compares<float, CMPMODE::EQ>(valid, maskValues, 0.0F, all);
        for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
            LoadAlign(values, scores + row * pitch + part * 64);
            Select(values, values, invalid, valid);
            StoreAlign(scores + row * pitch + part * 64, values, all);
        }
    }
}

}
