#ifndef COPY_SPARSE_ATTENTION_LAYOUT_ARCH35_H
#define COPY_SPARSE_ATTENTION_LAYOUT_ARCH35_H

#include "kernel_operator.h"

namespace CopySfaBase {

constexpr uint32_t COPY_SFA_NZ_SCRATCH_ROWS = 16;
constexpr uint32_t COPY_SFA_NZ_SCRATCH_ELEMENTS = (COPY_SFA_NZ_SCRATCH_ROWS + 1) * 512;
constexpr uint32_t COPY_SFA_NZ_SCRATCH_BYTES = COPY_SFA_NZ_SCRATCH_ELEMENTS * sizeof(bfloat16_t);

template <typename BitType>
__simd_vf__ void CopyNope512NdToNzVF(
    __ubuf__ BitType *destination, __ubuf__ BitType *source,
    uint32_t rows, uint32_t pitch)
{
    AscendC::MicroAPI::RegTensor<BitType> values;
    auto mask = AscendC::MicroAPI::CreateMask<BitType, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t tile = 0; tile < 4; ++tile) {
        __ubuf__ BitType *sourceRow = source + tile * 128;
        __ubuf__ BitType *destinationRow = destination + tile * 128 * pitch;
        for (uint16_t row = 0; row < static_cast<uint16_t>(rows); ++row) {
            AscendC::MicroAPI::LoadAlign<BitType>(values, sourceRow);
            AscendC::MicroAPI::StoreAlign<BitType,
                AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    destinationRow, values, pitch, 1, mask);
            sourceRow += 576;
        }
    }
}

__aicore__ inline void CopyBf16Nope512UbToL1(
    AscendC::LocalTensor<bfloat16_t> destinationL1,
    AscendC::LocalTensor<bfloat16_t> gatheredUb,
    AscendC::LocalTensor<bfloat16_t> scratchUb,
    uint32_t dealRow, uint32_t s2Start, uint32_t s2RealSize, uint32_t scratchRows)
{
    const auto scratchReady = static_cast<AscendC::TEventID>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));

    auto sourceBits = reinterpret_cast<__ubuf__ half *>(gatheredUb.GetPhyAddr());
    auto destinationBits = reinterpret_cast<__ubuf__ half *>(scratchUb.GetPhyAddr());
    const uint32_t pitch = scratchRows + 1;
    const auto layoutReady = static_cast<AscendC::TEventID>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    for (uint32_t rowOffset = 0; rowOffset < dealRow; rowOffset += scratchRows) {
        const uint32_t remainingRows = dealRow - rowOffset;
        const uint32_t rows = remainingRows < scratchRows ? remainingRows : scratchRows;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(scratchReady);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(scratchReady);
        CopyNope512NdToNzVF<half>(destinationBits, sourceBits + rowOffset * 576, rows, pitch);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(layoutReady);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(layoutReady);
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 32;
        copyParams.blockLen = rows;
        copyParams.srcGap = pitch - rows;
        copyParams.dstGap = ((s2RealSize + 15) / 16 * 16) - rows;
        AscendC::DataCopy(destinationL1[(s2Start + rowOffset) * 16], scratchUb, copyParams);
    }
}

}

#endif
