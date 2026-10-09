"""Stage and narrowly extend the pinned SDK kernel without editing the SDK."""

import hashlib
import json
import sys
from pathlib import Path

SOURCES = {
    "lightning_indexer_common.h": "a8af67d754561462a510735cd6f7334e03f8bb304ea9acc92bb07bb5256a59c1",
    "arch35/lightning_indexer_kernel.h": "7e73c45edab4213e758420abde8c408badccf21717d929f526924f9d5e3ca6a1",
    "arch35/lightning_indexer_service_cube.h": "c69626a0e20189aa72acf5a3bb943599932199b5b5639b2e358142e36fc8cccd",
    "arch35/lightning_indexer_service_vector.h": "25a8e06579941425362ad8f45d15012a2fa7a833252d94eb95d933d9e85903d4",
    "arch35/vf/lightning_indexer_topk.h": "ecb3846c76b8f61675cf806e65b09e7da3ec4fce2aea8375c00008e7fde49bbb",
    "arch35/vf/lightning_indexer_vector1.h": "2490851ce8582128ebe0635fcc0931c5c8e46777183f351dab3c982c9da9849c",
    "arch35/vf/vf_topk_gather.h": "c84d230a03afc3328118a1bb0db277d890c1bfcf27d5ea66c517ff451951e248",
    "arch35/vf/common/lightning_indexer_vector1_base.h": "a7804b5f5de6155f8a129c79b832062b199383831e7d242c9fe807f629c9ad99",
    "arch35/vf/common/vf_topk_16_gather.h": "0b969a603f6e47f393f496072b5f266dc10631b877a18384bb4148b4dab27932",
}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Pinned LIM source anchor changed: {old[:80]}")
    return source.replace(old, new, 1)


def stage_reference(source_root, destination):
    manifest = {}
    for relative, expected in SOURCES.items():
        raw = (source_root / relative).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Unqualified CANN source revision: {relative}: {actual}")
        source = raw.decode("utf-8")
        if relative == "arch35/lightning_indexer_kernel.h":
            source = replace_once(
                source,
                "    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);",
                "    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);\n    constInfo.isLDOpen = true;",
            )
            source = replace_once(
                source,
                """        if (runInfo.isLastS2InnerLoop) {   // 本核s2last
            vectorService.ProcessTopK(runInfo);
        }""",
                "",
            )
            source = replace_once(
                source,
                "    __aicore__ inline void Process();",
                """    __aicore__ inline void Process();
    template <typename RowPlan>
    __aicore__ inline void FinishTopK(GM_ADDR scores, RowPlan &rowPlan)
    {
        if ASCEND_IS_AIV {
            vectorService.AllocEventID();
            rowPlan.BindScratch(vectorService.ResidentScratch());
            vectorService.BeginRowTopK();
            {
                for (uint32_t row = GetBlockIdx(); row < constInfo.batchSize; row += GetBlockNum() * 2) {
                    GlobalTensor<SCORE_T> rowScores;
                    rowScores.SetGlobalBuffer((__gm__ SCORE_T *)scores + row * constInfo.kSeqSize);
                    vectorService.InitVecWorkspaceTensor(rowScores);
                    LICommon::RunInfo info{};
                    info.actS1Size = 1;
                    info.actS2Size = actualSeqLengthsGmKv.GetValue(row);
                    info.actS2SizeOrig = info.actS2Size;
                    info.indiceOutOffset = row * constInfo.sparseCount;
                    vectorService.ProcessTopK(info);
                    rowPlan(row, vectorService.ResidentIndices());
                }
            }
            vectorService.FreeEventID();
        }
    }""",
            )
        if relative == "arch35/lightning_indexer_service_vector.h":
            source = replace_once(
                source,
                "    __aicore__ inline LightningIndexerServiceVector(){};",
                """    __aicore__ inline LightningIndexerServiceVector(){};
    GlobalTensor<SCORE_T> exportedScores;
    __aicore__ inline void SetExportedScores(GM_ADDR output)
    {
        exportedScores.SetGlobalBuffer((__gm__ SCORE_T *)output);
    }
    __aicore__ inline LocalTensor<uint32_t> ResidentIndices() { return indicesOutLocal_; }
    __aicore__ inline void BeginRowTopK() { blockId_ = 0; }
    __aicore__ inline LocalTensor<int32_t> ResidentScratch() { return resMm1UB_.template ReinterpretCast<int32_t>(); }""",
            )
            anchor = """    DataCopyPad(scoreGm[vec1OutGmOffset],
                vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_],
                copyOutParams);"""
            source = replace_once(
                source,
                anchor,
                """
    const uint64_t exportRow = info.indiceOutOffset / topkCount_ + curAivS1Idx;
    DataCopyPad(exportedScores[exportRow * kSeqSizeAlign + curS2Idx],
                vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_],
                copyOutParams);""",
            )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8", newline="\n")
        manifest[relative] = dict(
            original=actual, staged=hashlib.sha256(target.read_bytes()).hexdigest()
        )
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    stage_reference(Path(sys.argv[1]), Path(sys.argv[2]))
