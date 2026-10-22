#include <ATen/ATen.h>
#include <algorithm>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "aclrtlaunch_sgl_lim_indexer.h"
#include "validation.h"

namespace SglLimHost {

int64_t SelectIndexerCores(const at::Tensor &query, const at::Tensor &scores, const at::Tensor &workspace)
{
    TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 && scores.dim() == 2 && scores.size(1) > 0,
        "Invalid LIM dimensions for runtime tiling");
    const int64_t bytesPerCore = 4 * scores.size(1) * 4;
    TORCH_CHECK(workspace.dim() == 1 && workspace.numel() % bytesPerCore == 0,
        "LIM workspace must reserve an integral number of cores");
    const int64_t capacity = workspace.numel() / bytesPerCore;
    TORCH_CHECK(capacity >= 1 && capacity <= 64,
        "LIM workspace core capacity exceeds the supported A5 range");
    const int64_t tasks = query.size(0) * ((scores.size(1) + 127) / 128);
    return std::min(capacity, tasks);
}

void Validate(const at::Tensor &tensor, const at::Tensor &query, at::ScalarType dtype, const char *name)
{
    TORCH_CHECK(tensor.device() == query.device() && tensor.scalar_type() == dtype && tensor.is_contiguous(),
        name, " has incompatible device, dtype or layout");
}

void ValidateIndexer(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &queryEnds, const at::Tensor &poolLengths, const at::Tensor &table,
    at::Tensor indices, at::Tensor scores, at::Tensor workspace, int64_t cores)
{
    TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1, "LIM requires NPU inputs");
    Validate(query, query, at::kBFloat16, "query");
    Validate(key, query, at::kBFloat16, "key");
    Validate(weights, query, at::kBFloat16, "weights");
    Validate(queryEnds, query, at::kInt, "query_ends");
    Validate(poolLengths, query, at::kInt, "pool_lengths");
    Validate(table, query, at::kInt, "table");
    Validate(indices, query, at::kInt, "indices");
    Validate(scores, query, at::kInt, "scores");
    Validate(workspace, query, at::kByte, "workspace");
    TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 && query.size(2) == 128,
        "Expected nonempty query[rows,heads,128]");
    const int64_t rows = query.size(0);
    const int64_t heads = query.size(1);
    TORCH_CHECK(heads == 32 || heads == 64, "A5 LIM supports 32 or 64 index heads");
    TORCH_CHECK(key.dim() == 4 && key.size(0) > 0 && key.size(1) == 128 && key.size(2) == 1 && key.size(3) == 128,
        "Expected pooled key[pages,128,1,128]");
    TORCH_CHECK(weights.dim() == 2 && weights.size(0) == rows && weights.size(1) == heads,
        "Expected weights[rows,heads]");
    TORCH_CHECK(queryEnds.dim() == 1 && queryEnds.numel() == rows && poolLengths.dim() == 1 && poolLengths.numel() == rows,
        "Indexer requires one per-row causal length");
    TORCH_CHECK(table.dim() == 2 && table.size(0) == rows && table.size(1) > 0,
        "Expected pooled page table[rows,columns]");
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == rows && indices.size(1) == 512,
        "Expected indices[rows,512]");
    TORCH_CHECK(scores.dim() == 2 && scores.size(0) == rows && scores.size(1) >= 128 &&
        scores.size(1) <= (1 << 19) && scores.size(1) % 128 == 0 && table.size(1) * 128 >= scores.size(1),
        "Scores must cover the page-aligned pooled source capacity");
    TORCH_CHECK(cores >= 1 && cores <= SelectIndexerCores(query, scores, workspace),
        "LIM core count exceeds reserved workspace or available tasks");
    TORCH_CHECK(workspace.numel() >= cores * 4 * scores.size(1) * 4, "LIM workspace is too small");
}

void IndexerOut(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &queryEnds, const at::Tensor &poolLengths, const at::Tensor &table,
    at::Tensor indices, at::Tensor scores, at::Tensor workspace, int64_t cores)
{
    ValidateIndexer(query, key, weights, queryEnds, poolLengths, table, indices, scores, workspace, cores);
    const auto rows = query.size(0);
    const auto heads = query.size(1);
    c10::DeviceGuard guard(query.device());
    auto npuStream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : {query, key, weights, queryEnds, poolLengths, table, indices, scores, workspace}) {
        tensor.record_stream(npuStream);
    }
    const auto stream = npuStream.stream(false);
    auto launch = [=]() -> int {
        return ACLRT_LAUNCH_KERNEL(sgl_lim_indexer)(
            static_cast<uint32_t>(cores), stream, query.data_ptr(), key.data_ptr(), weights.data_ptr(),
            queryEnds.data_ptr(), poolLengths.data_ptr(), table.data_ptr(), indices.data_ptr(), scores.data_ptr(),
            workspace.data_ptr(), static_cast<uint32_t>(rows), static_cast<uint32_t>(heads),
            static_cast<uint32_t>(scores.size(1)), static_cast<uint32_t>(table.size(1)));
    };
    at_npu::native::OpCommand::RunOpApi("sgl_lim_indexer", launch);
}

void IndexerMeta(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, const at::Tensor &, at::Tensor, at::Tensor, at::Tensor, int64_t)
{}

}

TORCH_LIBRARY(sgl_lim_native, module)
{
    module.def("indexer_out(Tensor query, Tensor key, Tensor weights, Tensor query_ends, Tensor pool_lengths, "
        "Tensor table, Tensor(a!) indices, Tensor(b!) scores, Tensor(c!) workspace, int cores) -> ()");
}

TORCH_LIBRARY_IMPL(sgl_lim_native, PrivateUse1, module)
{
    module.impl("indexer_out", TORCH_FN(SglLimHost::IndexerOut));
}

TORCH_LIBRARY_IMPL(sgl_lim_native, Meta, module)
{
    module.impl("indexer_out", TORCH_FN(SglLimHost::IndexerMeta));
}
