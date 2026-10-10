#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "configured_indexer.h"
#include "validation.h"

namespace {
void IndexerManageOut(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &indexTable, const at::Tensor &ends, const at::Tensor &lengths,
    const at::Tensor &table, const at::Tensor &requests, const at::Tensor &generations,
    at::Tensor indexEnds, at::Tensor poolLengths, at::Tensor indices, at::Tensor scores, at::Tensor workspace,
    at::Tensor mapping, at::Tensor sources, at::Tensor physical, at::Tensor state,
    at::Tensor outputIds, at::Tensor outputSlots, at::Tensor membership, at::Tensor unionSources,
    at::Tensor unionSlots, at::Tensor unionPhysical, int64_t hostTokens, int64_t mutableTail)
{
    const int64_t cores = SglLimHost::SelectIndexerCores(query, scores, workspace);
    SglLimHost::ValidateIndexer(query, key, weights, indexEnds, poolLengths, indexTable, indices, scores, workspace, cores);
    SglLimHost::ValidateManager(indices, scores, ends, lengths, table, requests, generations, mapping,
        sources, physical, state, outputIds, outputSlots, membership, unionSources, unionSlots,
        unionPhysical, hostTokens, mutableTail, true, true);
    c10::DeviceGuard guard(query.device());
    auto npuStream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : {query, key, weights, indexTable, ends, lengths, table, requests, generations,
            indexEnds, poolLengths, indices, scores, workspace, mapping, sources, physical, state,
            outputIds, outputSlots, membership, unionSources, unionSlots, unionPhysical})
        tensor.record_stream(npuStream);
    const auto stream = npuStream.stream(false);
    auto launch = [=]() -> int {
        return sgl_lim_configured_indexer_manage(static_cast<uint32_t>(cores), stream,
            query.data_ptr(), key.data_ptr(), weights.data_ptr(), indexEnds.data_ptr(), poolLengths.data_ptr(),
            indexTable.data_ptr(), indices.data_ptr(), scores.data_ptr(), workspace.data_ptr(),
            ends.data_ptr(), lengths.data_ptr(), table.data_ptr(), requests.data_ptr(), generations.data_ptr(),
            mapping.data_ptr(), sources.data_ptr(), physical.data_ptr(), state.data_ptr(),
            outputIds.data_ptr(), outputSlots.data_ptr(), membership.data_ptr(),
            unionSources.data_ptr(), unionSlots.data_ptr(), unionPhysical.data_ptr(),
            static_cast<uint32_t>(query.size(0)), static_cast<uint32_t>(query.size(1)),
            static_cast<uint32_t>(scores.size(1)), static_cast<uint32_t>(indexTable.size(1)),
            static_cast<uint32_t>(ends.numel()), static_cast<uint32_t>(mapping.size(0)),
            static_cast<uint32_t>(mapping.size(1)), static_cast<uint32_t>(sources.size(1)),
            static_cast<uint32_t>(table.size(1)), static_cast<uint32_t>(unionSources.size(1)),
            static_cast<uint64_t>(hostTokens), static_cast<uint32_t>(mutableTail));
    };
    at_npu::native::OpCommand::RunOpApi("sgl_lim_indexer_manage", launch);
}

void IndexerManageMeta(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t)
{}
}

TORCH_LIBRARY_FRAGMENT(sgl_lim_native, module)
{
    module.def("indexer_manage_out(Tensor query, Tensor key, Tensor weights, Tensor index_table, "
        "Tensor ends, Tensor lengths, Tensor table, Tensor requests, Tensor generations, "
        "Tensor(a!) index_ends, Tensor(b!) pool_lengths, Tensor(c!) indices, Tensor(d!) scores, Tensor(e!) workspace, "
        "Tensor(f!) mapping, Tensor(g!) sources, Tensor(h!) physical, Tensor(i!) state, Tensor(j!) output_ids, "
        "Tensor(k!) output_slots, Tensor(l!) membership, Tensor(m!) union_sources, Tensor(n!) union_slots, "
        "Tensor(o!) union_physical, int host_tokens, int mutable_tail) -> ()");
}
TORCH_LIBRARY_IMPL(sgl_lim_native, PrivateUse1, module)
{
    module.impl("indexer_manage_out", TORCH_FN(IndexerManageOut));
}
TORCH_LIBRARY_IMPL(sgl_lim_native, Meta, module)
{
    module.impl("indexer_manage_out", TORCH_FN(IndexerManageMeta));
}
