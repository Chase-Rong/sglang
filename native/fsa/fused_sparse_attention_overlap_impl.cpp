// Included by fused_sparse_attention_overlap.cpp after the common ACL runtime helpers.
#include "fused_sparse_attention_overlap.h"
#include "opp_abi_guard.h"
#include "aclrtlaunch_fused_sparse_attention_overlap_finish.h"
namespace sglang::npu_kernel {
namespace {
const OverlapOppApi &GetOverlapOppApi() {
    // Resolve and validate once; the exact verified pair is passed to the
    // launch macro below. A marker found in another vendor cannot authorize an
    // old same-name row OPP. The resolver retains its dlopen library handles.
    static const OverlapOppApi api = []() {
        OverlapOppApi resolved{
            SharedSymbol("aclnnFusedSparseAttentionOverlapGetWorkspaceSize"),
            SharedSymbol("aclnnFusedSparseAttentionOverlap")};
        auto *marker = SharedSymbol(kFusedSparseAttentionOverlapAbiSymbol);
        const char *error = ValidateOverlapOppAbi(resolved, marker);
        TORCH_CHECK(error == nullptr,
                    "fused_sparse_attention_overlap requires the matching ABI3 OPP: ",
                    error ? error : "");
        return resolved;
    }();
    return api;
}
}
std::vector<int64_t> SharedV3Capabilities() {
 GetOverlapOppApi();SharedWorkspaceBytes();
 return {3,1,32,512,0,128,2051,32767,32767,9,1024,1,8,16,1};
}
at::Tensor FusedSparseAttentionOverlap(
    const at::Tensor & query,
    at::Tensor & shared_kv,
    const at::Tensor & full_kv,
    int64_t full_kv_address,
    const at::Tensor & full_block_table,
    const at::Tensor & full_kv_actual_seq,
    const at::Tensor & full_q_actual_seq,
    const at::Tensor & logical_topk,
    const at::Tensor & request_slots,
    const at::Tensor & row_to_request,
    const at::Tensor & row_union_index,
    const at::Tensor & union_keys,
    const at::Tensor & union_slots,
    const at::Tensor & union_count,
    const at::Tensor & union_miss,
    const at::Tensor & union_owner,
    const at::Tensor & union_cacheable,
    const at::Tensor & identity,
    at::Tensor & ready,
    at::Tensor & status,
    at::Tensor & completion_identity,
    at::Tensor & counters,
    at::Tensor & safe_topk,
    at::Tensor & membership,
    at::Tensor & selection_table,
    at::Tensor & selection_status,
    at::Tensor & workspace,
    int64_t resident_capacity,
    double scale,
    int64_t abi_version) {
    TORCH_CHECK(abi_version == 3, "fused_sparse_attention_overlap ABI version must be 3");
    const auto &opp_api = GetOverlapOppApi(); // before descriptors, GWS or any kernel submission
    auto workspace_bound=SharedWorkspaceBytes();
    TORCH_CHECK(query.device().type() == at::DeviceType::PrivateUse1 && query.is_contiguous() &&
                query.scalar_type() == at::kBFloat16 && query.dim() == 3 && query.size(0)>0 &&
                query.size(1)==32 && query.size(2)==512,
                "fused_sparse_attention_overlap supports BF16 TND H32 NoPE512 RoPE0 only");
    TORCH_CHECK(c10_npu::getCurrentNPUStream().device_index()==query.device().index(),
                "fused_sparse_attention_overlap current stream device differs from query");
    TORCH_CHECK(resident_capacity>0 && resident_capacity<=32767 && resident_capacity%128==0 &&
                std::isfinite(scale), "fused_sparse_attention_overlap invalid capacity/page128/scale");
    auto T=query.size(0), C=resident_capacity;
    TORCH_CHECK(request_slots.dim()==1 && request_slots.numel()>0 && logical_topk.dim()==3 &&
                union_keys.dim()==2 && shared_kv.dim()==3, "fused_sparse_attention_overlap invalid ranks");
    auto B=request_slots.numel(), K=logical_topk.size(2), W=union_keys.size(1);
    TORCH_CHECK(B<=1024 && B<=T,"fused_sparse_attention_overlap requires B<=min(T,1024)");
    TORCH_CHECK(K>0 && K<=2051 && W>=K && W<=32767 && W%16==0 && T<=INT32_MAX/K &&
                shared_kv.size(0)>0 && shared_kv.size(0)%(C/128)==0,
                "fused_sparse_attention_overlap invalid K/W/row/payload capacities");
    auto Q=shared_kv.size(0)/(C/128);
    TORCH_CHECK(Q>1 && Q<=32767, "fused_sparse_attention_overlap invalid Qcap");
    CheckTensor(shared_kv,query,at::kBFloat16,{Q*C/128,128,512},"shared_kv");
    TORCH_CHECK(full_kv.device().is_cpu() && full_kv.is_contiguous() && full_kv.storage_offset()==0 &&
                full_kv.scalar_type()==at::kBFloat16 && full_kv.dim()==3 && full_kv.size(0)>0 &&
                full_kv.size(1)==128 && full_kv.size(2)==512 && full_kv_address>0 &&
                full_kv.size(0)<=INT64_MAX/(128*512*2) && full_kv_address<=INT64_MAX-full_kv.nbytes(),
                "fused_sparse_attention_overlap needs positive base-aligned registered BF16 Host pages128 NoPE512");
    TORCH_CHECK(shm_validate_mapping(full_kv.data_ptr(),full_kv_address,full_kv.nbytes(),query.device().index())==0 ||
                ValidateMemFabricRegion(full_kv.data_ptr(),full_kv_address,full_kv.nbytes(),query.device().index()),
                "fused_sparse_attention_overlap full_kv does not match a live Host registration on this NPU");
    TORCH_CHECK(full_block_table.dim()==2 && full_block_table.size(1)>0,
                "fused_sparse_attention_overlap full_block_table must have positive columns");
    auto N=full_block_table.size(1);
    CheckTensor(full_block_table,query,at::kInt,{B,N},"full_block_table");
    CheckTensor(full_kv_actual_seq,query,at::kInt,{B},"full_kv_actual_seq");
    CheckTensor(full_q_actual_seq,query,at::kInt,{B},"full_q_actual_seq");
    CheckTensor(logical_topk,query,at::kInt,{T,1,K},"logical_topk");
    CheckTensor(request_slots,query,at::kLong,{B},"request_slots");
    CheckTensor(row_to_request,query,at::kInt,{T},"row_to_request");
    CheckTensor(row_union_index,query,at::kInt,{T,K},"row_union_index");
    CheckTensor(union_keys,query,at::kLong,{B,W},"union_keys");
    CheckTensor(union_slots,query,at::kInt,{B,W},"union_slots");
    CheckTensor(union_count,query,at::kInt,{B},"union_count");
    CheckTensor(union_miss,query,at::kBool,{B,W},"union_miss");
    CheckTensor(union_owner,query,at::kInt,{B,W},"union_owner");
    CheckTensor(union_cacheable,query,at::kBool,{B,W},"union_cacheable");
    CheckTensor(identity,query,at::kLong,{B,9},"identity");
    CheckTensor(ready,query,at::kInt,{B,W},"ready");
    CheckTensor(status,query,at::kInt,{B},"status");
    CheckTensor(completion_identity,query,at::kLong,{B,9},"completion_identity");
    CheckTensor(counters,query,at::kLong,{B,8},"counters");
    CheckTensor(safe_topk,query,at::kInt,{T,1,K},"safe_topk");
    CheckTensor(membership,query,at::kShort,{T,16400},"membership");
    CheckTensor(selection_table,query,at::kInt,{T,C/128},"selection_table");
    TORCH_CHECK(selection_status.dim()==3 && selection_status.size(2)>K,
                "fused_sparse_attention_overlap selection_status stride must exceed K");
    auto S=selection_status.size(2);
    CheckTensor(selection_status,query,at::kInt,{T,1,S},"selection_status");
    TORCH_CHECK(workspace.dim()==1 && workspace.device()==query.device() &&
                workspace.scalar_type()==at::kByte && workspace.is_contiguous() &&
                workspace.numel()>=workspace_bound,"fused_sparse_attention_overlap requires fixed uint8 workspace from shared_workspace_bytes()");
    std::vector<at::Tensor> mutable_buffers={shared_kv,ready,status,completion_identity,counters,workspace,
                                           safe_topk,membership,selection_table,selection_status};
    std::vector<at::Tensor> inputs={query,full_kv,full_block_table,full_kv_actual_seq,full_q_actual_seq,
        logical_topk,request_slots,row_to_request,row_union_index,union_keys,union_slots,union_count,
        union_miss,union_owner,union_cacheable,identity};
    for(size_t i=0;i<mutable_buffers.size();++i) {
        for(auto &input:inputs) TORCH_CHECK(!Overlaps(mutable_buffers[i],input),"fused_sparse_attention_overlap mutable/input alias");
        for(size_t j=0;j<i;++j) TORCH_CHECK(!Overlaps(mutable_buffers[i],mutable_buffers[j]),"fused_sparse_attention_overlap workspace alias");
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(membership.data_ptr())%4==0,"fused_sparse_attention_overlap membership needs int32 alignment");
    // Entry runs outside a PTA custom handler. State/Graph own the tensor for
    // the entire execution; no Tensor owner is captured in queued callbacks.
    c10_npu::NPUCachingAllocator::recordStream(workspace.storage().data_ptr(),c10_npu::getCurrentNPUStream());
    uint32_t blocks=1;
    int64_t sparse_block=1,sparse_mode=3;
    // The first occurrence of each initial miss loads registered Host payload and
    // writes shared_kv once. All later occurrences read the resident shared_kv.
    auto key=full_kv.unsqueeze(2);
    RegisteredAclTensor key_input{key,full_kv_address};
    auto rope=at::empty({shared_kv.size(0),128,0},query.options());
    auto output=at::empty_like(query);
    auto actual=selection_status.view({T,S}).select(1,K);
    auto seq=membership.view(at::kInt).view({-1});
    c10::optional<at::Tensor> block=full_block_table, qseq=seq.slice(0,0,B), kvseq=seq.slice(0,B,2*B);
    c10::optional<at::Tensor> no_rope=c10::nullopt;
    char layout_q[]="TND", layout_kv[]="PA_BSND";
    PREPARE_SHARED_NPU_CMD(aclnnFusedSparseAttentionOverlap,opp_api,query,key_input,key_input,logical_topk,
        block,qseq,kvseq,no_rope,no_rope,rope,shared_kv,selection_table,selection_status,membership,
        row_union_index,union_keys,union_count,ready,counters,request_slots,union_slots,
        scale,sparse_block,layout_q,layout_kv,sparse_mode,output,actual);
    call_lifetime->MarkQueued();
    cmd.Run();
    EXEC_SHARED_KERNEL_CMD(fused_sparse_attention_overlap_finish,blocks,output,request_slots,row_to_request,row_union_index,
        union_slots,union_count,identity,ready,status,completion_identity,safe_topk,membership,union_miss,counters,
        B,T,K,W,C);
    return output;
}
}
