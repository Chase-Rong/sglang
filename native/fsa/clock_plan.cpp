#include "clock_plan_op.h"
#include "clock_plan.h"

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sglang {
namespace npu_kernel {
namespace {

struct PlannerPayload {
    at::Tensor keys, owners, lengths, resident, ages, row_state, slots, workspace;
    int64_t rows;
    bool delete_after_run;
    std::vector<uint8_t> protected_slots;
};

struct GraphPayloads {
    aclmdlRI graph;
    std::vector<std::unique_ptr<PlannerPayload>> payloads;
};

std::mutex &RegistryMutex()
{
    static auto *mutex = new std::mutex();
    return *mutex;
}

auto &GraphRegistry()
{
    static auto *registry = new std::unordered_map<aclmdlRI, std::unique_ptr<GraphPayloads>>();
    return *registry;
}

void DeleteGraphPayloads(void *data) noexcept
{
    auto *registry = static_cast<GraphPayloads *>(data);
    std::lock_guard<std::mutex> lock(RegistryMutex());
    auto iterator = GraphRegistry().find(registry->graph);
    if (iterator != GraphRegistry().end() && iterator->second.get() == registry) {
        GraphRegistry().erase(iterator);
    }
}

template<typename Index>
void PlanRowsTyped(PlannerPayload *payload) noexcept
{
    const int64_t width = payload->keys.size(1);
    const int64_t capacity = payload->resident.size(1);
    const auto *keys = payload->keys.data_ptr<int64_t>();
    const auto *owners = payload->owners.data_ptr<int64_t>();
    const auto *lengths = payload->lengths.data_ptr<int64_t>();
    auto *residents = payload->resident.data_ptr<int64_t>();
    auto *ages = payload->ages.data_ptr<int64_t>();
    auto *states = payload->row_state.data_ptr<int64_t>();
    auto *outputs = payload->slots.data_ptr<int64_t>();
    auto* workspace = payload->workspace.data_ptr<Index>();
    for (int64_t row = 0; row < payload->rows; ++row) {
        PlanClockRow(keys + row * width, width, owners[row], lengths[row],
                     residents + row * capacity, ages + row * capacity,
                     states + row * 3, outputs + row * width, capacity,
                     workspace + row * ClockWorkspaceSize(capacity),
                     payload->protected_slots.data());
    }
}

void PlanRows(void *data) noexcept
{
    auto *payload = static_cast<PlannerPayload *>(data);
    if (payload->workspace.scalar_type() == at::kShort) {
        PlanRowsTyped<int16_t>(payload);
    } else {
        PlanRowsTyped<int64_t>(payload);
    }
    if (payload->delete_after_run) {
        delete payload;
    }
}

void CheckHostLong(const at::Tensor &tensor)
{
    TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous() && tensor.scalar_type() == at::kLong,
                "FSA planner tensors must be contiguous CPU int64 storage");
}

}

void enqueue_fsa_clock_plan(const at::Tensor &trigger, const at::Tensor &keys,
                          const at::Tensor &owners, const at::Tensor &lengths,
                          at::Tensor resident, at::Tensor ages, at::Tensor row_state,
                          at::Tensor slots, at::Tensor workspace, int64_t active_rows)
{
    TORCH_CHECK(trigger.device().type() == at::DeviceType::PrivateUse1,
                "FSA planner trigger must be an NPU tensor");
    for (const auto &tensor : {keys, owners, lengths, resident, ages, row_state, slots}) {
        CheckHostLong(tensor);
    }
    TORCH_CHECK(workspace.device().is_cpu() && workspace.is_contiguous() &&
                (workspace.scalar_type() == at::kLong || workspace.scalar_type() == at::kShort),
                "FSA CLOCK workspace must be contiguous CPU int16 or int64 storage");
    TORCH_CHECK(keys.dim() == 2 && resident.dim() == 2 && active_rows > 0 && active_rows <= keys.size(0),
                "invalid FSA planner row geometry");
    const int64_t capacity = resident.size(1);
    TORCH_CHECK(keys.size(1) > 0 && keys.size(1) <= 2051 && capacity >= keys.size(1) && capacity <= 32767 && resident.size(0) == keys.size(0),
                "FSA resident capacity must cover TopK and fit the int16 plan ABI");
    TORCH_CHECK(ages.sizes() == resident.sizes() && slots.sizes() == keys.sizes() &&
                owners.numel() == keys.size(0) && lengths.numel() == keys.size(0) &&
                row_state.dim() == 2 && row_state.size(0) == keys.size(0) && row_state.size(1) == 3,
                "FSA planner state geometry mismatch");
    TORCH_CHECK(workspace.dim() == 2 && workspace.size(0) == keys.size(0) &&
                workspace.size(1) == static_cast<int64_t>(ClockWorkspaceSize(capacity)),
                "FSA CLOCK workspace geometry mismatch");
    const auto stream = c10_npu::getCurrentNPUStream();
    TORCH_CHECK(stream.device_index() == trigger.device().index(), "FSA planner stream device mismatch");
    aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI graph = nullptr;
    TORCH_CHECK(aclmdlRICaptureGetInfo(stream.stream(), &status, &graph) == ACL_SUCCESS,
                "FSA planner could not query graph capture");
    TORCH_CHECK(status != ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED, "FSA planner graph capture invalidated");
    const bool captured = status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    TORCH_CHECK(!captured || graph != nullptr, "FSA planner active graph has no runtime instance");
    auto payload = std::make_unique<PlannerPayload>(PlannerPayload{
        keys, owners, lengths, resident, ages, row_state, slots, workspace, active_rows, !captured, std::vector<uint8_t>(capacity)});
    auto *raw = payload.get();
    if (captured) {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        auto &registry = GraphRegistry()[graph];
        if (!registry) {
            registry = std::make_unique<GraphPayloads>();
            registry->graph = graph;
            if (aclmdlRIDestroyRegisterCallback(graph, DeleteGraphPayloads, registry.get()) != ACL_SUCCESS) {
                GraphRegistry().erase(graph);
                TORCH_CHECK(false, "FSA planner could not retain payload until graph destruction");
            }
        }
        registry->payloads.push_back(std::move(payload));
    }
    const auto result = aclrtLaunchHostFunc(stream.stream(), PlanRows, raw);
    if (result == ACL_SUCCESS && !captured) {
        payload.release();
    }
    TORCH_CHECK(result == ACL_SUCCESS, "FSA planner host callback launch failed: ", result);
}

}
}
