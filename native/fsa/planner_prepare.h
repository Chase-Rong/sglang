#pragma once
#include <ATen/ATen.h>
#include <vector>

namespace sglang::npu_kernel {
// Fixed slot order is documented in PLANNER_PREPARE_ABI.md. No allocation or
// Torch tensor computation is performed by this entry point.
void PlannerPrepare(at::TensorList inputs, at::TensorList outputs,
                    int64_t capacity, int64_t max_routes, int64_t host_tokens,
                    int64_t abi_version);
std::vector<int64_t> PlannerPrepareCapabilities();
}
