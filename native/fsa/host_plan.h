#pragma once

#include <ATen/ATen.h>

namespace sglang::npu_kernel {
void MaterializeHostPlan(const at::Tensor &trigger, const at::Tensor &keys,
                         const at::Tensor &slots, const at::Tensor &owners,
                         const at::Tensor &lengths, const at::Tensor &cacheable,
                         at::Tensor applied, at::Tensor state, at::Tensor membership,
                         at::Tensor misses, int64_t rows, bool commit);
}
