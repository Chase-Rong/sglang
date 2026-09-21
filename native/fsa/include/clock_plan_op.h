#pragma once
#include <ATen/ATen.h>
namespace sglang::npu_kernel {
void enqueue_fsa_clock_plan(const at::Tensor& trigger, const at::Tensor& keys,
                           const at::Tensor& owners, const at::Tensor& lengths,
                           at::Tensor resident, at::Tensor ages, at::Tensor row_state,
                           at::Tensor slots, at::Tensor workspace, int64_t active_rows);
}
