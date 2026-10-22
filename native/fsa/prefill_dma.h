#pragma once

#include <ATen/ATen.h>

namespace sglang::npu_kernel {
void PrefillDmaCopy(const at::Tensor &device_tensor, const at::Tensor &host_metadata,
                   int64_t device_address, const at::Tensor &runs, bool to_host);
}
