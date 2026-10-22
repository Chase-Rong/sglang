#pragma once
#include <ATen/ATen.h>

namespace SglLimHost {
int64_t SelectIndexerCores(const at::Tensor &, const at::Tensor &, const at::Tensor &);
void ValidateIndexer(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, const at::Tensor &, at::Tensor, at::Tensor, at::Tensor, int64_t);
void ValidateManager(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &,
    const at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    int64_t, int64_t, bool, bool);
}
