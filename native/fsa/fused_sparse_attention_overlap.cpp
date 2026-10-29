// Single adapter translation unit: donor op_api_common.h has non-inline helpers.
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "aclnn_torch_adapter/op_api_common.h"
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
#include "shared_runtime.cpp"
#include "fused_sparse_attention_overlap_impl.cpp"
#include "external_plan_warmup.cpp"
