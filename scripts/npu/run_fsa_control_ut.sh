#!/usr/bin/env bash
set -euo pipefail

fsa_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fsa_work=${SGLANG_FSA_WORK_DIR:?Set SGLANG_FSA_WORK_DIR under the assigned work directory}
mkdir -p "$fsa_work/tmp" "$fsa_work/logs"
export PYTHONPATH="$fsa_repo/python${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$fsa_work/tmp"
cd "$fsa_repo"
python3 -m pytest --override-ini addopts= \
  -o "cache_dir=$fsa_work/tmp/pytest_cache" \
  test/registered/unit/npu/test_glm53_fsa_contract.py \
  test/registered/unit/npu/test_external_sparse_planner.py \
  test/registered/unit/npu/test_decode_memory_budget.py \
  test/registered/unit/npu/test_fused_sparse_attention_dispatch.py \
  test/registered/unit/npu/test_speculative_sparse.py \
  test/registered/unit/npu/test_shared_host_kv.py \
  test/registered/unit/npu/test_offload_config.py \
  test/registered/unit/npu/test_registered_host_kv_adapter.py \
  test/registered/unit/npu/test_prefill_offload_slots.py \
  test/registered/unit/npu/test_layer_receive_protocol.py \
  test/registered/unit/npu/test_layer_staging.py \
  test/registered/unit/npu/test_layer_receive_group.py \
  test/registered/unit/npu/test_prefill_layer_transfer.py \
  test/registered/unit/npu/test_prefill_offload_adapter.py \
  test/registered/unit/npu/test_prefill_offload_runtime.py \
  test/registered/unit/npu/test_decode_host_kv.py "$@"
python3 -m pytest --override-ini addopts= \
  -o "cache_dir=$fsa_work/tmp/pytest_runtime_config" \
  test/registered/unit/npu/test_fsa_runtime_config.py "$@"
