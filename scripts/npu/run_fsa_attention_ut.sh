#!/usr/bin/env bash
set -eo pipefail

fsa_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fsa_work=${SGLANG_FSA_WORK_DIR:?Set the assigned isolated work directory}
fsa_vendor=${SGLANG_FSA_OPP_VENDOR:?Set the matching A5 OPP vendor directory}
: "${SGLANG_NPU_FSA_EXTENSION:?Set the standalone helper extension}"
: "${SGLANG_FSA_TEST_DEVICE_ID:?Select an idle healthy physical device first}"
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="$fsa_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$fsa_vendor/op_api/lib:/usr/local/python3.12.13/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$fsa_repo/python${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$fsa_work/tmp"
export ASCEND_WORK_PATH="$fsa_work/tmp/ascend"
export ASCEND_CACHE_PATH="$fsa_work/tmp/ascend_cache"
export SGLANG_FSA_TEST_ATTENTION=1
export TASK_QUEUE_ENABLE=0
export ACL_DEVICE_SYNC_TIMEOUT=10
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$ASCEND_WORK_PATH" "$ASCEND_CACHE_PATH"
cd "$fsa_repo"
python3 - <<'PY'
import os
import re
import subprocess

device = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
report = subprocess.check_output(["npu-smi", "info"], text=True)
idle = f"No running processes found in NPU {device} " in report
healthy = re.search(rf"\|\s*{device}\s*\|\s*Ascend950\w*\s*\|\s*OK\s*\|", report)
if not idle or healthy is None:
    raise SystemExit(f"NPU {device} is occupied, unhealthy, or could not be verified; UT not started")
print(f"Preflight: NPU {device} is healthy and has no reported processes", flush=True)
PY
python3 -m pytest -o addopts= -o "cache_dir=$TMPDIR/pytest_attention" \
    test/registered/unit/npu/test_fsa_attention_a5.py \
    test/registered/unit/npu/test_fsa_pool_lifecycle_npu.py "$@"
