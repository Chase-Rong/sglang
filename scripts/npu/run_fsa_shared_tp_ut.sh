#!/usr/bin/env bash
set -eo pipefail

fsa_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fsa_work=${SGLANG_FSA_WORK_DIR:?Set the assigned work directory}
: "${SGLANG_NPU_FSA_EXTENSION:?Set the validated extension}"
: "${SGLANG_FSA_OPP_VENDOR:?Set the validated OPP vendor}"
: "${SGLANG_FSA_TEST_DEVICES:?Set comma-separated idle physical NPU IDs}"
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHONPATH="$fsa_repo/python${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_CUSTOM_OPP_PATH="$SGLANG_FSA_OPP_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$SGLANG_FSA_OPP_VENDOR/op_api/lib:/usr/local/python3.12.13/lib:$LD_LIBRARY_PATH"
export TMPDIR="$fsa_work/tmp"
export ASCEND_WORK_PATH="$fsa_work/tmp/shared_tp_ascend"
export ASCEND_CACHE_PATH="$fsa_work/tmp/shared_tp_cache"
export SGLANG_FSA_TEST_SHARED_TP=1
export TASK_QUEUE_ENABLE=0
export HCCL_BUFFSIZE=64
export DEEPEP_HCCL_BUFFSIZE=64
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=30
export OMP_NUM_THREADS=1
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES
mkdir -p "$TMPDIR" "$ASCEND_WORK_PATH" "$ASCEND_CACHE_PATH"
fsa_world=$(python3 - <<'PY'
import os
import re
import subprocess

devices = [int(value) for value in os.environ["SGLANG_FSA_TEST_DEVICES"].split(",")]
if len(devices) < 2 or len(set(devices)) != len(devices):
    raise SystemExit("Specify at least two distinct NPU IDs")
report = subprocess.check_output(["npu-smi", "info"], text=True)
for device in devices:
    idle = f"No running processes found in NPU {device} " in report
    healthy = re.search(rf"\|\s*{device}\s*\|\s*Ascend950\w*\s*\|\s*OK\s*\|", report)
    if not idle or healthy is None:
        raise SystemExit(f"NPU {device} is occupied, unhealthy or unverified")
print(len(devices))
PY
)
cd "$fsa_repo"
fsa_port=$(python3 -c 'import socket; sock=socket.socket(); sock.bind(("127.0.0.1",0)); print(sock.getsockname()[1]); sock.close()')
python3 -m torch.distributed.run --nnodes=1 --nproc-per-node="$fsa_world" \
    --master-addr=127.0.0.1 --master-port="$fsa_port" \
    -m pytest -o addopts= -o "cache_dir=$TMPDIR/pytest_shared_tp" \
    test/registered/unit/npu/test_fsa_shared_tp_npu.py -x -v "$@"
