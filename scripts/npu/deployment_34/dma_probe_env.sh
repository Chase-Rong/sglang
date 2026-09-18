#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
MF_ROOT=/home/j00628475/memfabric_a5_fd334486_20260917_v1
RELEASE=/home/j00628475/sglang_release_20260917_f524c79a
export PYTHONPATH="$RELEASE/sglang_remote/python:$MF_ROOT/source/src/smem/python/memfabric_hybrid:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$MF_ROOT/source/src/smem/python/memfabric_hybrid/memfabric_hybrid/lib:/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
export MEMFABRIC_HYBRID_EXTEND_LIB_PATH="$MF_ROOT/extend"
export SGLANG_NPU_FSA_EXTENSION=${SGLANG_NPU_FSA_EXTENSION:-$MF_ROOT/fsa-binding-production/libsgl_fsa_extension.so}
export ASCEND_WORK_PATH="$RELEASE/dma_probe_logs"
mkdir -p "$ASCEND_WORK_PATH"
if [ "$#" -gt 0 ]; then
    exec "$@"
fi
exec python3 "$RELEASE/dma_boundary_probe.py"
