#!/usr/bin/env bash

fsa_role=${1:?Source this script with prefill or decode}
fsa_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fsa_work=${SGLANG_FSA_WORK_DIR:?Set the assigned isolated work directory}
: "${SGLANG_NPU_FSA_EXTENSION:?Set the validated standalone extension path}"
: "${SGLANG_FSA_OPP_VENDOR:?Set the validated A5 OPP vendor directory}"
if [ ! -f "$SGLANG_NPU_FSA_EXTENSION" ] || [ ! -d "$SGLANG_FSA_OPP_VENDOR/op_api/lib" ]; then
    echo "FSA extension or OPP vendor is missing" >&2
    return 1
fi
source /usr/local/Ascend/cann-9.1.0/set_env.sh
if [ -f /usr/local/memfabric_hybrid/set_env.sh ]; then
    source /usr/local/memfabric_hybrid/set_env.sh
fi
if [ "$fsa_role" != integrated ] && [ "$fsa_role" != resident ] && ! python3 -c 'from memfabric_hybrid import TransferEngine'; then
    echo "PD offload requires an importable container MemFabric TransferEngine" >&2
    return 1
fi
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
for fsa_existing_vendor in customize custom_transformer; do
    fsa_existing_env="/usr/local/Ascend/cann-9.1.0/opp/vendors/$fsa_existing_vendor/bin/set_env.bash"
    if [ -f "$fsa_existing_env" ]; then
        source "$fsa_existing_env"
    fi
done
export ASCEND_CUSTOM_OPP_PATH="$SGLANG_FSA_OPP_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$SGLANG_FSA_OPP_VENDOR/op_api/lib:/usr/local/python3.12.13/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$fsa_repo/python${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$fsa_work/tmp"
export ASCEND_WORK_PATH="$fsa_work/tmp/ascend"
export ASCEND_CACHE_PATH="$fsa_work/tmp/ascend_cache"
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_NPU_PROFILING=0
export SGLANG_NPU_FSA_RESIDENT_CAPACITY=${SGLANG_NPU_FSA_RESIDENT_CAPACITY:-4096}
export SGLANG_NPU_SHARED_HOST_KV=${SGLANG_NPU_SHARED_HOST_KV:-1}
export SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=0
case "$fsa_role" in
    resident)
        export SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD=0
        export SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION=0
        export SGLANG_NPU_DECODE_HOST_KV_BYTES=0
        export SGLANG_NPU_PREFILL_HOST_KV_BYTES=0
        export SGLANG_NPU_LAYER_STAGING_BYTES=0
        ;;
    integrated)
        : "${SGLANG_NPU_DECODE_HOST_KV_BYTES:?Set the shared Prefill/Decode Host budget}"
        export SGLANG_NPU_DECODE_HOST_KV_BYTES
        export SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=1
        export SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD=1
        export SGLANG_NPU_PREFILL_LAYER_SLOTS=${SGLANG_NPU_PREFILL_LAYER_SLOTS:-2}
        export SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION=1
        export SGLANG_NPU_LAYER_STAGING_BYTES=0
        export SGLANG_NPU_FUSED_SPARSE_ATTENTION_MODE=${SGLANG_NPU_FUSED_SPARSE_ATTENTION_MODE:-fused}
        ;;
    prefill)
        : "${SGLANG_NPU_PREFILL_HOST_KV_BYTES:?Set the Prefill Host backing budget}"
        export SGLANG_NPU_PREFILL_HOST_KV_BYTES
        export SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD=1
        export SGLANG_NPU_PREFILL_LAYER_SLOTS=${SGLANG_NPU_PREFILL_LAYER_SLOTS:-2}
        export SGLANG_NPU_DECODE_HOST_KV_BYTES=0
        export SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION=0
        ;;
    decode)
        : "${SGLANG_NPU_DECODE_HOST_KV_BYTES:?Set the Decode Host backing budget}"
        : "${SGLANG_NPU_LAYER_STAGING_BYTES:?Set the per-rank aligned NPU staging budget}"
        export SGLANG_NPU_DECODE_HOST_KV_BYTES SGLANG_NPU_LAYER_STAGING_BYTES
        export SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD=0
        export SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION=1
        export SGLANG_NPU_FUSED_SPARSE_ATTENTION_MODE=${SGLANG_NPU_FUSED_SPARSE_ATTENTION_MODE:-fused}
        ;;
    *) echo "Role must be prefill, decode, integrated or resident" >&2; return 1 ;;
esac
mkdir -p "$TMPDIR" "$ASCEND_WORK_PATH" "$ASCEND_CACHE_PATH"
unset fsa_role fsa_repo fsa_work fsa_existing_vendor fsa_existing_env
