#!/usr/bin/env bash
set -eo pipefail

fsa_role=${1:?Specify prefill, decode, router or integrated}
fsa_action=${2:-launch}
fsa_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export SGLANG_FSA_WORK_DIR=${SGLANG_FSA_WORK_DIR:-/home/j00628475/fsa_glm53_a5_20260915}
export SGLANG_CACHE_DIR="$SGLANG_FSA_WORK_DIR/tmp/sglang_cache"
export TORCH_EXTENSIONS_DIR="$SGLANG_FSA_WORK_DIR/tmp/torch_extensions"
export SGLANG_NPU_FSA_EXTENSION=${SGLANG_NPU_FSA_EXTENSION:-$SGLANG_FSA_WORK_DIR/native-build-v5/libsgl_fsa_extension.so}
export SGLANG_FSA_OPP_VENDOR=${SGLANG_FSA_OPP_VENDOR:-$SGLANG_FSA_WORK_DIR/attention-opp-v8/install/packages/vendors/sgl_fsa_a5}
export SGLANG_NPU_PREFILL_HOST_KV_BYTES=${SGLANG_NPU_PREFILL_HOST_KV_BYTES:-17179869184}
export SGLANG_NPU_DECODE_HOST_KV_BYTES=${SGLANG_NPU_DECODE_HOST_KV_BYTES:-17179869184}
export SGLANG_NPU_LAYER_STAGING_BYTES=${SGLANG_NPU_LAYER_STAGING_BYTES:-67108864}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=120 HCCL_BUFFSIZE=256
export HCCL_HOST_SOCKET_PORT_RANGE=auto HCCL_OP_EXPANSION_MODE=AIV
export DEEPEP_HCCL_BUFFSIZE=2048 ACL_DEVICE_SYNC_TIMEOUT=120
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True STREAMS_PER_DEVICE=32
export ASCEND_USE_FIA=1 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0 SGLANG_ENABLE_JIT_DEEPGEMM=False
export SGLANG_ENABLE_SPEC_V2=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600
export ASCEND_MF_STORE_URL="tcp://127.0.0.1:${FSA_STORE_PORT:-24681}"
export MF_CONFIG_STORE_URL="$ASCEND_MF_STORE_URL"
export ASCEND_MF_TRANSFER_PROTOCOL=${ASCEND_MF_TRANSFER_PROTOCOL:-sdma}
export MF_HYBM_RDMA_FORCE_UNREGISTERED=${MF_HYBM_RDMA_FORCE_UNREGISTERED:-1}
export ASCEND_MF_LOG_LEVEL=1
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES ASCEND_NPU_PHY_ID
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
unset HCCL_IF_IP HCCL_SOCKET_FAMILY RANK_TABLE_FILE

fsa_env_role=$fsa_role
if [ "$fsa_role" = router ]; then fsa_env_role=decode; fi
source "$fsa_repo/scripts/npu/fsa_offload_env.sh" "$fsa_env_role"
fsa_role=$1
fsa_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fsa_model=${FSA_MODEL_PATH:-/mnt/share/w00936111/weights/GLM-5.3-Flash}
fsa_module=sglang.launch_server
fsa_tp=4
if [ "$fsa_role" = integrated ] || [ "$fsa_role" = resident ]; then fsa_tp=8; fi
fsa_args=(--model-path "$fsa_model" --served-model-name glm53flash
    --host 127.0.0.1 --tp-size "$fsa_tp" --dp-size 1 --nnodes 1
    --attention-backend ascend --device npu --quantization fp8 --kv-cache-dtype bf16
    --trust-remote-code --disable-radix-cache --skip-server-warmup
    --enable-metrics
    --mem-fraction-static "${FSA_MEM_FRACTION:-0.85}"
    --context-length "${FSA_CONTEXT_LENGTH:-32768}"
    --page-size 128 --cuda-graph-bs-decode 1 2 4
    --chunked-prefill-size "${FSA_PREFILL_CHUNK:-8192}"
    --max-prefill-tokens "${FSA_PREFILL_CHUNK:-8192}"
    --watchdog-timeout "${FSA_WATCHDOG_TIMEOUT:-1200}")
if ! [[ "${FSA_MAX_RUNNING_REQUESTS:-1}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FSA_MAX_RUNNING_REQUESTS must be a positive integer" >&2
    exit 2
fi
fsa_args+=(--max-running-requests "${FSA_MAX_RUNNING_REQUESTS:-1}")
if [ "${FSA_MAX_TOTAL_TOKENS:-32768}" != auto ]; then
    fsa_args+=(--max-total-tokens "${FSA_MAX_TOTAL_TOKENS:-32768}")
fi
case "$fsa_role" in
    integrated|resident)
        unset ASCEND_MF_STORE_URL MF_CONFIG_STORE_URL
        fsa_args+=(--base-gpu-id 0 --port "${FSA_INTEGRATED_PORT:-33181}"
            --nccl-port "${FSA_INTEGRATED_DIST_PORT:-29611}") ;;
    prefill)
        fsa_args+=(--disaggregation-mode prefill --disaggregation-transfer-backend ascend --base-gpu-id 0
            --port "${FSA_PREFILL_PORT:-31181}" --nccl-port "${FSA_PREFILL_DIST_PORT:-29581}"
            --disaggregation-bootstrap-port "${FSA_BOOTSTRAP_PORT:-8991}") ;;
    decode)
        fsa_args+=(--disaggregation-mode decode --disaggregation-transfer-backend ascend --base-gpu-id 4
            --port "${FSA_DECODE_PORT:-32181}" --nccl-port "${FSA_DECODE_DIST_PORT:-29591}") ;;
    router)
        fsa_module=sglang_router.launch_router
        fsa_args=(--host 127.0.0.1 --port "${FSA_ROUTER_PORT:-6681}" --pd-disaggregation
            --prefill "http://127.0.0.1:${FSA_PREFILL_PORT:-31181}" "${FSA_BOOTSTRAP_PORT:-8991}"
            --decode "http://127.0.0.1:${FSA_DECODE_PORT:-32181}") ;;
    *) echo "Unsupported role: $fsa_role" >&2; exit 2 ;;
esac
if [ "$fsa_role" != router ] && [ "${FSA_MTP_STEPS:-0}" -gt 0 ]; then
    fsa_args+=(--speculative-algorithm NEXTN --speculative-num-steps "$FSA_MTP_STEPS"
        --speculative-eagle-topk 1 --speculative-num-draft-tokens "$((FSA_MTP_STEPS + 1))")
fi
cd "$fsa_repo"
printf 'FSA_ROLE=%s FSA_SOURCE=%s SHARED_HOST=%s\n' "$fsa_role" "$fsa_repo" "$SGLANG_NPU_SHARED_HOST_KV"
printf 'COMMAND: python3 -m %s' "$fsa_module"
printf ' %q' "${fsa_args[@]}"
printf '\n'
if [ "$fsa_action" = check ]; then
    if [ "$fsa_role" = router ]; then
        python3 -m "$fsa_module" --help >/dev/null
    else
        python3 - "${fsa_args[@]}" <<'PY'
import argparse
import pathlib
import sys

import sgl_kernel_npu
import sglang
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.fsa_native import native_ops
from sglang.srt.server_args import ServerArgs

parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
args = parser.parse_args(sys.argv[1:])
assert pathlib.Path(args.model_path, "config.json").is_file()
native_ops()
print("CHECK_ONLY", sglang.__file__, sgl_kernel_npu.__file__)
print("shared_host", envs.SGLANG_NPU_SHARED_HOST_KV.get())
PY
    fi
    exit 0
fi
exec python3 -m "$fsa_module" "${fsa_args[@]}"
