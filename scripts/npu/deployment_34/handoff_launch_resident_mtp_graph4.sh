#!/usr/bin/env bash
# GLM-5.3-Flash TP8 resident baseline, decode graphs through BS8, NEXTN MTP3.
# Historical filename retained; effective graph sizes are recorded in the manifest.
# Usage: bash handoff_launch_resident_mtp_graph4.sh <run-root>
set -euo pipefail
RUN_ROOT=${1:?usage: handoff_launch_resident_mtp_graph4.sh <run-root>}
SRC=/home/j00628475/sglang_release_20260917_f524c79a/sglang_remote
WORK=/home/j00628475/sglang_release_20260917_f524c79a/runtime_resident
OPP=/home/j00628475/fsa_glm53_pd_20260915/attention-opp-v8/install/packages/vendors/sgl_fsa_a5
PORT=${FSA_RESIDENT_PORT:-19244}; DIST_PORT=${FSA_RESIDENT_DIST_PORT:-29244}
MODEL=/mnt/share/w00936111/weights/GLM-5.3-Flash
mkdir -p "$RUN_ROOT"; LOG=$RUN_ROOT/server.log; MANIFEST=$RUN_ROOT/manifest.json
if ps -eo args | grep -E 'sglang\.launch_server|sglang::' | grep -v grep >/dev/null; then exit 3; fi
for p in "$PORT" "$DIST_PORT"; do
    if (exec 3<>/dev/tcp/127.0.0.1/"$p") 2>/dev/null; then exec 3>&-; exit 4; fi
done
IDLE=$(npu-smi info 2>/dev/null | grep -c "No running processes found in NPU")
[ "$IDLE" -eq 8 ] || exit 6
source /usr/local/Ascend/cann-9.1.0/set_env.sh
[ -f /usr/local/memfabric_hybrid/set_env.sh ] && source /usr/local/memfabric_hybrid/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
for vendor in customize custom_transformer; do
    vendor_env="/usr/local/Ascend/cann-9.1.0/opp/vendors/$vendor/bin/set_env.bash"
    if [ -f "$vendor_env" ]; then source "$vendor_env"; fi
done
export ASCEND_CUSTOM_OPP_PATH="$OPP${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$OPP/op_api/lib:/usr/local/python3.12.13/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_FSA_WORK_DIR="$WORK" SGLANG_FSA_OPP_VENDOR="$OPP"
export SGLANG_CACHE_DIR=/home/j00628475/sglang_release_20260917_f524c79a/runtime/tmp/sglang_cache
export TORCH_EXTENSIONS_DIR=/home/j00628475/sglang_release_20260917_f524c79a/runtime/tmp/torch_extensions
export TMPDIR="$WORK/tmp" ASCEND_WORK_PATH="$WORK/tmp/ascend" ASCEND_CACHE_PATH="$WORK/tmp/ascend_cache"
export SGLANG_NPU_SHARED_HOST_KV=0 SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=0
export SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD=0 SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION=0
export SGLANG_NPU_DECODE_HOST_KV_BYTES=0 SGLANG_NPU_PREFILL_HOST_KV_BYTES=0 SGLANG_NPU_LAYER_STAGING_BYTES=0
export SGLANG_NPU_USE_MLAPO=0 SGLANG_NPU_PROFILING=0 FSA_MTP_STEPS=3
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-lo} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=120 HCCL_BUFFSIZE=256 HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_OP_EXPANSION_MODE=AIV DEEPEP_HCCL_BUFFSIZE=2048 ACL_DEVICE_SYNC_TIMEOUT=120
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True STREAMS_PER_DEVICE=32 ASCEND_USE_FIA=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35 SGLANG_ENABLE_SPEC_V2=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_ENABLE_JIT_DEEPGEMM=False ASCEND_MF_LOG_LEVEL=1
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES ASCEND_NPU_PHY_ID http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
unset HCCL_IF_IP HCCL_SOCKET_FAMILY RANK_TABLE_FILE ASCEND_MF_STORE_URL MF_CONFIG_STORE_URL ASCEND_MF_TRANSFER_PROTOCOL MF_HYBM_RDMA_FORCE_UNREGISTERED || true
mkdir -p "$TMPDIR" "$ASCEND_WORK_PATH" "$ASCEND_CACHE_PATH"
ARGS=(--model-path "$MODEL" --served-model-name glm53flash --host 127.0.0.1 --tp-size 8 --dp-size 1 --nnodes 1
    --attention-backend ascend --device npu --quantization fp8 --kv-cache-dtype bf16 --trust-remote-code
    --disable-radix-cache --skip-server-warmup --enable-metrics --mem-fraction-static 0.85 --context-length 1048576
    --page-size 128 --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 4 8 --cuda-graph-max-bs-decode 8
    --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
    --chunked-prefill-size 8192 --max-prefill-tokens 8192 --watchdog-timeout 1200 --max-running-requests 64
    --base-gpu-id 0 --port "$PORT" --nccl-port "$DIST_PORT")
cd "$SRC"; export PYTHONPATH="$SRC/python${PYTHONPATH:+:$PYTHONPATH}"
echo "COMMAND: python3 -m sglang.launch_server ${ARGS[*]}"
nohup python3 -m sglang.launch_server "${ARGS[@]}" >"$LOG" 2>&1 & SERVER_PID=$!; sleep 6
START_TICKS=$(awk '{print $22}' /proc/"$SERVER_PID"/stat 2>/dev/null || echo "")
cat >"$MANIFEST" <<JSON
{"run_root":"$RUN_ROOT","role":"resident","source":"$SRC","port":$PORT,"dist_port":$DIST_PORT,"server_pid":$SERVER_PID,"start_ticks":"$START_TICKS","cuda_graph_bs_decode":[1,2,4,8],"cuda_graph_max_bs_decode":8,"speculative_algorithm":"NEXTN","speculative_num_steps":3,"speculative_num_draft_tokens":4,"context_length":1048576,"max_running_requests":64,"max_total_tokens":"omitted","integrated_offload":0,"shared_host_kv":0}
JSON
for i in $(seq 1 480); do code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" || true); if [ "$code" = 200 ]; then exit 0; fi; kill -0 "$SERVER_PID" 2>/dev/null || { tail -40 "$LOG" >&2; exit 7; }; sleep 5; done
tail -40 "$LOG" >&2; exit 8
