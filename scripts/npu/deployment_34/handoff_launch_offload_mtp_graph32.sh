#!/usr/bin/env bash
# GLM-5.3-Flash TP8 integrated offload, verified BS11 graphs, NEXTN MTP3.
# Source: local sglang_remote.zip, SHA256 f524c79a9579e424b73a10bde8e5c6a080f7c549ef32725264dacea4a0e8627d.
# Run inside jzx_sglang_5.3_flash only when model execution is authorized.
# Usage: bash handoff_launch_offload_mtp_graph32.sh <new-run-directory>
# Integrated MTP preserves asynchronous Prefill with Host-visibility graph gates.
# CPU regressions cover graph gates, admission, stream ordering and ownership.

set -euo pipefail

RUN_ROOT=${1:?usage: handoff_launch_offload_mtp_graph32.sh <run-root>}
SRC=/home/j00628475/sglang_release_20260917_f524c79a/sglang_remote
WORK=/home/j00628475/sglang_release_20260917_f524c79a/runtime
MF_ROOT=/home/j00628475/memfabric_a5_fd334486_20260917_v1
EXT=${SGLANG_NPU_FSA_EXTENSION:-/home/j00628475/sglang_release_20260917_f524c79a/fsa-binding-dma-segments/libsgl_fsa_extension.so}
OPP=/home/j00628475/fsa_glm53_pd_20260915/attention-opp-v8/install/packages/vendors/sgl_fsa_a5
PORT=${FSA_INTEGRATED_PORT:-19234}
DIST_PORT=${FSA_INTEGRATED_DIST_PORT:-29234}
MODEL=/mnt/share/w00936111/weights/GLM-5.3-Flash

mkdir -p "$RUN_ROOT"
LOG=$RUN_ROOT/server.log
MANIFEST=$RUN_ROOT/manifest.json

echo "=== preflight ==="
if ps -eo args | grep -E 'sglang\.launch_server|sglang::' | grep -v grep >/dev/null; then
    echo "REFUSING: an sglang service is already running" >&2
    ps -eo pid,args | grep -E 'sglang\.launch_server|sglang::' | grep -v grep | head -3 >&2
    exit 3
fi
echo "[ok] no sglang service running"

for p in "$PORT" "$DIST_PORT"; do
    if (exec 3<>/dev/tcp/127.0.0.1/"$p") 2>/dev/null; then
        exec 3>&- 2>/dev/null || true
        echo "REFUSING: port $p in use" >&2; exit 4
    fi
done
echo "[ok] ports free"

for f in "$SRC/python/sglang/launch_server.py" "$EXT" "$OPP/op_api/lib" \
         "$SRC/scripts/npu/fsa_offload_env.sh" \
         "$MF_ROOT/source/src/smem/python/memfabric_hybrid/memfabric_hybrid/lib" \
         "$MF_ROOT/extend"; do
    [ -e "$f" ] || { echo "REFUSING: missing $f" >&2; exit 5; }
done
echo "[ok] artefacts present"

IDLE=$(npu-smi info 2>/dev/null | grep -c "No running processes found in NPU")
[ "$IDLE" -eq 8 ] || { echo "REFUSING: only $IDLE/8 NPUs idle" >&2; exit 6; }
echo "[ok] 8/8 NPUs idle"

# ---------------------------------------------------------------- environment
# mirrors fsa_pd_role.sh's environment block, then sources the offload switches
export SGLANG_FSA_WORK_DIR="$WORK"
export SGLANG_NPU_FSA_EXTENSION="$EXT"
export SGLANG_FSA_OPP_VENDOR="$OPP"
export SGLANG_CACHE_DIR="$WORK/tmp/sglang_cache"
export TORCH_EXTENSIONS_DIR="$WORK/tmp/torch_extensions"
export SGLANG_NPU_DECODE_HOST_KV_BYTES=137438953472     # 128 GiB shared Host KV
export SGLANG_NPU_PREFILL_LAYER_SLOTS=2
export SGLANG_NPU_LAYER_STAGING_BYTES=0
export SGLANG_NPU_FSA_RESIDENT_CAPACITY=4096
export SGLANG_NPU_SHARED_HOST_KV=1
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_NPU_PROFILING=0
export SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=1
export SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD=1
export SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION=1
export SGLANG_NPU_FUSED_SPARSE_ATTENTION_MODE=fused

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
export FSA_MTP_STEPS=3
export ASCEND_MF_LOG_LEVEL=1
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES ASCEND_NPU_PHY_ID
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
unset HCCL_IF_IP HCCL_SOCKET_FAMILY RANK_TABLE_FILE

# shellcheck disable=SC1090
source "$SRC/scripts/npu/fsa_offload_env.sh" integrated

# Select the matching MemFabric package after the generic environment script.
export PYTHONPATH="$MF_ROOT/source/src/smem/python/memfabric_hybrid${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$MF_ROOT/source/src/smem/python/memfabric_hybrid/memfabric_hybrid/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MEMFABRIC_HYBRID_EXTEND_LIB_PATH="$MF_ROOT/extend"

# integrated role has no disaggregation store
unset ASCEND_MF_STORE_URL MF_CONFIG_STORE_URL ASCEND_MF_TRANSFER_PROTOCOL \
      MF_HYBRM_RDMA_FORCE_UNREGISTERED || true

echo "offload switches:"
env | grep -E '^SGLANG_NPU_(ENABLE|DECODE|PREFILL|FUSED|SHARED|FSA)' | sort | sed 's/^/  /'

# ---------------------------------------------------------------------- launch
ARGS=(
    --model-path "$MODEL"
    --served-model-name glm53flash
    --host 127.0.0.1 --tp-size 8 --dp-size 1 --nnodes 1
    --attention-backend ascend --device npu
    --quantization fp8 --kv-cache-dtype bf16
    --trust-remote-code --disable-radix-cache --skip-server-warmup
    --enable-metrics
    --mem-fraction-static 0.85
    --context-length 1048576
    --page-size 128
    --cuda-graph-backend-decode full
    # Verified with 12 requests of 256K input + 1K output: Decode11 on graph,
    # the 12th request queues; all complete with MTP3 and zero retractions.
    --cuda-graph-bs-decode 1 2 4 8 11
    --cuda-graph-max-bs-decode 11
    --speculative-algorithm NEXTN
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --chunked-prefill-size 8192
    --max-prefill-tokens 8192
    --watchdog-timeout 1200
    --max-running-requests 64
    --base-gpu-id 0
    --port "$PORT"
    --nccl-port "$DIST_PORT"
)
# --max-total-tokens deliberately omitted (handoff: uncapped automatic pool)

cd "$SRC"
export PYTHONPATH="$SRC/python${PYTHONPATH:+:$PYTHONPATH}"
echo "COMMAND: python3 -m sglang.launch_server ${ARGS[*]}"

nohup python3 -m sglang.launch_server "${ARGS[@]}" >"$LOG" 2>&1 &
SERVER_PID=$!
sleep 6
START_TICKS=$(awk '{print $22}' /proc/"$SERVER_PID"/stat 2>/dev/null || echo "")
cat >"$MANIFEST" <<JSON
{
  "run_root": "$RUN_ROOT",
  "role": "integrated",
  "source": "$SRC",
  "port": $PORT,
  "dist_port": $DIST_PORT,
  "server_pid": $SERVER_PID,
  "start_ticks": "$START_TICKS",
  "enable_metrics": true,
  "cuda_graph_bs_decode": [1, 2, 4, 8, 11],
  "cuda_graph_max_bs_decode": 11,
  "speculative_algorithm": "NEXTN",
  "speculative_num_steps": 3,
  "speculative_eagle_topk": 1,
  "speculative_num_draft_tokens": 4,
  "context_length": 1048576,
  "max_running_requests": 64,
  "max_total_tokens": "omitted",
  "host_kv_bytes": 137438953472,
  "extension": "$EXT",
  "opp_vendor": "$OPP"
}
JSON
echo "server pid=$SERVER_PID start_ticks=$START_TICKS"
echo "manifest: $MANIFEST"

echo "=== waiting for health (up to 40 min) ==="
for i in $(seq 1 480); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" || true)
    if [ "$code" = "200" ]; then
        echo "HEALTH_OK after $((i*5))s"
        curl -s -o /dev/null -w 'metrics_http=%{http_code}\n' "http://127.0.0.1:$PORT/metrics"
        exit 0
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "service exited early" >&2; tail -30 "$LOG" >&2; exit 7; }
    sleep 5
done
echo "TIMEOUT waiting for health" >&2
tail -30 "$LOG" >&2
exit 8
