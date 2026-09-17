#!/bin/bash

# ===== CPU 调优 =====
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# ===== Cleanup =====
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
pkill -9 python  2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
pkill -9 VLLM   2>/dev/null || true
# ===== 使用你自己的 sglang 源码 =====
export PYTHONPATH=`pwd`/python:$PYTHONPATH

# ===== 屏蔽deep_gemm 调用(新添加) =====
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0


# ===== 通信/算子环境变量 =====
export DEEPEP_HCCL_BUFFSIZE=2048
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=68
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_DEVICE_SYNC_TIMEOUT=60
# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
# [FIA] / [MLAPO] / [DEEPEP]
export ASCEND_USE_FIA=1
# 修改部分，这里设置为0，跳过问题4
export SGLANG_NPU_USE_MLAPO=0
# export SGLANG_NPU_USE_MLAPO=1
#export SGLANG_NPU_GLM_NEXTN_BF16_KV_CACHE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=35
# [MTP / Speculative]
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export TRANSFORMERS_VERBOSITY=error
# 单机：随便用一个稳定端口做分布式初始化即可
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export GLOO_SOCKET_IFNAME=data0.3001   # 按本机实际网卡调整
export HCCL_SOCKET_IFNAME=data0.3001   # 按本机实际网卡调整
unset HCCL_IF_IP 2>/dev/null || true
unset HCCL_SOCKET_FAMILY 2>/dev/null || true
unset RANK_TABLE_FILE 2>/dev/null || true
# ===== 模型与地址配置 =====
MODEL_PATH=/mnt/share/w00936111/weights/GLM-5.3-Flash
SERVED_MODEL_NAME=glm53flash
SERVER_HOST=0.0.0.0           # 监听地址，按需改
SERVER_PORT=8810
DIST_INIT_ADDR=127.0.0.1:5569 # 单机可不指定真实 IP
NNODES=1
NODE_RANK=0
TP_SIZE=8
DP_SIZE=1
echo "========================================"
echo "Launching GLM-5.3-Flash single-node"
echo "tp-size : ${TP_SIZE}   dp-size : ${DP_SIZE}"
echo "port    : ${SERVER_PORT}"
echo "========================================"
export ASCEND_LAUNCH_BLOCKING=1
# ===== 启动（用自己的源码，绕开镜像内旧 sglang）=====
python3 -m sglang.launch_server --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --tp-size "${TP_SIZE}" \
  --trust-remote-code \
  --attention-backend ascend \
  --device npu \
  --cuda-graph-bs 16 \
  --watchdog-timeout 9000 \
  --max-running-requests 52 \
  --mem-fraction-static 0.84 \
  --quantization fp8 \
  --max-prefill-tokens 2048000 \
  --chunked-prefill-size 16384 \
  --kv-cache-dtype "bf16" \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --enable-metrics
#  --speculative-draft-kv-cache-dtype bf16 \
#  --speculative-algorithm NEXTN --speculative-draft-model-quantization fp8 \
#  --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
#  --dp "${DP_SIZE}" \
#  --enable-dp-attention \
#  --enable-dp-lm-head \
#  --load-balance-method round_robin \
  
