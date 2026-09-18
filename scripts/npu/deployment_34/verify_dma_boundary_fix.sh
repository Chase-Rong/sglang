#!/usr/bin/env bash
set -eo pipefail
RELEASE=/home/j00628475/sglang_release_20260917_f524c79a
export SGLANG_NPU_FSA_EXTENSION=$RELEASE/fsa-binding-dma-segments/libsgl_fsa_extension.so
test -f "$SGLANG_NPU_FSA_EXTENSION"
cd "$RELEASE/sglang_remote"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
if [ "${1:-}" != "--tp-only" ]; then
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest -q \
    test/registered/unit/npu/test_prefill_dma_plan.py
bash "$RELEASE/dma_probe_env.sh"
export SGLANG_UNIFIED_HOST_TEST=1 SGLANG_FSA_TEST_DEVICE_ID=0
bash "$RELEASE/dma_probe_env.sh" python3 -m pytest -q \
    test/registered/unit/npu/test_prefill_dma_npu.py::test_local_production_dma
fi
export SGLANG_UNIFIED_HOST_TEST=1
export SGLANG_FSA_TEST_SHARED_TP=1 SGLANG_FSA_TEST_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=120 HCCL_OP_EXPANSION_MODE=AIV
export HCCL_HOST_SOCKET_PORT_RANGE=auto
bash "$RELEASE/dma_probe_env.sh" python3 -m torch.distributed.run \
    --master-addr=127.0.0.1 --master-port=29585 --nnodes=1 --nproc_per_node=8 -m pytest -q \
    test/registered/unit/npu/test_prefill_dma_npu.py::test_shared_production_dma
echo 'PASS DMA boundary hardware verification'
