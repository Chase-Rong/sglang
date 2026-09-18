"""Minimal production-DMA reproducer across a MemFabric 32 GiB shared slice."""
import ctypes
import json
import os
import torch
import torch_npu
import memfabric_hybrid as fabric
from sglang.srt.hardware_backend.npu.fsa_native import native_ops

torch.npu.set_device(0)
MiB = 1 << 20
GiB = 1 << 30
config = fabric.offload.OffloadConfig()
config.device_id = 0
config.rank_id = 0
config.world_size = 1
config.scene = fabric.offload.Scene.SHARED
config.reserve_size = config.alloc_size = 33 * GiB
assert fabric.offload.initialize(config) == 0
padding = fabric.offload.malloc(32 * GiB - 4 * MiB)
pointer = fabric.offload.malloc(16 * MiB)
assert padding and pointer
ops = native_ops()
host, dva = ops.memfabric_region(pointer, (16 * MiB // 1024, 512), torch.empty(0, dtype=torch.bfloat16), 0, True)
print(json.dumps({'padding': hex(padding), 'host': hex(pointer), 'dva': hex(dva),
                  'boundary': hex((dva // (32 * GiB) + 1) * (32 * GiB))}), flush=True)
pattern = ((torch.arange(host.numel()).reshape(host.shape) % 97).to(torch.bfloat16) / 128)
source = pattern.to('npu:0')
restored = torch.zeros_like(source)
torch.npu.synchronize()
acl = ctypes.CDLL('libascendcl.so')
acl.aclrtMemGetAddressRange.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
for offset in (0, 4 * MiB, 8 * MiB):
    base, length = ctypes.c_void_p(), ctypes.c_size_t()
    rc = acl.aclrtMemGetAddressRange(dva + offset, ctypes.byref(base), ctypes.byref(length))
    print(json.dumps({'query': hex(dva + offset), 'return': rc, 'base': hex(base.value or 0), 'bytes': length.value}), flush=True)
# Within-slice control followed by the same 8 MiB crossing that failed in serving.
for label, plan in [('within', [[128, 128]]), ('crossing', [[128, 8192]]),
                    ('ends_at_boundary', [[128, 4096 - 128]]),
                    ('starts_at_boundary', [[4096, 8192]]),
                    ('multiple_runs', [[128, 8192], [10000, 1024]])]:
    print('BEGIN ' + label, flush=True)
    runs = torch.tensor(plan, dtype=torch.int64)
    ops.prefill_dma_copy(source, host, dva, runs, True)
    ops.prefill_dma_copy(restored, host, dva, runs, False)
    torch.npu.synchronize()
    for start, count in plan:
        torch.testing.assert_close(restored[start:start + count].cpu(), pattern[start:start + count], atol=0, rtol=0)
        torch.testing.assert_close(host[start:start + count], pattern[start:start + count], atol=0, rtol=0)
    print('PASS ' + label, flush=True)

copy_stream = torch.npu.Stream()
compute_stream = torch.npu.current_stream()
runs = torch.tensor([[128, 8192]], dtype=torch.int64)
for shift in (1, 2):
    source.copy_((pattern + shift).to(source.device))
    ready, saved = torch.npu.Event(), torch.npu.Event()
    ready.record(compute_stream)
    with torch.npu.stream(copy_stream):
        copy_stream.wait_event(ready)
        ops.prefill_dma_copy(source, host, dva, runs, True)
        saved.record(copy_stream)
    compute_stream.wait_event(saved)
    ops.prefill_dma_copy(restored, host, dva, runs, False)
    torch.npu.synchronize()
    torch.testing.assert_close(restored[128:8320].cpu(), (pattern + shift)[128:8320], atol=0, rtol=0)
print('PASS crossing stream events across generations', flush=True)

graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    ops.prefill_dma_copy(source, host, dva, runs, True)
    ops.prefill_dma_copy(restored, host, dva, runs, False)
source.copy_((pattern + 3).to(source.device))
graph.replay()
torch.npu.synchronize()
torch.testing.assert_close(restored[128:8320].cpu(), (pattern + 3)[128:8320], atol=0, rtol=0)
graph.reset()
print('PASS crossing graph replay with new data', flush=True)
ops.unregister_memfabric_region(host, 0)
fabric.offload.free(pointer)
fabric.offload.free(padding)
fabric.offload.uninitialize()
