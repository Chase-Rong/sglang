"""DMA and AIV must consume the same physical shared Host allocation."""

import os
import time

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_UNIFIED_HOST_TEST") != "1", reason="requires idle NPU"
)


def candidate_type():
    if os.environ.get("SGLANG_UNIFIED_HOST_BACKEND") == "memfabric":
        from memfabric_host_lab import MemFabricHostRegion

        return MemFabricHostRegion
    from unified_host_lab import UnifiedHostRegion

    return UnifiedHostRegion


def test_unified_host_pool_and_aiv():
    import torch_npu
    from sglang.srt.hardware_backend.npu.fsa_native import (
        copy_rows_from_registered_host,
        copy_rows_to_registered_host,
    )

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch_npu.npu.set_device(device_id)
    region_type = candidate_type()
    rows, width = 256, 512
    reference = (torch.arange(rows * width).reshape(rows, width) % 97).to(
        torch.bfloat16
    ) / 128
    device = torch.empty_like(reference, device=f"npu:{device_id}")
    indices = torch.arange(rows, dtype=torch.int64, device=device.device)
    valid = torch.ones(rows, dtype=torch.bool, device=device.device)
    for cycle in range(2):
        region = None
        try:
            region = region_type.allocate(
                (rows, width), torch.bfloat16, device_id=device_id
            )
            for generation in range(3):
                expected = reference + generation
                region.tensor.copy_(expected)
                copy_rows_from_registered_host(region, device, indices, indices, valid)
                torch.npu.synchronize()
                assert torch.equal(device.cpu(), expected), "AIV Host read mismatch"
                device.add_(0.25)
                copy_rows_to_registered_host(device, region, indices, indices, valid)
                torch.npu.synchronize()
                assert torch.equal(region.tensor, expected + 0.25), (
                    "AIV Host write mismatch"
                )
                print("POOL_AIV_PASS", cycle, generation, flush=True)
        finally:
            torch.npu.synchronize()
            if region is not None:
                region.close(drained=True)
            if hasattr(region_type, "close_pool"):
                region_type.close_pool()


def test_unified_host_dma_and_aiv():
    import torch_npu
    from sdma_copy_lab import AclCopyBridge, BoundRowCopy
    from sglang.srt.hardware_backend.npu.fsa_native import (
        copy_rows_from_registered_host,
        copy_rows_to_registered_host,
    )

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch_npu.npu.set_device(device_id)
    UnifiedHostRegion = candidate_type()
    rows, width = 8192, 512
    region = UnifiedHostRegion.allocate(
        (rows, width), torch.bfloat16, device_id=device_id
    )
    bridge = AclCopyBridge(os.environ["SGLANG_SDMA_TEST_LIBRARY"])
    device = torch.empty_like(region.tensor, device=f"npu:{device_id}")
    ids = torch.arange(rows, dtype=torch.int64, device=device.device)
    valid = torch.ones(rows, dtype=torch.bool, device=device.device)
    plan = BoundRowCopy(
        list(range(rows)), list(range(rows)), [True] * rows, rows, rows, width * 2
    )
    stream = torch.npu.current_stream()
    print("ALLOCATED", flush=True)
    try:
        for generation in range(3):
            region.tensor.fill_(generation + 0.5)
            before = time.perf_counter()
            plan.submit(bridge, region.tensor, device, 1, stream, region)
            print("H2D_SUBMITTED_MS", (time.perf_counter() - before) * 1000, flush=True)
            stream.synchronize()
            assert bool((device == generation + 0.5).all().item()), (
                "DMA H2D data mismatch"
            )
            device.add_(0.25)
            plan.submit(bridge, device, region.tensor, 2, stream, region)
            stream.synchronize()
            assert bool((region.tensor == generation + 0.75).all().item()), (
                "DMA D2H data mismatch"
            )
            print("DMA_OK", generation, flush=True)
            device.zero_()
            copy_rows_from_registered_host(region, device, ids, ids, valid)
            stream.synchronize()
            assert bool((device == generation + 0.75).all().item()), (
                "AIV read after DMA write mismatch"
            )
            device.add_(0.25)
            copy_rows_to_registered_host(device, region, ids, ids, valid)
            stream.synchronize()
            assert bool((region.tensor == generation + 1).all().item()), (
                "AIV write mismatch"
            )
            print("AIV_OK", generation, flush=True)
            bridge.release_after_drain()
    finally:
        torch.npu.synchronize()
        bridge.release_after_drain()
        region.close(drained=True)
        if hasattr(UnifiedHostRegion, "close_pool"):
            UnifiedHostRegion.close_pool()


def test_unified_host_fused_attention_graph(monkeypatch):
    import torch_npu
    from test_fsa_attention_a5 import test_host_attention_cold_warm_reshuffle_and_graph
    from sglang.srt.hardware_backend.npu import fsa_native

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch_npu.npu.set_device(device_id)
    region_type = candidate_type()
    monkeypatch.setattr(fsa_native, "RegisteredHostRegion", region_type)
    try:
        test_host_attention_cold_warm_reshuffle_and_graph(
            torch.device(f"npu:{device_id}"),
            torch.bfloat16,
            1,
            16,
            3,
            137,
            2048,
            "fused",
        )
    finally:
        if hasattr(region_type, "close_pool"):
            region_type.close_pool()


def test_memfabric_binding_registration_guards():
    import torch_npu
    from memfabric_host_lab import MemFabricHostRegion
    from sglang.srt.hardware_backend.npu.fsa_native import native_ops

    if os.environ.get("SGLANG_MEMFABRIC_REGISTER_BINDING") != "1":
        pytest.skip("requires the candidate external-allocation binding")
    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch_npu.npu.set_device(device_id)
    region = MemFabricHostRegion.allocate(
        (256, 512), torch.bfloat16, device_id=device_id
    )
    operations = native_ops()
    try:
        with pytest.raises(RuntimeError, match="already registered"):
            operations.register_memfabric_region(
                region.tensor, region.device_ptr, device_id
            )
        with pytest.raises(RuntimeError, match="not mapped"):
            operations.register_memfabric_region(
                region.tensor, region.device_ptr + 32, device_id
            )
        with pytest.raises(RuntimeError, match="current NPU device"):
            operations.register_memfabric_region(
                region.tensor, region.device_ptr, device_id + 1
            )
        operations.unregister_memfabric_region(region.tensor, device_id)
        region.binding_registered = False
        with pytest.raises(RuntimeError, match="live allocation"):
            operations.unregister_memfabric_region(region.tensor, device_id)
        operations.register_memfabric_region(
            region.tensor, region.device_ptr, device_id
        )
        region.binding_registered = True
    finally:
        torch.npu.synchronize()
        region.close(drained=True)
        MemFabricHostRegion.close_pool()


def test_original_sysv_fsa_with_candidate_binding():
    import torch_npu
    from test_fsa_attention_a5 import test_host_attention_cold_warm_reshuffle_and_graph

    device_id = int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"])
    torch_npu.npu.set_device(device_id)
    test_host_attention_cold_warm_reshuffle_and_graph(
        torch.device(f"npu:{device_id}"),
        torch.bfloat16,
        1,
        16,
        3,
        137,
        2048,
        "fused",
    )
