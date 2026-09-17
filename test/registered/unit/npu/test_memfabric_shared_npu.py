"""Qualify the branch's rank-zero allocation and documented shared-GVA access."""

import ctypes
import os

import pytest
import torch

from test_fsa_shared_tp_npu import tp as tp

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_UNIFIED_HOST_TEST") != "1"
    or os.environ.get("SGLANG_FSA_TEST_SHARED_TP") != "1",
    reason="requires explicitly selected idle TP devices",
)


def test_memfabric_one_host_pool_graph_and_collective(request):
    import memfabric_hybrid as fabric
    from memfabric_host_lab import MemFabricHostRegion
    from sdma_copy_lab import AclCopyBridge
    from sglang.srt.hardware_backend.npu.fsa_native import (
        RegisteredHostRegion,
        copy_rows_from_registered_host,
        copy_rows_to_registered_host,
    )

    parallel = request.getfixturevalue("tp")
    devices = os.environ["SGLANG_FSA_TEST_DEVICES"].split(",")
    world_size = len(devices)
    MemFabricHostRegion.initialize_pool(
        parallel.device_id, rank=parallel.rank, world_size=world_size
    )
    rows, width = 256, 512
    size = rows * width * 2
    pointer = fabric.offload.malloc(size) if parallel.rank == 0 else 0
    pointers = parallel.group.all_gather_object(pointer)
    assert pointers[0] and all(value == 0 for value in pointers[1:])
    pointer = pointers[0]
    raw = (ctypes.c_ubyte * size).from_address(pointer)
    host = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(rows, width)
    region = RegisteredHostRegion(
        host, pointer, pointer, parallel.device_id, "memfabric-shared-gva", False
    )
    output = torch.empty_like(host, device=parallel.device)
    indices = torch.arange(rows, dtype=torch.int64, device=parallel.device)
    valid = torch.ones(rows, dtype=torch.bool, device=parallel.device)
    pattern = (torch.arange(rows * width).reshape(rows, width) % 97).to(
        torch.bfloat16
    ) / 128
    graph = None
    bridge = (
        AclCopyBridge(os.environ["SGLANG_SDMA_TEST_LIBRARY"])
        if os.environ.get("MEMFABRIC_SHARED_DMA") == "1"
        else None
    )

    def read():
        if bridge is None:
            copy_rows_from_registered_host(region, output, indices, indices, valid)
        else:
            bridge.submit(
                pointer,
                output.data_ptr(),
                ((0, 0, size),),
                3,
                torch.npu.current_stream(),
                (region, output),
            )

    def agree(local_ok, stage):
        results = parallel.group.all_gather_object(bool(local_ok))
        assert all(results), f"{stage} failed on ranks: {results}"

    try:
        if parallel.rank == 0:
            host.copy_(pattern)
        parallel.group.all_gather_object("writer ready")
        agree(
            torch.equal(host, pattern) if parallel.rank == 0 else True,
            "owner CPU view",
        )
        read()
        torch.npu.synchronize()
        agree(torch.equal(output.cpu(), pattern), "shared AIV read")
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            read()
        for generation in range(3):
            expected = pattern + generation
            if parallel.rank == 0:
                host.copy_(expected)
            parallel.group.all_gather_object("generation ready")
            graph.replay()
            torch.npu.synchronize()
            agree(torch.equal(output.cpu(), expected), "shared Graph replay")
            result = output.float()
            torch.distributed.all_reduce(result, group=parallel.group.device_group)
            torch.npu.synchronize()
            agree(
                torch.equal(result.cpu(), expected.float() * world_size),
                "TP collective",
            )
            for writer in range(world_size):
                if parallel.rank == writer:
                    output.fill_(writer + 4)
                    row = torch.tensor(
                        [writer], dtype=torch.int64, device=parallel.device
                    )
                    copy_rows_to_registered_host(output, region, row, row, valid[:1])
                    torch.npu.synchronize()
                parallel.group.all_gather_object("peer write ready")
                agree(
                    bool((host[writer] == writer + 4).all())
                    if parallel.rank == 0
                    else True,
                    "peer write visibility on owner",
                )
            print("MEMFABRIC_SHARED_PASS", parallel.rank, generation, flush=True)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        if bridge is not None:
            bridge.release_after_drain()
        parallel.group.all_gather_object("all readers drained")
        if parallel.rank == 0:
            fabric.offload.free(pointer)
        parallel.group.all_gather_object("owner allocation freed")
        MemFabricHostRegion.close_pool()
