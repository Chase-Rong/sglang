"""Explicit device and dependency gates for bounded offload kernel checks."""

import os
from pathlib import Path

import pytest
import torch


def offload_test_device(*, extensions=(), memfabric=False):
    if os.environ.get("SGLANG_NPU_OFFLOAD_TEST") != "1":
        pytest.skip("set SGLANG_NPU_OFFLOAD_TEST=1 to opt into device checks")
    device_id = os.environ.get("SGLANG_FSA_TEST_DEVICE_ID")
    if device_id is None:
        pytest.skip("select one idle device with SGLANG_FSA_TEST_DEVICE_ID")
    for extension in extensions:
        variable = {
            "fsa": "SGLANG_NPU_FSA_EXTENSION",
            "lim": "SGLANG_LIM_NATIVE_EXTENSION",
        }[extension]
        library = os.environ.get(variable)
        if not library:
            pytest.skip(f"requires {variable}")
        if not Path(library).is_file():
            pytest.fail(f"{variable} does not name a library file: {library}")
    pytest.importorskip("torch_npu")
    pytest.importorskip("triton")
    if memfabric:
        pytest.importorskip("memfabric_hybrid")
    if not torch.npu.is_available():
        pytest.skip("NPU runtime is unavailable")
    device = torch.device("npu", int(device_id))
    torch.npu.set_device(device)
    total = torch.npu.get_device_properties(device).total_memory
    torch.npu.set_per_process_memory_fraction(min(1.0, (512 << 20) / total), device)
    return device
