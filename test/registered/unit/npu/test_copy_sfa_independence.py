"""CopySFA source ownership and device-capacity allocation contracts."""

from npu_test_paths import npu_module_name, npu_source_path
import importlib.util
import ast
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).parents[4]
NATIVE = ROOT / "native/lim_copy_sfa"


def test_copy_sfa_owns_its_kernel_dependencies():
    pending = [NATIVE / "op_kernel/lim_copy_sfa.cpp"]
    visited = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        assert path.is_relative_to(NATIVE.resolve())
        source = path.read_text(encoding="utf-8")
        assert "FusedSparseAttentionOverlap" not in source
        for include in re.findall(r'^#include\s+"([^"]+)"', source, re.M):
            target = (path.parent / include).resolve()
            if target.is_file():
                pending.append(target)
    assert len(visited) > 10
    assert "fsa_attention" not in (NATIVE / "CMakeLists.txt").read_text()


def test_copy_sfa_workspace_uses_device_core_capacity(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        npu_module_name("sglang.srt.hardware_backend.npu.hisparse.lim_copy_sfa_indexer"),
        SimpleNamespace(native_indexer_ops=lambda: object()),
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            get_device_properties=lambda device: SimpleNamespace(cube_core_num=32)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        torch, "empty", lambda size, **kwargs: calls.append((size, kwargs))
    )
    path = ROOT / "python/sglang/srt/hardware_backend/npu/hisparse/lim_copy_sfa_attention.py"
    spec = importlib.util.spec_from_file_location(npu_module_name("copy_sfa_allocation"), npu_source_path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    operator = module.NativeCopySparseAttention("npu:0", row_capacity=60)
    assert operator.core_count == 32
    assert calls == [
        (
            32 * 3 * 128 * 576 * 2 + 256 + 60 * 512 * 512 * 2,
            dict(dtype=torch.uint8, device="npu:0"),
        )
    ]


def test_copy_sfa_serving_adapter_does_not_import_fsa_implementation():
    directory = ROOT / "python/sglang/srt/hardware_backend/npu/hisparse"
    for filename in ("lim_copy_sfa_attention.py", "lim_copy_sfa_backend.py"):
        tree = ast.parse((directory / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").endswith("fused_sparse_attention")
            if isinstance(node, ast.Name):
                assert node.id not in (
                    "FusedSparseAttentionState",
                    "run_fused_sparse_attention",
                )


def test_copy_sfa_row_metadata_fits_aligned_id_and_plan_copies():
    source = (
        NATIVE / "op_kernel/copy_sfa/arch35/copy_sparse_attention_service_vector_mla.h"
    ).read_text(encoding="utf-8")
    count = int(re.search(r"ROW_METADATA_ELEMENTS = (\d+)", source).group(1))
    assert count >= (2051 + 15) // 16 * 16
    assert count * (4 + 2) % 32 == 0
    assert "InitBuffer(rowMetadataTBuf, ROW_METADATA_BYTES)" in source
