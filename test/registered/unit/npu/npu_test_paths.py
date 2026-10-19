"""Resolve mixed shared-NPU and HiSparse module fixtures after consolidation."""

from pathlib import Path

NPU = Path(__file__).parents[4] / "python/sglang/srt/hardware_backend/npu"
MODULES = frozenset(path.stem for path in (NPU / "hisparse").glob("*.py"))
PREFIX = "sglang.srt.hardware_backend.npu."


def npu_module_name(name):
    if name.startswith(PREFIX) and name[len(PREFIX) :] in MODULES:
        return PREFIX + "hisparse." + name[len(PREFIX) :]
    return name


def npu_source_path(path):
    path = Path(path)
    if path.parent == NPU and path.stem in MODULES:
        return NPU / "hisparse" / path.name
    return path
