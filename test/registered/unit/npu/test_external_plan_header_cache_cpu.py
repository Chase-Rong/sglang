"""Compile the actual ABI3 parser and InitGlobalBuffer with CPU tensor/DMA stubs.

No torch, CANN, attention implementation, or NPU execution is involved.
"""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
HEADER = ROOT / (
    "native/fsa_attention/op_kernel/arch35/"
    "fused_sparse_attention_overlap_service_vector_mla.h"
)
HARNESS = ROOT / "native/fsa/tests/external_plan_header_cache_test.cpp"


def extract_method(source, name, return_type):
    pattern = (
        r"FusedSparseAttentionOverlapVectorService<TEMPLATE_ARGS>::"
        + re.escape(name)
        + r"\s*\([^;]*?\)\s*\{"
    )
    matches = list(re.finditer(pattern, source))
    assert len(matches) == 1, f"expected one actual definition of {name}"
    match = matches[0]
    start = source.index("::", match.start()) + 2
    depth = 1
    tokens = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]'
    for token in re.finditer(tokens, source[match.end() :]):
        if token[0] == "{":
            depth += 1
        elif token[0] == "}":
            depth -= 1
            if depth == 0:
                end = match.end() + token.end()
                line = source.count("\n", 0, start) + 1
                return (
                    f'#line {line} "{HEADER.relative_to(ROOT).as_posix()}"\n'
                    f"{return_type} {source[start:end]}\n"
                )
    raise AssertionError(f"unterminated actual definition of {name}")


def extracted_members(source):
    names = (
        "selectionKvBlockSize selectionMaxBlockNum selectionMembershipStride "
        "selectionDataRow selectionDirectRowStride selectionPlanOffset "
        "selectionPreparedRow selectionPreparedCount externalSelectionPlanActive "
        "SELECTION_EXTERNAL_PLAN_READY_MARKER SELECTION_MEMBERSHIP_CONTROL_OFFSET "
        "SELECTION_MEMBERSHIP_STORAGE_COUNT"
    ).split()
    declarations = []
    for name in names:
        matches = re.findall(
            rf"^\s*((?:static constexpr\s+)?(?:int64_t|int32_t|int16_t|bool)\s+"
            rf"{name}\s*=[^;]+;)",
            source,
            re.MULTILINE,
        )
        assert len(matches) == 1, f"expected one real member declaration for {name}"
        declarations.append(matches[0])
    methods = [
        extract_method(source, "GetSelectionRow", "int64_t"),
        extract_method(source, "PrepareExternalSelectionPlan", "bool"),
        extract_method(source, "InitGlobalBuffer", "void"),
    ]
    return "\n".join(declarations + methods)


def compiler_command():
    configured = os.environ.get("CXX", "").strip()
    if configured:
        if Path(configured).is_file():
            command = [configured]
        else:
            command = shlex.split(configured, posix=os.name != "nt")
        if command and shutil.which(command[0]):
            return command
    for name in ("c++", "g++", "clang++"):
        compiler = shutil.which(name)
        if compiler:
            return [compiler]
    if os.name == "nt":
        compiler = (
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "LLVM/bin/clang++.exe"
        )
        if compiler.is_file():
            return [str(compiler)]
    pytest.skip("CPU parser harness requires CXX, c++, g++, or clang++")


@pytest.fixture(scope="module")
def parser_binary(tmp_path_factory):
    compiler = compiler_command()
    build = tmp_path_factory.mktemp("external_plan_header_cache")
    extracted = build / "external_plan_header_cache_extracted.inc"
    extracted.write_text(
        extracted_members(HEADER.read_text(encoding="utf-8")), encoding="utf-8"
    )
    binary = build / (
        "header_cache_test.exe" if os.name == "nt" else "header_cache_test"
    )
    command = compiler + [
        "-std=c++17",
        "-O0",
        "-Wall",
        "-Wextra",
        "-I",
        str(build),
        str(HARNESS),
        "-o",
        str(binary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"CPU harness compilation failed: {command!r}\n{result.stdout}\n{result.stderr}"
    )
    return binary


CASES = [
    ("same-row-17-tiles",),
    ("fresh-signed-slices",),
    ("row-refresh-tnd",),
    ("row-refresh-bsnd",),
    ("head-row-refresh",),
    ("valid-boundaries",),
    ("negative-row",),
    ("short-membership",),
    ("zero-membership",),
    ("short-cached-plan",),
    ("new-launch",),
    ("graph-replay-reset",),
]

INVALID_HEADERS = [
    ("bad-marker", 0, 0),
    ("legacy-marker", 0, 0x5A44),
    ("not-ready", 1, 0),
    ("short-count", 2, 127),
    ("negative-count", 2, -1),
    ("count-over-control", 2, 16385),
    ("negative-offset", 3, -1),
    ("overrun-offset", 3, 16384 - 2176 + 1),
    ("not-end-aligned", 3, 16384 - 2176 - 1),
    ("negative-owner", 4, -1),
    ("bad-layout", 5, 0),
    ("legacy-layout", 5, 0x5A44),
    ("zero-direct-stride", 6, 0),
    ("negative-direct-stride", 6, -1),
    ("version-one", 7, 1),
    ("version-two", 7, 2),
    ("future-version", 7, 4),
]


@pytest.mark.parametrize(
    "arguments",
    [pytest.param(case, id=case[0]) for case in CASES]
    + [
        pytest.param(("invalid-header", str(field), str(value)), id=name)
        for name, field, value in INVALID_HEADERS
    ]
    + [
        pytest.param(
            ("actual-seq", str(subblock), str(tile)), id=f"actual-seq-{subblock}-{tile}"
        )
        for subblock in (0, 1)
        for tile in (0, 1)
    ],
)
def test_actual_external_plan_parser_cpu(parser_binary, arguments):
    result = subprocess.run(
        [str(parser_binary), *arguments], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, f"{arguments!r}\n{result.stdout}\n{result.stderr}"
    assert f"PASS {arguments[0]}" in result.stdout
