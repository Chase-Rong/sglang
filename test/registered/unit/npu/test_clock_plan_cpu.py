"""Exercise the production portable planner without Torch/NPU dependencies."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_clock_plan_cpu_contract(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    root = Path(__file__).resolve().parents[4]
    output = tmp_path / "clock_plan_test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(root / "native/fsa/include"),
            str(Path(__file__).with_name("clock_plan_test.cpp")),
            "-o",
            str(output),
        ],
        check=True,
    )
    result = subprocess.run(
        [str(output)], check=True, capture_output=True, text=True, timeout=90
    )
    assert "CLOCK_CONTRACT_OK" in result.stdout
