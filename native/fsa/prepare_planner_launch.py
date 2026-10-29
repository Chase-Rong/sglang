"""Derive the configured launch from the SDK's exact generated argument ABI.

Adapted from native/lim_copy_sfa/prepare_configured_launch.py; independent
external_plan binary, entry, and registry. Fail closed on SDK stub changes.
"""

import re
import sys
from pathlib import Path


def prepare(build, destination):
    stub = (build / "auto_gen/fsa_planner_kernel/host_stub.cpp").read_text()
    name = "external_plan_prepare"
    anchor = f'extern "C" uint32_t aclrtlaunch_{name}('
    start = stub.index(anchor)
    body = stub.index("{", start)
    depth, end = 1, body + 1
    while depth:
        depth += (stub[end] == "{") - (stub[end] == "}")
        end += 1
    function = stub[start:end].replace(
        f"aclrtlaunch_{name}", "external_plan_configured", 1
    )
    pattern = rf"launch_and_profiling_{name}\((\d+), numBlocks, stream, \(void \*\*\)&__ascendc_args, sizeof\(__ascendc_args\)\)"
    function, count = re.subn(
        pattern,
        lambda m: f'ExternalPlanLaunch(numBlocks, stream, &__ascendc_args, sizeof(__ascendc_args), "{name}_{m[1]}")',
        function,
    )
    if count != 1:
        raise ValueError("SDK external planner launch ABI changed")
    function = function.replace("    KernelHandleGradUnregister::GetInstance();\n", "")
    binary = (build / "fsa_planner_kernel_merge_obj_dir/device.o").read_bytes()
    if not binary.startswith(b"\x7fELF"):
        raise ValueError("expected independently built planner ELF")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "configured_planner.h").write_text(
        "#pragma once\n#include <cstdint>\n"
        + function[: function.index("{")].strip()
        + ";\n"
    )
    payload = ",\n".join(
        ",".join(map(str, binary[i : i + 32])) for i in range(0, len(binary), 32)
    )
    (destination / "configured_planner.cpp").write_text(
        '#include "planner_configured_launch.h"\n'
        'extern "C" uint32_t AllocAscendMemDevice(void **, uint64_t);\n'
        'extern "C" uint32_t FreeAscendMemDevice(void *);\n'
        "static const unsigned char plannerBinary[] = {\n" + payload + "\n};\n"
        "static uint32_t ExternalPlanLaunch(uint32_t blocks, void *stream, const void *args, uint32_t size, const char *name) {\n"
        "return ExternalPlanHost::Launch(plannerBinary, sizeof(plannerBinary), blocks, stream, args, size, name);\n}\n"
        + function
        + "\n"
    )


if __name__ == "__main__":
    prepare(Path(sys.argv[1]), Path(sys.argv[2]))
