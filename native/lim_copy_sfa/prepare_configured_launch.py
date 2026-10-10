"""Embed the SDK-generated mixed kernel and preserve its launch argument ABI."""

import re
import sys
from pathlib import Path


def prepare_launch(build_root, destination):
    stub = (build_root / "auto_gen/sgl_lim_indexer_kernel/host_stub.cpp").read_text()
    anchor = 'extern "C" uint32_t aclrtlaunch_sgl_lim_indexer_manage('
    start = stub.index(anchor)
    body_start = stub.index("{", start)
    depth = 1
    end = body_start + 1
    while depth:
        depth += (stub[end] == "{") - (stub[end] == "}")
        end += 1
    function = stub[start:end].replace(
        "aclrtlaunch_sgl_lim_indexer_manage", "sgl_lim_configured_indexer_manage", 1
    )
    function, replacements = re.subn(
        r"launch_and_profiling_sgl_lim_indexer_manage\(2, numBlocks, stream, "
        r"\(void \*\*\)&__ascendc_args, sizeof\(__ascendc_args\)\)",
        "SglLimLaunchConfigured(numBlocks, stream, &__ascendc_args, sizeof(__ascendc_args))",
        function,
    )
    if replacements != 1:
        raise ValueError("SDK fused LIM launch ABI changed")
    function = function.replace("    KernelHandleGradUnregister::GetInstance();\n", "")
    binary = (build_root / "sgl_lim_indexer_kernel_merge_obj_dir/device.o").read_bytes()
    if not binary.startswith(b"\x7fELF"):
        raise ValueError("Expected the SDK mixed-core ELF")
    declaration = function[: function.index("{")].strip() + ";\n"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "configured_indexer.h").write_text(
        "#pragma once\n#include <cstdint>\n" + declaration, encoding="utf-8"
    )
    payload = ",\n".join(
        ",".join(str(value) for value in binary[offset : offset + 32])
        for offset in range(0, len(binary), 32)
    )
    (destination / "configured_indexer.cpp").write_text(
        '#include "configured_launch.h"\n'
        'extern "C" uint32_t AllocAscendMemDevice(void **, uint64_t);\n'
        'extern "C" uint32_t FreeAscendMemDevice(void *);\n'
        "static const unsigned char sglLimMixedBinary[] = {\n" + payload + "\n};\n"
        "static uint32_t SglLimLaunchConfigured(uint32_t blocks, void *stream, "
        "const void *arguments, uint32_t size) {\n"
        "return SglLimHost::LaunchConfiguredIndexer(sglLimMixedBinary, "
        "sizeof(sglLimMixedBinary), blocks, stream, arguments, size);\n}\n"
        + function
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    prepare_launch(Path(sys.argv[1]), Path(sys.argv[2]))
