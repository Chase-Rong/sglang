"""Prepare an isolated A5 operator project using the installed CANN templates."""

import argparse
import pathlib
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    source = pathlib.Path(__file__).resolve().parents[2] / "native/fsa_attention"
    template = args.cann / "tools/op_project_templates/ascendc/customize"
    if args.output.exists():
        raise FileExistsError(f"Use a new build directory: {args.output}")
    shutil.copytree(template, args.output)
    for name in ("op_host", "op_kernel"):
        shutil.copytree(source / name, args.output / name, dirs_exist_ok=True)
    shutil.copyfile(template / "op_host/CMakeLists.txt", args.output / "op_host/CMakeLists.txt")
    kernel_cmake = """add_ops_compile_options(FusedSparseAttentionOverlap OPTIONS
    --cce-auto-sync=off -Wno-deprecated-declarations
    -mllvm -cce-vf-remove-membar=false -mllvm -cce-aicore-hoist-movemask=false)
add_kernels_compile()
"""
    (args.output / "op_kernel/CMakeLists.txt").write_text(kernel_cmake)
    interface = args.output / "cmake/intf.cmake"
    text = interface.read_text().replace("-std=c++11", "-std=c++17")
    text += """
target_include_directories(intf_pub INTERFACE
    ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
    ${CMAKE_SOURCE_DIR}/op_host/include)
"""
    interface.write_text(text)
    functions = args.output / "cmake/func.cmake"
    text = functions.read_text().replace("-std=c++11", "-std=c++17")
    text = text.replace(
        "-I ${ASCEND_CANN_PACKAGE_PATH}/include -I",
        "-I ${ASCEND_CANN_PACKAGE_PATH}/include "
        "-I ${ASCEND_CANN_PACKAGE_PATH}/include/op_common "
        "-I ${CMAKE_SOURCE_DIR}/op_host/include -I",
    )
    functions.write_text(text)
    print(args.output)


if __name__ == "__main__":
    main()
