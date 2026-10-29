"""Package the sole external-plan SFA implementation (native ABI 3).

The operator keeps its established name and exposes an ABI3 contract marker.
The materialized op_host/op_kernel tree is the complete implementation; no source
specialization or legacy Attention implementation is generated here.
"""
import argparse
import hashlib
import json
import pathlib
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann", type=pathlib.Path, required=True)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    a = parser.parse_args()
    template = a.cann / "tools/op_project_templates/ascendc/customize"
    if a.output.exists():
        raise FileExistsError(a.output)
    required = a.source / "op_kernel/fused_sparse_attention_overlap.cpp"
    if not required.is_file():
        raise FileNotFoundError(required)
    shutil.copytree(template, a.output)
    for folder in ("op_host", "op_kernel", "abi"):
        shutil.copytree(a.source / folder, a.output / folder, dirs_exist_ok=True)
    f = a.output / "cmake/intf.cmake"
    s = f.read_text().replace("-std=c++11", "-std=c++17")
    s += "\ntarget_include_directories(intf_pub INTERFACE ${ASCEND_CANN_PACKAGE_PATH}/include/op_common ${CMAKE_SOURCE_DIR}/op_host/include)\n"
    f.write_text(s)
    f = a.output / "cmake/func.cmake"
    s = (
        f.read_text()
        .replace("-std=c++11", "-std=c++17")
        .replace(
            "-I ${ASCEND_CANN_PACKAGE_PATH}/include -I",
            "-I ${ASCEND_CANN_PACKAGE_PATH}/include -I ${ASCEND_CANN_PACKAGE_PATH}/include/op_common -I ${CMAKE_SOURCE_DIR}/op_host/include -I",
        )
    )
    f.write_text(s)
    manifest = {
        str(f.relative_to(a.output)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(a.output.rglob("*"))
        if f.is_file()
    }
    (a.output / "generated-source-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(a.output)


if __name__ == "__main__":
    main()
