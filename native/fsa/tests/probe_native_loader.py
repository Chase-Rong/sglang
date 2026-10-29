"""Exercise the production standalone loader against an actual Linux PTA DSO.

Run in a fresh process after building native/fsa. This probe performs no Host
allocation or device-kernel invocation. The extension loader contract is ABI1;
normal Attention and the optional fill helper retain separate contracts.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch
import torch_npu  # Register PTA before loading its dependent native extension.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    extension = args.extension.resolve(strict=True)
    os.environ["SGLANG_NPU_FSA_EXTENSION"] = str(extension)
    module_path = source / "python/sglang/srt/hardware_backend/npu/fsa_native.py"
    spec = importlib.util.spec_from_file_location("fsa_native_loader_probe", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.load_native_extension()
    ops = module.native_ops()
    assert ops.abi_version() == 1
    schema = ops.npu_fused_sparse_attention_overlap.default._schema
    assert len(schema.arguments) == 30 and schema.arguments[-1].default_value == 3
    assert not hasattr(ops, "npu_fused_sparse_attention_shared_v3")
    assert module.load_native_extension()  # Same normalized path is cached.
    with tempfile.TemporaryDirectory(prefix="loader-probe-", dir=extension.parent) as temp:
        other = Path(temp) / extension.name
        shutil.copyfile(extension, other)
        os.environ["SGLANG_NPU_FSA_EXTENSION"] = str(other)
        try:
            module.load_native_extension()
        except RuntimeError as error:
            assert "cannot change after loading" in str(error), str(error)
        else:
            raise AssertionError("loaded standalone extension path changed")
        finally:
            os.environ["SGLANG_NPU_FSA_EXTENSION"] = str(extension)
    print(json.dumps({
        "status": "PASS", "source_root": str(source),
        "loader_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "extension": str(extension),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "standalone_abi": 1, "normal_abi": 3, "schema_arguments": 30,
        "cached_load": True, "changed_path_rejected": True,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "scope": "actual production loader and Linux PTA DSO; no Host allocation or device kernels",
    }))


if __name__ == "__main__":
    main()
