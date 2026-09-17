"""Build and qualify an isolated row-copy extension without replacing the wheel."""

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile

import paramiko

try:
    from ssh_auth import connect_host
except ImportError:  # script directory not on sys.path
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from ssh_auth import connect_host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--container", default="jzx_sglang_5.3_flash")
    parser.add_argument("--root", required=True)
    parser.add_argument("--action", choices=("inspect", "build", "test"), required=True)
    args = parser.parse_args()
    if not args.root.startswith("/home/j00628475/") or ".." in args.root.split("/"):
        raise ValueError("Use the assigned work directory")
    repo = Path(__file__).resolve().parents[2]
    baseline = "/home/j00628475/fsa_glm53_pd_20260915"
    client = connect_host(args.host)

    def run(command):
        print("COMMAND", shlex.join(command), flush=True)
        _, output, _ = client.exec_command(shlex.join(command))
        output.channel.set_combine_stderr(True)
        lines = []
        for line in output:
            print(line, end="", flush=True)
            lines.append(line)
        code = output.channel.recv_exit_status()
        result = "".join(lines)
        if code:
            raise RuntimeError(f"Remote exit {code}")
        return result

    def container(command):
        return run(["docker", "exec", args.container, *command])

    try:
        if args.action == "inspect":
            container(["bash", "-c", "\n".join([
                "find " + shlex.quote(baseline) + " -maxdepth 3 -type d",
                "grep -nE 'operator\\[\\]|SetGlobalBuffer' /usr/local/Ascend/cann-9.1.0/aarch64-linux/asc/impl/basic_api/kernel_tensor_impl.h | tail -20",
            ])])
            return
        if args.action == "build":
            target = args.root + "/kernel_source"
            with tempfile.TemporaryDirectory() as temporary:
                archive = Path(temporary) / "kernel_source.tar"
                subprocess.run(["git", "archive", "--format=tar", "--output", str(archive),
                                "539e7c938546bce796d90abefdf4aea3080d57f8", "csrc"],
                               cwd=repo.parent / "sgl-kernel-npu", check=True)
                with client.open_sftp() as transfer:
                    transfer.put(str(archive), args.root + "/kernel_source.tar")
            container(["mkdir", target])
            container(["tar", "-xf", args.root + "/kernel_source.tar", "-C", target])
            paths = ["csrc/sparsity_driven_kv_offload/unidex_copy/op_host/unidex_copy.cpp",
                     "csrc/sparsity_driven_kv_offload/unidex_copy/op_kernel/unidex_copy_kernel.cpp"]
            hashes = {}
            with client.open_sftp() as transfer:
                for path in paths:
                    local = repo.parent / "sgl-kernel-npu" / path
                    transfer.put(str(local), target + "/" + path)
                    hashes[path] = hashlib.sha256(local.read_bytes()).hexdigest()
            command = "\n".join([
                "set -eo pipefail",
                "source /usr/local/Ascend/cann-9.1.0/set_env.sh",
                shlex.join(["cmake", "-S", args.root + "/source/native/fsa", "-B", args.root + "/native-build",
                            "-DFSA_SOURCE_ROOT=" + target,
                            "-DSOC_VERSION=Ascend950PR_9599",
                            "-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0"]),
                shlex.join(["cmake", "--build", args.root + "/native-build", "-j", "8"]),
                shlex.join(["sha256sum", args.root + "/native-build/libsgl_fsa_extension.so"]),
            ])
            result = container(["bash", "-c", command])
            destination = repo / "results/joint_budget" / args.host / Path(args.root).name
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "native_build.txt").write_text(result)
            (destination / "native_source_sha256.json").write_text(json.dumps(hashes, indent=2))
        else:
            report = run(["npu-smi", "info"])
            if "No running processes found in NPU 0" not in report:
                raise RuntimeError("NPU0 is not confirmed idle")
            container(["mkdir", "-p", args.root + "/tmp"])
            command = "\n".join([
                "set -eo pipefail",
                "source /usr/local/Ascend/cann-9.1.0/set_env.sh",
                shlex.join(["env", "PYTHONPATH=" + args.root + "/source/python",
                            "SGLANG_NPU_FSA_EXTENSION=" + args.root + "/native-build/libsgl_fsa_extension.so",
                            "SGLANG_FSA_TEST_DEVICE_ID=0", "TMPDIR=" + args.root + "/tmp",
                            "python3", "-m", "pytest", "--override-ini", "addopts=", "-q",
                            args.root + "/source/test/registered/unit/npu/test_fsa_native_helpers_npu.py"]),
            ])
            result = container(["bash", "-c", command])
            destination = repo / "results/joint_budget" / args.host / Path(args.root).name
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "native_ut.txt").write_text(result)
    finally:
        client.close()


if __name__ == "__main__":
    main()
