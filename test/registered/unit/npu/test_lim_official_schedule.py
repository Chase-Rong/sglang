"""Host-only gates for the official-PR schedule adapted to pooled BF16 KV."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "official_lim_reference", ROOT / "test/manual/npu/lim_copy_sfa_reference.py"
)
REFERENCE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REFERENCE
SPEC.loader.exec_module(REFERENCE)


@pytest.mark.parametrize("routes", [1, 4, 6])
@pytest.mark.parametrize("source_capacity", [16384, 1048576])
def test_protected_scan_preserves_every_query_and_remapping(routes, source_capacity):
    capacity = 4096 if routes == 1 else 12416
    state = REFERENCE.ReferenceResident(capacity, source_capacity, "protected_hash")
    table = torch.arange(source_capacity // 128)
    lengths = [source_capacity - routes + route + 1 for route in range(routes)]
    source_base = source_capacity - 16384
    for step in range(9):
        ids = torch.stack(
            [
                (torch.arange(2051) + source_base + step * 1200 + route * 32)
                % (source_capacity - 32)
                for route in range(routes)
            ]
        )
        if step % 2:
            ids = ids.flip(1)
        if step == 4:
            table = table.flip(0)
        plan = state.plan(ids, table, lengths, step // 6 + 1)
        valid = plan.logical_ids >= 0
        assert torch.equal(state.sources[plan.slots[valid]], plan.logical_ids[valid])
        assert len(state.sources[state.sources >= 0].unique()) == int(
            (state.sources >= 0).sum()
        )
        assert len(plan.copy_slots.unique()) == len(plan.copy_slots)
        if step == 4:
            assert plan.misses[valid].all()
        state.commit(plan)
        repeated = state.plan(ids.flip(1), table, lengths, step // 6 + 1)
        assert not repeated.misses[repeated.stable].any()
        state.commit(repeated)


def test_native_schedule_boundaries(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("A host C++ compiler is required")
    source = tmp_path / "schedule.cpp"
    source.write_text(
        '#include "lim_schedule.h"\n'
        "#include <cassert>\n"
        "int main() {\n"
        "using namespace SglLimSchedule;\n"
        "assert(!NeedsUnionCopy(0) && !NeedsUnionCopy(512) && NeedsUnionCopy(513));\n"
        "assert(HashVictimScanSeed(1048576) != HashVictimScanSeed(1048577));\n"
        "for (unsigned batch=0; batch<64; ++batch) {\n"
        "bool seen[2051] = {};\n"
        "for(unsigned index=0; index<2051; ++index) {\n"
        "auto mapped=GatherIndex(batch,index); assert(mapped<2051 && !seen[mapped]);\n"
        "seen[mapped]=true; if(index>=2048) assert(mapped==index);\n"
        "if(index<2048) assert(mapped%128==index%128); }\n"
        "assert(GatherIndex(batch,(3-batch%3)*512)==0); }\n"
        "}\n",
        encoding="utf-8",
    )
    executable = tmp_path / "schedule"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "native/lim_copy_sfa/op_kernel"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True)


def test_configured_launcher_preserves_generated_abi(tmp_path):
    module_spec = importlib.util.spec_from_file_location(
        "configured_lim_launch",
        ROOT / "native/lim_copy_sfa/prepare_configured_launch.py",
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    stub_directory = tmp_path / "auto_gen/sgl_lim_indexer_kernel"
    stub_directory.mkdir(parents=True)
    stub = (
        'extern "C" uint32_t aclrtlaunch_sgl_lim_indexer_manage(uint32_t numBlocks, '
        "void *stream, void *query) {\n"
        "    struct { void *query; } __ascendc_args;\n"
        "    __ascendc_args.query = query;\n"
        "    uint32_t __ascendc_ret = launch_and_profiling_sgl_lim_indexer_manage("
        "2, numBlocks, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));\n"
        "    KernelHandleGradUnregister::GetInstance();\n"
        "    return __ascendc_ret;\n}\n"
    )
    (stub_directory / "host_stub.cpp").write_text(stub, encoding="utf-8")
    binary_directory = tmp_path / "sgl_lim_indexer_kernel_merge_obj_dir"
    binary_directory.mkdir()
    binary = binary_directory / "device.o"
    binary.write_bytes(b"\x7fELF\x00\x01")
    destination = tmp_path / "configured"
    module.prepare_launch(tmp_path, destination)
    generated = (destination / "configured_indexer.cpp").read_text()
    assert "struct { void *query; } __ascendc_args;" in generated
    assert "__ascendc_args.query = query;" in generated
    assert "SglLimLaunchConfigured(numBlocks, stream, &__ascendc_args" in generated
    assert "launch_and_profiling" not in generated
    assert "KernelHandleGradUnregister" not in generated
    assert "127,69,76,70,0,1" in generated
    binary.write_bytes(b"not an ELF")
    with pytest.raises(ValueError, match="mixed-core ELF"):
        module.prepare_launch(tmp_path, destination)
    binary.write_bytes(b"\x7fELF")
    (stub_directory / "host_stub.cpp").write_text(
        stub.replace("2, numBlocks", "3, numBlocks"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ABI changed"):
        module.prepare_launch(tmp_path, destination)
