"""CPU tests of real Host constructors and KDA shapes, without NPU stream claims."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.mem_cache.device_pool_info import DevicePoolInfo, MambaStateBufferInfo
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = Path(__file__).parents[4] / "python/sglang/srt"


def source_definitions(monkeypatch, name, path, selected, **namespace):
    module = ModuleType(name)
    module.__dict__.update(namespace)
    monkeypatch.setitem(sys.modules, name, module)
    source = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node for node in ast.walk(source) if getattr(node, "name", None) in selected
    ]
    assert {node.name for node in nodes} == set(selected)
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


@pytest.fixture
def modules(monkeypatch):
    from sglang.srt.configs import mamba_utils
    from sglang.srt.hardware_backend.npu import hicache_mamba_pool
    from sglang.srt.mem_cache.pool_host import mamba

    monkeypatch.setattr(
        mamba, "host_memory_budget_bytes", lambda *args, **kwargs: 1 << 40
    )
    monkeypatch.setitem(
        mamba.ALLOC_MEMORY_FUNCS,
        "cpu",
        lambda dims, **kwargs: torch.empty(dims, dtype=kwargs["dtype"]),
    )
    return hicache_mamba_pool, mamba, mamba_utils


def make_pool(modules, *, strided=False):
    module, _, shapes = modules
    shape = shapes.KimiLinearStateShape.create(
        tp_world_size=8,
        num_heads=64,
        head_dim=128,
        conv_kernel_size=4,
    )
    assert shape.conv == [(3, 3072)]
    assert shape.temporal == (8, 128, 128)
    temporal = torch.arange(2 * 4 * 8 * 128 * 128, dtype=torch.float32)
    temporal = (temporal.remainder(997) / 17).reshape(2, 4, *shape.temporal)
    conv = torch.arange(2 * 4 * 3 * 3072).remainder(113).to(torch.bfloat16)
    conv = conv.reshape(2, 4, *shape.conv[0])
    if strided:
        temporal = temporal.transpose(-1, -2)
        conv = conv.transpose(-1, -2).contiguous().transpose(-1, -2)
    device = SimpleNamespace(
        device="cpu",
        size=3,
        host_capacity_tokens=None,
        num_mamba_layers=2,
        is_kda=True,
        enable_linear_replayssm=False,
        enable_linear_replayssm_spec=False,
        mamba_cache=SimpleNamespace(temporal=temporal, conv=[conv]),
    )
    device.get_device_pool_infos = lambda: (
        DevicePoolInfo(
            pool_name=PoolName.MAMBA,
            indices_from_pool=PoolName.MAMBA,
            layer_ids=(0, 1),
            buffer_info=MambaStateBufferInfo(
                temporal=device.mamba_cache.temporal,
                conv=tuple(device.mamba_cache.conv),
            ),
        ),
    )
    host = module.NpuMambaPoolHost(device, 2, 0, pin_memory=False)
    host.temporal_buffer.fill_(-7)
    for buffer in host.conv_buffer:
        buffer.fill_(-9)
    return device, host


@pytest.mark.parametrize("strided", [False, True])
def test_kda_checkpoint_roundtrip_all_components_and_layers(modules, strided):
    device, host = make_pool(modules, strided=strided)
    original_temporal = device.mamba_cache.temporal.clone()
    original_conv = device.mamba_cache.conv[0].clone()
    host.backup_from_device_all_layer(
        device, torch.tensor([0, 4]), torch.tensor([3, 1])
    )
    assert host.layer_num == 2 and host.page_size == 1
    assert host.layout == "page_first" and not host.can_use_write_back_jit
    assert host.temporal_buffer.dtype == torch.float32
    assert host.conv_buffer[0].dtype == torch.bfloat16
    assert torch.equal(host.temporal_buffer[0, :, 0], original_temporal[:, 3])
    assert torch.equal(host.conv_buffer[0][4, :, 0], original_conv[:, 1])
    assert torch.all(host.temporal_buffer[1] == -7)
    device.mamba_cache.temporal.fill_(-1)
    device.mamba_cache.conv[0].fill_(-2)
    for layer in range(2):
        host.load_to_device_per_layer(
            device, torch.tensor([4, 0]), torch.tensor([2, 3]), layer
        )
        expected = torch.full_like(original_temporal[layer], -1)
        expected[2] = original_temporal[layer, 1]
        expected[3] = original_temporal[layer, 3]
        assert torch.equal(device.mamba_cache.temporal[layer], expected)
        expected_conv = torch.full_like(original_conv[layer], -2)
        expected_conv[2] = original_conv[layer, 1]
        expected_conv[3] = original_conv[layer, 3]
        assert torch.equal(device.mamba_cache.conv[0][layer], expected_conv)
        if layer == 0:
            assert torch.all(device.mamba_cache.temporal[1] == -1)
    assert host.size_per_token * host.size == sum(
        buffer.nbytes for buffer in host.kv_buffer
    )


def test_mutated_index_tensors_are_read_again(modules):
    device, host = make_pool(modules)
    host_slots, device_slots = torch.tensor([0]), torch.tensor([1])
    host.backup_from_device_all_layer(device, host_slots, device_slots)
    host_slots.fill_(5)
    device_slots.fill_(3)
    host.backup_from_device_all_layer(device, host_slots, device_slots)
    assert torch.equal(host.temporal_buffer[5, :, 0], device.mamba_cache.temporal[:, 3])
    host_slots.fill_(0)
    device_slots.fill_(2)
    host.load_to_device_per_layer(device, host_slots, device_slots, 0)
    host_slots.fill_(5)
    device_slots.fill_(3)
    host.load_to_device_per_layer(device, host_slots, device_slots, 0)
    assert torch.equal(device.mamba_cache.temporal[0, 3], host.temporal_buffer[5, 0, 0])


def test_checkpoint_fanout_allows_duplicate_source_only(modules):
    device, host = make_pool(modules)
    host.backup_from_device_all_layer(
        device, torch.tensor([0, 1]), torch.tensor([3, 3])
    )
    assert torch.equal(host.temporal_buffer[0], host.temporal_buffer[1])
    device.mamba_cache.temporal.fill_(-1)
    host.load_to_device_per_layer(device, torch.tensor([0, 0]), torch.tensor([1, 3]), 0)
    assert torch.equal(
        device.mamba_cache.temporal[0, 1], device.mamba_cache.temporal[0, 3]
    )
    with pytest.raises(ValueError, match="destination"):
        host.load_to_device_per_layer(
            device, torch.tensor([0, 1]), torch.tensor([3, 3]), 0
        )
    with pytest.raises(ValueError, match="destination"):
        host.backup_from_device_all_layer(
            device, torch.tensor([0, 0]), torch.tensor([1, 3])
        )


def test_prepared_mamba_load_snapshots_mutable_cpu_indices(modules, monkeypatch):
    device, host = make_pool(modules)
    host.backup_from_device_all_layer(
        device, torch.tensor([0, 1]), torch.tensor([1, 3])
    )
    original = device.mamba_cache.temporal[:, 1].clone()
    host_indices, device_indices = torch.tensor([0]), torch.tensor([2])
    load_layer = host.prepare_load_to_device(
        device, host_indices, device_indices, "kernel_ascend"
    )
    host_indices.fill_(1)
    device_indices.fill_(3)
    device.mamba_cache.temporal.fill_(-1)
    monkeypatch.setattr(
        host, "_checkpoint_pairs", Mock(side_effect=AssertionError("metadata repeated"))
    )
    for layer in range(2):
        load_layer(layer)
    assert torch.equal(device.mamba_cache.temporal[:, 2], original)
    assert torch.all(device.mamba_cache.temporal[:, 3] == -1)


@pytest.mark.parametrize(
    "host_slots,device_slots",
    [
        ([0], [0]),
        ([0], [4]),
        ([-1], [1]),
        ([7], [1]),
        ([0, 1], [1]),
    ],
)
def test_bad_indices_fail_without_partial_writes(modules, host_slots, device_slots):
    device, host = make_pool(modules)
    with pytest.raises(ValueError):
        host.backup_from_device_all_layer(
            device, torch.tensor(host_slots), torch.tensor(device_slots)
        )
    assert torch.all(host.temporal_buffer == -7)
    assert torch.all(host.conv_buffer[0] == -9)


def test_bad_metadata_layout_and_lifetime_are_rejected(modules):
    device, host = make_pool(modules)
    for indices in (torch.tensor([1.5]), torch.tensor([[1]]), torch.tensor([True])):
        with pytest.raises(ValueError, match="integer vectors"):
            host.backup_from_device_all_layer(device, torch.tensor([0]), indices)
    with pytest.raises(ValueError, match="layer"):
        host.load_to_device_per_layer(device, torch.tensor([0]), torch.tensor([1]), 2)
    with pytest.raises(ValueError, match="main-model"):
        host.load_to_device_per_layer(
            device, torch.tensor([0]), torch.tensor([1]), 0, is_draft=True
        )
    with pytest.raises(ValueError, match="wrong device"):
        host.backup_from_device_all_layer(
            SimpleNamespace(), torch.tensor([0]), torch.tensor([1])
        )
    host.conv_buffer[0] = host.conv_buffer[0].float()
    with pytest.raises(ValueError, match="dtypes differ"):
        host.backup_from_device_all_layer(device, torch.tensor([0]), torch.tensor([1]))
    assert torch.all(host.temporal_buffer == -7)
    host.destroy()
    with pytest.raises(RuntimeError, match="destroyed"):
        host.backup_from_device_all_layer(device, torch.tensor([0]), torch.tensor([1]))


def test_storage_and_allocator_reuse_inherited_contract(modules):
    device, host = make_pool(modules)
    slots = host.alloc(2)
    assert slots.tolist() == [0, 1]
    host.backup_from_device_all_layer(device, slots[:1], torch.tensor([3]))
    data = host.get_data_page(0).clone()
    host.set_from_flat_data_page(1, data)
    assert torch.equal(host.get_data_page(1), data)
    assert data.dtype == torch.uint8
    assert data.numel() == host.size_per_token
    assert host.free(slots) == 2
    assert host.available_size() == host.size
    empty = torch.empty(0, dtype=torch.int64)
    host.backup_from_device_all_layer(device, empty, empty)
    host.load_to_device_per_layer(device, empty, empty, 0)


def test_budget_and_ring_state_are_checked_before_allocation(modules, monkeypatch):
    module, parent, _ = modules
    device, _ = make_pool(modules)
    monkeypatch.setattr(parent, "host_memory_budget_bytes", lambda *args: 0)
    with pytest.raises(ValueError, match="Not enough host memory"):
        module.NpuMambaPoolHost(device, 2, 0, pin_memory=False)
    device.enable_linear_replayssm = True
    with pytest.raises(NotImplementedError, match="ReplaySSM"):
        module.NpuMambaPoolHost(device, 2, 0, pin_memory=False)
    with pytest.raises(ValueError, match="page_first"):
        module.NpuMambaPoolHost(
            device, 2, 0, pin_memory=False, layout="page_first_kv_split"
        )


@pytest.mark.parametrize(
    "encoded_depth,expected_intermediate,expected_final",
    [
        (128, [], [1]),
        (129, [2], []),
        (150, [2], []),
        (192, [], [1]),
    ],
)
def test_kda_64_token_state_chunks_under_128_token_checkpoint_grid(
    monkeypatch, encoded_depth, expected_intermediate, expected_final
):
    metadata = source_definitions(
        monkeypatch,
        "hicache_mamba_checkpoint_metadata",
        ROOT / "layers/attention/hybrid_linear_attn_backend.py",
        {"_init_track_ssm_indices", "_init_track_conv_indices"},
        torch=torch,
        KDA_CHUNK_SIZE=64,
        Mamba2AttnBackend=type("UnusedMamba2", (), {}),
        mamba_cache_chunk_size=lambda: 128,
    )
    backend = SimpleNamespace(
        device="cpu",
        _has_cpu_prefill_track_metadata=lambda batch: False,
        req_to_token_pool=SimpleNamespace(mamba_pool=SimpleNamespace(is_kda=True)),
        conv_states_shape=(2, 4, 3072, 3),
    )
    batch = SimpleNamespace(
        mamba_track_mask=torch.tensor([True]),
        extend_seq_lens=torch.tensor([192]),
        mamba_track_indices=torch.tensor([3]),
        mamba_track_seqlens=torch.tensor([encoded_depth]),
        extend_prefix_lens=torch.tensor([0]),
    )
    tracking = source_definitions(
        monkeypatch,
        "hicache_track_alignment",
        ROOT / "model_executor/forward_batch_info.py",
        {"mamba_track_aligned_lens"},
        mamba_cache_chunk_size=lambda: 128,
    )
    batch.mamba_track_aligned_lens = lambda: tracking.mamba_track_aligned_lens(batch)
    result = metadata._init_track_ssm_indices(backend, torch.tensor([1]), batch)
    source, destination, final_source, final_destination = (
        result[1],
        result[2],
        result[4],
        result[5],
    )
    assert source.tolist() == expected_intermediate
    assert final_source.tolist() == expected_final
    assert destination.tolist() == ([3] if expected_intermediate else [])
    assert final_destination.tolist() == ([3] if expected_final else [])
    conv_indices = metadata._init_track_conv_indices(
        backend, torch.tensor([0, 192]), batch
    )
    assert conv_indices.tolist() == [[125, 126, 127]]


@pytest.mark.parametrize(
    "extend_length,donated,encoded",
    [(128, 128, 128), (150, 128, 129), (192, 128, 129), (256, 256, 256)],
)
def test_scheduler_encodes_kda_checkpoint_before_later_final_state(
    monkeypatch, extend_length, donated, encoded
):
    scheduler = source_definitions(
        monkeypatch,
        "hicache_mamba_checkpoint_scheduler",
        ROOT / "managers/schedule_batch.py",
        {
            "_mamba_radix_cache_v2_req_prepare_for_extend",
            "_MambaRadixCacheV2TrackEntry",
        },
        NamedTuple=NamedTuple,
        mamba_cache_chunk_size=lambda: 128,
        mamba_checkpoint_grid=lambda page_size: 128,
        mamba_extra_buffer_lazy_enabled=lambda: False,
        get_parallel=lambda: SimpleNamespace(dcp_enabled=False),
        get_exec=lambda: SimpleNamespace(
            mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=False)
        ),
    )
    batch = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(mamba_chunk_size=64)
        ),
        tree_cache=SimpleNamespace(page_size=128),
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(is_kda=True),
            get_mamba_ping_pong_other_idx=lambda index: 1 - index,
        ),
    )
    request = SimpleNamespace(
        prefix_len=0,
        extend_len=extend_length,
        extend_end=extend_length,
        kv=SimpleNamespace(
            mamba_next_track_idx=0, mamba_ping_pong_track_buffer=torch.tensor([2, 3])
        ),
        extend_range=SimpleNamespace(length=extend_length),
        prefix_indices=[],
        mamba_branching_seqlen=None,
    )
    result = scheduler._mamba_radix_cache_v2_req_prepare_for_extend(batch, request)
    assert request.kv.mamba_last_track_seqlen == donated
    assert result.track_seqlen == encoded
    assert result.track_index == 2 and result.track_mask


def test_extra_checkpoint_buffers_roundtrip_through_shared_mamba_format(modules):
    module, _, _ = modules
    device, _ = make_pool(modules)
    extra = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    info = MambaStateBufferInfo(
        temporal=device.mamba_cache.temporal,
        conv=tuple(device.mamba_cache.conv),
        extra_checkpoint_buffers=(extra,),
    )
    host = module.NpuMambaPoolHost(device, 2, 0, pin_memory=False, buffer_info=info)
    expected = extra[2].clone()
    host.backup_from_device_all_layer(device, torch.tensor([0]), torch.tensor([2]))
    extra.fill_(-1)
    host.load_to_device_per_layer(device, torch.tensor([0]), torch.tensor([3]), 0)
    torch.testing.assert_close(extra[3], expected)
    assert torch.all(extra[:3] == -1)
    assert host.slot_state_buffers[0].dtype == extra.dtype


if __name__ == "__main__":
    run_pytest(__file__)
