"""Decode backing geometry and write-through tests using CPU copy primitives."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

import test_prefill_offload_runtime as runtime_tests

NPU = runtime_tests.NPU
production_pool_type = runtime_tests.production_pool_type


@pytest.fixture
def modules(monkeypatch):
    return runtime_tests.modules.__wrapped__(monkeypatch)


@pytest.fixture
def backing_module(modules, monkeypatch):
    name = "sglang.srt.hardware_backend.npu.decode_host_kv"
    spec = importlib.util.spec_from_file_location(name, NPU / "decode_host_kv.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    class HostPool:
        def __init__(self, device_id):
            self.device_id = device_id
            self.regions = {}

        def allocate_component(self, name, shape, dtype, **kwargs):
            tensor = torch.full(shape, -99, dtype=dtype)
            region = SimpleNamespace(tensor=tensor, name=name, nbytes=tensor.nbytes,
                                     view=lambda shape: tensor.view(shape))
            self.regions[name] = region
            return SimpleNamespace(region=region)

        def copy_to_host(self, name, source, source_rows, destinations, valid):
            self.regions[name].tensor[destinations[valid]] = source[source_rows[valid]]

    monkeypatch.setattr(module, "NpuRegisteredHostKvPool", HostPool)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False)
    return module


def test_decode_pool_allocates_host_kv_but_retains_device_indexer(backing_module):
    pool = production_pool_type()(
        size=16, page_size=2, dtype=torch.float32, kv_lora_rank=4,
        qk_rope_head_dim=2, index_head_dim=3, layer_num=2,
        device="cpu", enable_memory_saver=False, decode_host_budget=10000,
    )
    assert isinstance(pool.k_buffer, list) and len(pool.k_buffer) == 2
    assert pool.index_k_buffer.shape == (2, 9, 2, 1, 3)
    assert pool.decode_host_backing.nbytes == 2 * 18 * 6 * 4
    assert pool.get_kv_size_bytes() == pool.index_k_buffer.nbytes
    key, rope = pool.get_kv_buffer(1)
    assert key.data_ptr() == pool.decode_host_backing.regions[1][0].tensor.data_ptr()
    assert rope.data_ptr() == pool.decode_host_backing.regions[1][1].tensor.data_ptr()


def test_current_token_write_uses_physical_rows_and_masks_padding(backing_module):
    backing = backing_module.DecodeHostKvBacking(
        layers=2, pages=3, page_size=2, latent_dim=4, rope_dim=2,
        dtype=torch.float32, device_id=0, host_budget=10000,
    )
    positions = torch.tensor([4, 1, 0, -1, 6], dtype=torch.int32)
    latent = torch.arange(20, dtype=torch.float32).view(5, 4)
    rope = torch.arange(10, dtype=torch.float32).view(5, 2)
    backing.write_current(1, positions, latent, rope)
    for region, values in zip(backing.regions[1], (latent, rope)):
        assert torch.equal(region.tensor[4], values[0])
        assert torch.equal(region.tensor[1], values[1])
        assert torch.all(region.tensor[[0, 2, 3, 5]] == -99)
    assert torch.all(backing.regions[0][0].tensor == -99)
    backing.write_current(1, positions, latent + 10, rope + 10)
    assert torch.equal(backing.regions[1][0].tensor[4], latent[0] + 10)


def test_host_budget_rejected_before_allocating_components(backing_module):
    with pytest.raises(ValueError, match="budget"):
        backing_module.DecodeHostKvBacking(
            layers=2, pages=3, page_size=2, latent_dim=4, rope_dim=2,
            dtype=torch.float32, device_id=0, host_budget=1,
        )
