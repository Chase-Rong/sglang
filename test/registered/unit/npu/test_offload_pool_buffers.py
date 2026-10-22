import ast
import contextlib
import sys
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

SRT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def class_node(path, name):
    return next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def compile_nodes(nodes, namespace):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, "production_pool", "exec"), namespace)


class PoolName(Enum):
    KV = "kv"
    MAMBA = "mamba"


class PoolBase:
    def __init__(self, **values):
        self.__dict__.update(values)
        self.start_layer = values["start_layer"] or 0
        self.store_dtype = self.dtype
        self.layer_transfer_counter = None
        self.memory_saver_adapter = SimpleNamespace(
            region=lambda _: contextlib.nullcontext()
        )

    def _finalize_allocation_log(self, size):
        self.mem_usage = self.get_kv_size_bytes()


class HostBacking:
    def __init__(self, **values):
        self.values = values
        self.writes = []
        self.components = [
            tuple(
                torch.zeros(
                    values["pages"],
                    values["page_size"],
                    1,
                    values[name],
                    dtype=values["dtype"],
                )
                for name in ("latent_dim", "rope_dim")
            )
            for _ in range(values["layers"])
        ]

    def layer_views(self, layer):
        return self.components[layer]

    def write_current(self, layer, locations, latent, rope):
        self.writes.append((layer, locations.clone(), latent.clone(), rope.clone()))


@pytest.fixture
def pool_types(monkeypatch):
    from test_offload_config import MODULE as offload_config

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.hardware_backend.npu.hisparse.offload_config",
        offload_config,
    )
    info_path = SRT / "mem_cache/device_pool_info.py"
    info_namespace = dict(
        torch=torch, msgspec=msgspec, Callable=Callable, PoolName=PoolName
    )
    compile_nodes(
        [
            class_node(info_path, name)
            for name in (
                "MambaStateBufferInfo",
                "MHABufferInfo",
                "MLASplitBuffers",
                "MLABufferInfo",
                "DevicePoolInfo",
            )
        ],
        info_namespace,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.device_pool_info",
        SimpleNamespace(**info_namespace),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.hicache_storage",
        SimpleNamespace(PoolName=PoolName),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.hardware_backend.npu.hisparse.decode_host_kv",
        SimpleNamespace(DecodeHostKvBacking=HostBacking),
    )
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_device=lambda: 0), raising=False
    )
    memory_pool = SRT / "mem_cache/memory_pool.py"
    mla = class_node(memory_pool, "MLATokenToKVPool")
    mla.bases = [ast.Name(id="PoolBase", ctx=ast.Load())]
    mla.body = [
        node
        for node in mla.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_device_pool_infos"
    ]
    namespace = dict(
        PoolBase=PoolBase,
        torch=torch,
        get_bool_env_var=lambda _: False,
        envs=SimpleNamespace(
            SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=SimpleNamespace(get=lambda: False)
        ),
        get_parallel=lambda: SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
        GPU_MEMORY_TYPE_KV_CACHE="kv",
        get_tensor_size_bytes=lambda buffer: buffer.nbytes,
        unwrap_write_loc=lambda locations: (locations, None, None),
        GB=1 << 30,
        MHATokenToKVPool=type("MHA", (), {}),
        KVCache=PoolBase,
    )
    hybrid = class_node(memory_pool, "HybridLinearKVPool")
    hybrid.body = [
        node
        for node in hybrid.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "__init__",
            "get_kv_size_bytes",
            "prefill_layer_slots",
            "prefill_offload_runtime",
            "decode_host_backing",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "layer_num",
            "store_dtype",
        }
    ]
    compile_nodes(
        [
            mla,
            class_node(
                SRT / "hardware_backend/npu/memory_pool_npu.py", "NPUMLATokenToKVPool"
            ),
            hybrid,
        ],
        namespace,
    )
    return SimpleNamespace(
        pool=namespace["NPUMLATokenToKVPool"], hybrid=namespace["HybridLinearKVPool"]
    )


def make_pool(pool_type, **overrides):
    values = dict(
        size=256,
        page_size=128,
        dtype=torch.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=5,
        device="cpu",
        enable_memory_saver=False,
        index_head_dim=128,
        indexer_layer_ids=(0, 2, 4),
        index_kpool=4,
        index_kpool_compress=True,
        max_running_requests=2,
        skip_topk_layers=[False, True, False, True, False],
    )
    values.update(overrides)
    return pool_type(**values)


def test_bounded_slots_describe_real_hbm_without_preparing_layers(pool_types):
    pool = make_pool(pool_types.pool, prefill_layer_slots=2)
    (description,) = pool.get_device_pool_infos()
    description.buffer_info.validate(layer_ids=description.layer_ids)
    assert description.layer_ids == (0, 1)
    assert description.buffer_info.device_layer_num == 2
    assert description.buffer_info.buffers.k.shape[0] == 2
    assert description.buffer_info.buffers.index_layer_indices == (0, 2, 4)
    assert pool.raw_layer_components(4)[0].data_ptr() == pool.k_buffer[0].data_ptr()
    assert (
        pool.raw_layer_components(4)[2].data_ptr() == pool.index_k_buffer[2].data_ptr()
    )
    assert pool.prefill_offload_runtime is None


def test_decode_host_views_are_excluded_from_device_buffers_and_bytes(pool_types):
    pool = make_pool(pool_types.pool, decode_host_budget=1 << 30)
    (description,) = pool.get_device_pool_infos()
    description.buffer_info.validate(layer_ids=description.layer_ids)
    assert description.layer_ids == ()
    assert description.buffer_info.device_layer_num == 0
    assert description.buffer_info.buffers.k is None
    assert description.buffer_info.buffers.v is None
    assert description.buffer_info.buffers.index_k is pool.index_k_buffer
    assert pool.get_kv_size_bytes() == pool.index_k_buffer.nbytes
    assert pool.get_kv_buffer(4) is pool.decode_host_backing.components[4]
    with pytest.raises(RuntimeError, match="staging"):
        pool.get_contiguous_buf_infos()


def test_target_draft_share_request_identity_and_keep_separate_payloads(pool_types):
    target = make_pool(
        pool_types.pool,
        prefill_layer_slots=2,
        decode_host_budget=1 << 30,
        integrated_offload=True,
    )
    draft = make_pool(
        pool_types.pool,
        prefill_layer_slots=1,
        decode_host_budget=1 << 30,
        integrated_offload=True,
    )
    request_pool = object()
    target.prefill_offload_runtime = SimpleNamespace(request_pool=request_pool)
    draft.prefill_offload_runtime = SimpleNamespace(request_pool=request_pool)
    assert target.decode_host_backing is not draft.decode_host_backing
    assert target.k_buffer.data_ptr() != draft.k_buffer.data_ptr()
    assert target.index_k_buffer.data_ptr() != draft.index_k_buffer.data_ptr()


def test_hybrid_forwards_offload_options_and_preserves_wire_layer_ids(pool_types):
    pool = pool_types.hybrid(
        size=256,
        dtype=torch.bfloat16,
        page_size=128,
        head_num=1,
        head_dim=576,
        full_attention_layer_ids=[2, 6, 9],
        device="cpu",
        mamba_pool=object(),
        use_mla=True,
        use_dsa=True,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        kv_cache_dim=576,
        index_kpool=4,
        index_kpool_compress=True,
        max_running_requests=2,
        full_kv_pool_class=pool_types.pool,
        fsa_pool_kwargs=dict(
            prefill_layer_slots=1, decode_host_budget=1 << 30, integrated_offload=True
        ),
    )
    assert pool.layer_num == 3
    assert pool.prefill_layer_slots == 1
    assert pool.full_kv_pool.fsa_wire_layer_ids == (2, 6, 9)
    (description,) = pool.full_kv_pool.get_device_pool_infos()
    assert description.layer_ids == (2,)
    runtime = object()
    pool.prefill_offload_runtime = runtime
    assert pool.full_kv_pool.prefill_offload_runtime is runtime


def test_closed_offload_retains_full_device_layer_axis(pool_types):
    pool = make_pool(pool_types.pool)
    (description,) = pool.get_device_pool_infos()
    assert description.layer_ids == (0, 1, 2, 3, 4)
    assert pool.k_buffer.shape == (5, 3, 128, 1, 512)
    assert pool.decode_host_backing is None


if __name__ == "__main__":
    run_pytest(__file__)
