"""Use real runtime namespaces in a process separate from module-stub control UTs."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu.decode_memory_budget import budget_from_config
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.runtime_context import get_context, get_disagg, get_exec, reset_context
from sglang.srt.server_args import ServerArgs


@pytest.fixture(autouse=True)
def runtime_config():
    reset_context()
    get_context().set_server_args(ServerArgs(model_path="dummy"))
    try:
        yield
    finally:
        reset_context()


@pytest.mark.parametrize(
    "backend,graph_max,expected_rows",
    [
        ("full", 4, 4),
        ("disabled", 4, 0),
        ("full", 32, 32),
        ("disabled", 32, 0),
    ],
)
def test_budget_uses_published_graph_config_without_removed_server_fields(
    monkeypatch, backend, graph_max, expected_rows
):
    monkeypatch.setenv("SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD", "1")
    monkeypatch.setenv("SGLANG_NPU_DECODE_HOST_KV_BYTES", str(1 << 34))
    config = SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"],
        index_topk=2048,
        index_kpool=4,
    )
    model = SimpleNamespace(
        hf_config=config,
        hf_text_config=config,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_head_dim=128,
    )
    kvc = SimpleNamespace(
        device="npu",
        use_mla_backend=True,
        model_config=model,
        kv_cache_dtype=torch.bfloat16,
        mambaish_config=None,
        layer_info=SimpleNamespace(start_layer=0, end_layer=4),
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        is_draft_worker=False,
        server_args=SimpleNamespace(),
    )
    sizes = [bs for bs in (1, 2, 4, 8, 16, 32) if bs <= graph_max]
    graph = CudaGraphConfig(
        decode=PhaseConfig(backend=backend, bs=sizes, max_bs=graph_max)
    )
    with (
        get_disagg().override(disaggregation_mode="null"),
        get_exec().graph.override(cuda_graph_config=graph),
    ):
        actual = budget_from_config(kvc, 4)
    assert actual.graph_rows == expected_rows
    assert actual.prefill_layer_slots == 2
    assert actual.layers == 4


@pytest.mark.parametrize(
    "requests,graph_max,expected",
    [
        (8, 4, 8),
        (2, 8, 8),
        (28, 32, 32),
        (32, 32, 32),
        (64, 32, 64),
    ],
)
def test_fsa_reservation_covers_request_capacity_after_graph_lock(
    monkeypatch, requests, graph_max, expected
):
    from sglang.srt.hardware_backend.npu import fsa_backend, fsa_native

    monkeypatch.setenv("SGLANG_NPU_ENABLE_FUSED_SPARSE_ATTENTION", "1")
    monkeypatch.setattr(fsa_native, "native_ops", lambda: None)
    monkeypatch.setattr(fsa_backend, "sparse_layer_ids", lambda pool: [0])
    monkeypatch.setattr(
        fsa_backend,
        "FusedSparseAttentionState",
        lambda *args, **kwargs: SimpleNamespace(**kwargs),
    )
    config = SimpleNamespace(index_topk=2048, index_kpool=4)
    backend = SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        device="cpu",
        token_to_kv_pool=SimpleNamespace(),
        req_to_token_pool=SimpleNamespace(size=requests),
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config),
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
    )
    graph = CudaGraphConfig(
        decode=PhaseConfig(backend="full", bs=[1, graph_max], max_bs=graph_max)
    )
    with get_exec().graph.override(cuda_graph_config=graph):
        fsa_backend.initialize_fsa_backend(backend, runner)
    assert backend.fused_sparse_state.row_capacity == expected


def test_joint_result_can_be_stored_on_real_slotted_configurator(monkeypatch):
    from sglang.srt.hardware_backend.npu.decode_memory_budget import (
        DecodeKvMemoryBudget,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.model_executor import pool_configurator
    from sglang.srt.runtime_context import get_memory, get_schedule

    instance = object.__new__(KVCacheConfigurator)
    instance.ps = SimpleNamespace(attn_dp_size=1)
    instance.model_config = SimpleNamespace(context_len=8192)
    model = DecodeKvMemoryBudget(4, 1, 512, 0, 128, 2, 2048, 4096, 1 << 34)
    monkeypatch.setattr(
        pool_configurator,
        "create_memory_pool_configurator",
        lambda kvc: SimpleNamespace(_decode_budget=model),
    )
    with (
        get_memory().override(disable_radix_cache=True),
        get_schedule().override(
            page_size=64,
            max_total_tokens=8192,
            max_running_requests=8,
            max_mamba_cache_size=None,
        ),
    ):
        remaining = instance._resolve_joint_dsa_kda_budget(1.0, 1 << 20, 0, 0)
        assert instance._joint_capacity.requests == 8
        assert get_schedule().max_mamba_cache_size == 8
        assert 0 < remaining < 1


def test_real_configurator_preserves_requests_when_token_budget_shrinks(monkeypatch):
    from sglang.srt.hardware_backend.npu.decode_memory_budget import (
        DecodeKvMemoryBudget,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.model_executor import pool_configurator
    from sglang.srt.runtime_context import get_memory, get_schedule

    instance = object.__new__(KVCacheConfigurator)
    instance.ps = SimpleNamespace(attn_dp_size=1)
    instance.model_config = SimpleNamespace(context_len=1048576)
    model = DecodeKvMemoryBudget(4, 1, 512, 0, 128, 2, 2048, 4096, 1 << 34)
    monkeypatch.setattr(
        pool_configurator,
        "create_memory_pool_configurator",
        lambda kvc: SimpleNamespace(_decode_budget=model),
    )
    with (
        get_memory().override(disable_radix_cache=True),
        get_schedule().override(
            page_size=64,
            max_total_tokens=1048576,
            max_running_requests=8,
            max_mamba_cache_size=None,
        ),
    ):
        instance._resolve_joint_dsa_kda_budget(0.2, 1 << 20, 0, 0)
        assert instance._joint_capacity.target_tokens < 1048576
        assert (
            instance.resolve_max_num_reqs(instance._joint_capacity.target_tokens) == 8
        )
        assert get_schedule().max_running_requests == 8


def test_worker_single_request_limit_uses_shared_pool_without_dividing_by_concurrency():
    from sglang.srt.managers.tp_worker import TpModelWorker

    runner = SimpleNamespace(
        effective_max_total_num_tokens=8192,
        max_total_num_tokens=8192,
        max_running_requests=8,
        forward_stream=None,
        req_to_token_pool=SimpleNamespace(
            size=8,
            max_context_len=16384,
            schedulable_token_capacity=lambda tokens: tokens,
        ),
        token_to_kv_pool=SimpleNamespace(size=8192),
    )
    worker = SimpleNamespace(
        model_config=SimpleNamespace(context_len=16384),
        model_runner=runner,
        ps=SimpleNamespace(attn_dcp_size=1),
        random_seed=0,
        device="cpu",
    )
    actual = TpModelWorker.get_worker_info(worker)
    assert actual[2] == 8
    assert actual[4] == 8191
    assert actual[5] == 8186


def test_real_configurator_caps_requests_and_emits_warning(monkeypatch, caplog):
    from sglang.srt.hardware_backend.npu.decode_memory_budget import (
        DecodeKvMemoryBudget,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.model_executor import pool_configurator
    from sglang.srt.runtime_context import get_memory, get_schedule

    instance = object.__new__(KVCacheConfigurator)
    instance.ps = SimpleNamespace(attn_dp_size=1)
    instance.model_config = SimpleNamespace(context_len=8192)
    model = DecodeKvMemoryBudget(4, 1, 512, 0, 128, 2, 2048, 4096, 1 << 34)
    monkeypatch.setattr(
        pool_configurator,
        "create_memory_pool_configurator",
        lambda kvc: SimpleNamespace(_decode_budget=model),
    )
    with (
        get_memory().override(disable_radix_cache=True),
        get_schedule().override(
            page_size=64,
            max_total_tokens=8192,
            max_running_requests=8,
            max_mamba_cache_size=4,
        ),
    ):
        instance._resolve_joint_dsa_kda_budget(1.0, 1 << 20, 0, 0)
        assert instance._joint_capacity.requests == 4
        assert instance.resolve_max_num_reqs(8192) == 4
        assert get_schedule().max_running_requests == 8
        assert "reduced from the requested 8 to 4" in caplog.text
