"""Execute production sizing methods without importing accelerator dependencies."""

import ast
import sys
from types import SimpleNamespace
from typing import Optional

import pytest
from test_decode_memory_budget import MODULE, SOURCE, budget


@pytest.fixture
def runtime(monkeypatch):
    source = SOURCE.parents[2] / "mem_cache/kv_cache_configurator.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    methods = {
        "_handle_max_mamba_cache",
        "_resolve_joint_dsa_kda_budget",
        "config_from_budget",
        "resolve_max_num_reqs",
    }
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheConfigurator"
    )
    definition.bases = []
    definition.decorator_list = []
    definition.body = [
        node
        for node in definition.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    schedule = SimpleNamespace(
        max_running_requests=8,
        max_mamba_cache_size=None,
        max_total_tokens=8192,
        page_size=64,
    )
    memory = SimpleNamespace(disable_radix_cache=True, enable_unified_memory=False)
    calls = []
    warnings = []
    model = budget()
    pool = SimpleNamespace(
        _decode_budget=model,
        calculate_pool_sizes_from_max_tokens=lambda tokens, page: SimpleNamespace(
            max_total_num_tokens=tokens
        ),
    )

    def create_pool(configurator):
        calls.append("create_pool")
        return pool

    def override(reason, **values):
        for key, value in values.items():
            setattr(schedule, key, value)

    namespace = dict(
        Optional=Optional,
        MemoryPoolConfig=object,
        get_schedule=lambda: schedule,
        get_memory=lambda: memory,
        get_req_to_token_extra_context_len=lambda: 4,
        get_context=lambda: SimpleNamespace(override=override),
        get_exec=lambda: SimpleNamespace(
            mamba=SimpleNamespace(enable_linear_replayssm_spec=False)
        ),
        logger=SimpleNamespace(
            info=lambda *args: calls.append(args),
            warning=lambda *args: warnings.append(args),
        ),
    )
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    instance = namespace["KVCacheConfigurator"]()
    instance.ps = SimpleNamespace(attn_dp_size=1, pp_size=1)
    instance.model_config = SimpleNamespace(context_len=8192)
    instance.mambaish_config = SimpleNamespace(
        mamba2_cache_params=SimpleNamespace(layers=[1, 2], mamba_cache_per_req=1 << 20)
    )
    instance.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    instance._calculate_mamba_ratio = lambda: 1
    instance._apply_token_constraints = lambda tokens: tokens // 64 * 64
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.pool_configurator",
        SimpleNamespace(create_memory_pool_configurator=create_pool),
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.hardware_backend.npu.decode_memory_budget", MODULE
    )
    return SimpleNamespace(
        instance=instance,
        schedule=schedule,
        memory=memory,
        calls=calls,
        warnings=warnings,
        pool=pool,
        model=model,
    )


def test_production_entry_reads_budget_from_pool_configurator(runtime):
    instance = runtime.instance
    assert not hasattr(instance, "_decode_budget")
    remaining_gib = instance._handle_max_mamba_cache(1.0)
    result = instance._joint_capacity
    assert runtime.calls[0] == "create_pool"
    assert result.state_slots == result.requests == 8
    assert runtime.schedule.max_mamba_cache_size == 8
    assert remaining_gib == ((1 << 30) - result.kda_bytes - result.request_bytes) / (
        1 << 30
    )
    assert result.request_bytes == 9 * ((8192 + 4) * 4 + 28)
    assert result.dsa_bytes == runtime.model.required_bytes(8192, 8, 64)[1]


def test_production_token_solve_does_not_charge_dsa_twice(runtime):
    instance = runtime.instance
    available = instance._handle_max_mamba_cache(1.0)
    config = instance.config_from_budget(int(available * (1 << 30)))
    assert config.max_total_num_tokens == 8192
    assert instance.resolve_max_num_reqs(8192) == 8


def test_no_decode_pool_keeps_legacy_budget_path(runtime):
    runtime.pool._decode_budget = None
    assert runtime.instance._resolve_joint_dsa_kda_budget(1.0, 1 << 20, 0, 0) is None
    assert not hasattr(runtime.instance, "_joint_capacity")


def test_offload_requires_explicit_request_limit(runtime):
    runtime.schedule.max_running_requests = None
    with pytest.raises(ValueError, match="explicit --max-running-requests"):
        runtime.instance._handle_max_mamba_cache(1.0)


def test_explicit_requests_survive_insufficient_memory_for_token_cap(runtime):
    runtime.schedule.max_total_tokens = 1048576
    runtime.instance.model_config.context_len = 1048576
    instance = runtime.instance
    remaining = instance._handle_max_mamba_cache(0.2)
    result = instance._joint_capacity
    assert result.requests == result.state_slots == 8
    assert result.target_tokens < 1048576
    assert instance.resolve_max_num_reqs(result.target_tokens) == 8
    config = instance.config_from_budget(int(remaining * (1 << 30)))
    assert config.max_total_num_tokens == result.target_tokens


def test_fixed_requests_no_total_cap_can_allocate_more_than_one_context(runtime):
    runtime.schedule.max_total_tokens = None
    instance = runtime.instance
    instance._handle_max_mamba_cache(1.0)
    assert instance._joint_capacity.target_tokens == 8 * 8192
    assert instance._joint_capacity.requests == 8


def test_request_priority_reduces_requests_after_downstream_token_cap(runtime):
    runtime.instance._handle_max_mamba_cache(1.0)
    config = runtime.instance.config_from_budget(1 << 30, cap_tokens=64)
    assert config.max_total_num_tokens == 64
    assert runtime.instance.resolve_max_num_reqs(64) == 1
    assert runtime.warnings[-1][1:] == (8, 1)


def test_explicit_mamba_caps_requested_concurrency_with_warning(runtime):
    runtime.schedule.max_mamba_cache_size = 4
    runtime.instance._handle_max_mamba_cache(1.0)
    assert runtime.instance._joint_capacity.requests == 4
    assert runtime.instance.resolve_max_num_reqs(8192) == 4
    assert runtime.schedule.max_running_requests == 8
    assert runtime.warnings[-1][1:] == (8, 4)


def test_request_priority_memory_shortfall_reduces_concurrency_with_warning(runtime):
    model = runtime.model
    mapping_per_row = (8192 + 4) * 4 + 28
    available = model.required_bytes(3 * 64, 3, 64)[1] + 4 * (
        (1 << 20) + mapping_per_row
    )
    runtime.instance._handle_max_mamba_cache(available / (1 << 30))
    assert runtime.instance._joint_capacity.requests == 3
    assert runtime.schedule.max_mamba_cache_size == 3
    assert runtime.schedule.max_running_requests == 8
    assert runtime.instance.resolve_max_num_reqs(3 * 64) == 3
    assert runtime.warnings[-1][1:] == (8, 3)


def test_request_priority_no_warning_when_requested_concurrency_fits(runtime):
    runtime.instance._handle_max_mamba_cache(1.0)
    assert runtime.instance.resolve_max_num_reqs(8192) == 8
    assert not runtime.warnings


def test_request_priority_final_capacity_cannot_support_zero_requests(runtime):
    runtime.instance._handle_max_mamba_cache(1.0)
    with pytest.raises(MemoryError, match="cannot support one request"):
        runtime.instance.resolve_max_num_reqs(0)


def test_fixed_requests_are_sharded_exactly_over_attention_dp(runtime):
    runtime.instance.ps.attn_dp_size = 2
    runtime.instance._handle_max_mamba_cache(1.0)
    assert runtime.instance._joint_capacity.requests == 4
    assert runtime.instance.resolve_max_num_reqs(8192) == 4


def test_fixed_requests_reject_nondivisible_attention_dp(runtime):
    runtime.instance.ps.attn_dp_size = 3
    with pytest.raises(ValueError, match="divisible"):
        runtime.instance._handle_max_mamba_cache(1.0)
