"""Offload service and target/draft admission contracts; execution is remote-only."""

import ast
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import npu_module_name, npu_source_path, run_pytest

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SOURCE = (
    Path(__file__).parents[4]
    / "python/sglang/srt/hardware_backend/npu/hisparse/offload_config.py"
)
SPEC = importlib.util.spec_from_file_location(
    npu_module_name("offload_config_under_test"), npu_source_path(SOURCE)
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "changes",
    [
        {"integrated": False},
        {"prefill": False},
        {"mixed_chunk": True},
        {"cp_size": 2},
        {"dcp_size": 2},
    ],
)
def test_prefill_overlap_rejects_unsupported_modes(changes):
    options = dict(
        integrated=True,
        prefill=True,
        speculative=False,
        mixed_chunk=False,
        cp_size=1,
        dcp_size=1,
    )
    MODULE.validate_prefill_overlap(**options)
    options.update(changes)
    with pytest.raises(ValueError, match="Prefill copy overlap"):
        MODULE.validate_prefill_overlap(**options)


def test_prefill_overlap_and_mtp_can_coexist():
    MODULE.validate_prefill_overlap(
        integrated=True,
        prefill=True,
        speculative=True,
        mixed_chunk=False,
        cp_size=1,
        dcp_size=1,
    )


def service(**changes):
    values = dict(
        get_attention_backends=lambda: ("ascend", "ascend"),
        disaggregation_transfer_backend="ascend",
        disable_radix_cache=True,
        enable_hisparse=False,
        optimistic_prefill_attempts=0,
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_separate_ascend_services_are_admitted(role):
    values = (
        {}
        if role == "prefill"
        else dict(disable_radix_cache=False, enable_hierarchical_cache=True)
    )
    MODULE.validate_offload_service(service(**values), role)


def test_separate_decode_rejects_unmanaged_host_pages():
    with pytest.raises(ValueError, match="HiCache"):
        MODULE.validate_offload_service(service(), "decode")


@pytest.mark.parametrize(
    "role,host,integrated,expected",
    [
        ("decode", True, False, 64 * 1024 * 1024),
        ("prefill", True, False, 0),
        ("decode", False, False, 0),
        ("null", True, True, 0),
    ],
)
def test_layer_staging_default_is_only_allocated_for_pd_host_decode(
    role, host, integrated, expected
):
    assert (
        MODULE.layer_staging_bytes(
            0, role=role, host_offload=host, integrated=integrated
        )
        == expected
    )


@pytest.mark.parametrize("size", [-32, 1, 31, 33, True])
def test_invalid_layer_staging_size_is_rejected(size):
    with pytest.raises(ValueError, match="32-byte"):
        MODULE.layer_staging_bytes(size, role="decode", host_offload=True)


def test_explicit_staging_size_is_preserved():
    assert MODULE.layer_staging_bytes(4096, role="decode", host_offload=True) == 4096


def test_integrated_service_needs_explicit_admission_and_no_transport():
    MODULE.validate_offload_service(
        service(
            disaggregation_transfer_backend="unused",
            disable_radix_cache=False,
            enable_hierarchical_cache=True,
        ),
        "null",
        integrated=True,
    )
    with pytest.raises(ValueError):
        MODULE.validate_offload_service(service(), "null")
    with pytest.raises(ValueError):
        MODULE.validate_offload_service(service(), "prefill", integrated=True)
    with pytest.raises(ValueError):
        MODULE.validate_offload_service(
            service(disable_radix_cache=False), "null", integrated=True
        )
    with pytest.raises(ValueError):
        MODULE.validate_offload_service(
            service(enable_hierarchical_cache=True), "null", integrated=True
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"get_attention_backends": lambda: ("ascend", "triton")},
        {"get_attention_backends": lambda: ("triton", "ascend")},
        {"disaggregation_transfer_backend": "nixl"},
        {"disable_radix_cache": False},
        {"optimistic_prefill_attempts": 1},
    ],
)
def test_incompatible_prefill_lifecycle_rejected_before_pool_creation(changes):
    with pytest.raises(ValueError):
        MODULE.validate_offload_service(service(**changes), "prefill")


@pytest.mark.parametrize("role", ["null", "prefill", "decode"])
def test_hisparse_reuses_existing_offload_service_contract(role):
    values = dict(enable_hisparse=True)
    if role != "prefill":
        values.update(disable_radix_cache=False, enable_hierarchical_cache=True)
    MODULE.validate_offload_service(service(**values), role, integrated=role == "null")


def pool(**changes):
    values = dict(
        page_size=128,
        size=8192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        store_dtype="bf16",
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "name,value",
    [
        ("page_size", 64),
        ("size", 4096),
        ("kv_lora_rank", 256),
        ("qk_rope_head_dim", 32),
        ("index_head_dim", 64),
        ("store_dtype", "fp16"),
    ],
)
def test_mtp_geometry_cannot_invalidate_joint_memory_budget(name, value):
    with pytest.raises(ValueError, match=name):
        MODULE.validate_mtp_pool_pair(pool(), pool(**{name: value}))


def test_prefill_target_and_draft_share_scheduler_request_mapping():
    owner = object()
    target = pool(prefill_offload_runtime=SimpleNamespace(request_pool=owner))
    draft = pool(prefill_offload_runtime=SimpleNamespace(request_pool=owner))
    MODULE.validate_mtp_pool_pair(target, draft)
    draft.prefill_offload_runtime.request_pool = object()
    with pytest.raises(ValueError, match="ownership"):
        MODULE.validate_mtp_pool_pair(target, draft)


@pytest.mark.parametrize(
    "logical,override,visible,expected",
    [
        (0, -1, "4,5,6,7", 4),
        (3, -1, "4,5,6,7", 7),
        (1, 5, "4,5", 5),
        (2, -1, None, 2),
        (0, 6, None, 6),
    ],
)
def test_each_rank_uses_its_physical_ascend_transfer_device(
    logical, override, visible, expected
):
    assert MODULE.resolve_transfer_device(logical, override, visible) == expected


@pytest.mark.parametrize(
    "logical,override,visible",
    [
        (0, 0, "4,5"),
        (1, -1, "4"),
        (0, -1, "0,0"),
        (0, -1, ""),
        (0, -1, "0-3"),
        (-1, -1, "0"),
    ],
)
def test_wrong_rank_mapping_rejected_before_registering_memory(
    logical, override, visible
):
    with pytest.raises(ValueError):
        MODULE.resolve_transfer_device(logical, override, visible)


@pytest.mark.parametrize(
    "name,value",
    [
        ("index_size", 4096),
        ("index_page_size", 64),
        ("index_kpool", 2),
        ("_kpool_use_compress", False),
    ],
)
def test_target_draft_index_layouts_must_match(name, value):
    target = pool(
        index_size=8192, index_page_size=128, index_kpool=4, _kpool_use_compress=True
    )
    draft = pool(**vars(target))
    setattr(draft, name, value)
    with pytest.raises(ValueError, match=name):
        MODULE.validate_mtp_pool_pair(target, draft)


def test_disabled_pool_options_do_not_import_native_or_shared_memory(monkeypatch):
    flags = SimpleNamespace(
        SGLANG_NPU_ENABLE_PREFILL_LAYERWISE_OFFLOAD=SimpleNamespace(get=lambda: False),
        SGLANG_NPU_DECODE_HOST_KV_BYTES=SimpleNamespace(get=lambda: 0),
        SGLANG_NPU_ENABLE_INTEGRATED_OFFLOAD=SimpleNamespace(get=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", SimpleNamespace(envs=flags))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.arg_groups.overrides",
        SimpleNamespace(resolved_view=lambda value: value),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(get_disagg=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.hardware_backend.npu.fsa_native", None)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.hardware_backend.npu.shared_host_kv", None
    )
    assert MODULE.fsa_pool_options(SimpleNamespace(), 4) == {}


@pytest.fixture
def host_admission(monkeypatch):
    from test_decode_memory_budget import MODULE as budget_module

    def install(name, **values):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**values))

    state = SimpleNamespace(
        shared=True,
        local_ranks=8,
        rank_budget=1 << 30,
        identity=("host", "cgroup"),
        reports=None,
    )
    group = SimpleNamespace(world_size=4)
    group.all_gather_object = lambda report: (
        state.reports if state.reports is not None else [report] * group.world_size
    )
    install(
        "sglang.srt.environ",
        envs=SimpleNamespace(
            SGLANG_NPU_SHARED_HOST_KV=SimpleNamespace(get=lambda: state.shared)
        ),
    )
    install("sglang.srt.distributed.parallel_state", get_attn_tp_group=lambda: group)
    install(
        "sglang.srt.mem_cache.pool_host.base",
        ranks_per_host=lambda: state.local_ranks,
        host_memory_budget_bytes=lambda **kwargs: state.rank_budget,
    )
    install(
        "sglang.srt.hardware_backend.npu.hisparse.decode_memory_budget",
        HostMemoryBudgetPlan=budget_module.HostMemoryBudgetPlan,
    )
    monkeypatch.setattr(MODULE, "host_memory_scope_identity", lambda: state.identity)
    return state


def test_admission_uses_shared_domain_fraction_and_leaves_private_rank_allowances(
    host_admission,
):
    plan = MODULE.prepare_host_memory_budget_plan((1 << 30) + 1)
    assert plan.scope_available_bytes == 4 << 30
    assert plan.shared_ranks == 4
    assert plan.aligned_reserve_bytes == 2 << 30
    assert plan.sidecar_budget_bytes == 1 << 29
    assert plan.scope_identity == host_admission.identity


def test_unshared_admission_cannot_borrow_other_ranks_headroom(host_admission):
    host_admission.shared = False
    with pytest.raises(MemoryError, match="cgroup/OS"):
        MODULE.prepare_host_memory_budget_plan((1 << 30) + 1)


def test_shared_admission_rejects_distinct_cgroup_scopes(host_admission):
    host_admission.reports = [
        ((("host", "a"), 1 << 30, 1 << 30, 8), 1 << 30, None),
        ((("host", "b"), 1 << 30, 1 << 30, 8), 1 << 30, None),
    ]
    with pytest.raises(ValueError, match="cgroup"):
        MODULE.prepare_host_memory_budget_plan(1 << 30)


def test_shared_admission_propagates_discovery_errors_before_allocation(host_admission):
    host_admission.reports = [((), 0, "cgroup usage is unreadable")]
    with pytest.raises(RuntimeError, match="unreadable"):
        MODULE.prepare_host_memory_budget_plan(1 << 30)


def test_draft_reuses_target_plan_without_resampling_host_memory(monkeypatch):
    from test_decode_memory_budget import MODULE as budget_module

    plan = budget_module.HostMemoryBudgetPlan(1 << 30, 2 << 30)
    target = SimpleNamespace(host_memory_budget_plan=plan)
    draft = SimpleNamespace(
        is_draft_worker=True,
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(full_kv_pool=target),
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "prepare_host_memory_budget_plan",
        lambda *args, **kwargs: pytest.fail("resampled Host budget"),
    )
    assert MODULE._pool_host_memory_plan(draft, 1 << 30) is plan
    assert draft._offload_host_memory_plan is plan


def test_sidecar_scope_reuses_preallocation_allowance_and_restores_outer_scope(
    monkeypatch,
):
    from test_decode_memory_budget import MODULE as budget_module

    state = SimpleNamespace(remaining=123)

    @contextmanager
    def scope(value):
        before = state.remaining
        state.remaining = value
        try:
            yield
        finally:
            state.remaining = before

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.pool_host.base",
        SimpleNamespace(host_memory_budget_scope=scope),
    )
    plan = budget_module.HostMemoryBudgetPlan(
        1 << 30, 4 << 30, shared_ranks=4, planner_bytes_per_rank=4096
    )
    pool = SimpleNamespace(full_kv_pool=SimpleNamespace(host_memory_budget_plan=plan))
    with MODULE.offload_host_memory_budget_scope(pool):
        assert state.remaining == plan.sidecar_budget_bytes
        state.remaining -= 8192
    assert state.remaining == 123


def test_closed_offload_preserves_existing_host_scope(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.pool_host.base",
        SimpleNamespace(
            host_memory_budget_scope=lambda _: pytest.fail("unexpected scope")
        ),
    )
    with MODULE.offload_host_memory_budget_scope(SimpleNamespace()):
        pass


def test_backing_snapshot_and_planner_use_the_same_account_as_sidecar_build(
    monkeypatch,
):
    from test_decode_memory_budget import MODULE as budget_module

    path = SOURCE.parents[3] / "mem_cache/pool_host/base.py"
    definition = next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "HostMemoryAllocationAccount"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    state = SimpleNamespace(remaining=None)

    @contextmanager
    def scope(value):
        before = state.remaining
        state.remaining = value
        try:
            yield
        finally:
            state.remaining = before

    @contextmanager
    def ensure(account):
        with scope(account.remaining_bytes):
            yield
            account.remaining_bytes = state.remaining

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.mem_cache.pool_host.base",
        SimpleNamespace(
            host_memory_budget_scope=scope,
            ensure_host_memory_budget_scope=ensure,
        ),
    )
    plan = budget_module.HostMemoryBudgetPlan(
        1 << 30, 4 << 30, shared_ranks=4, planner_bytes_per_rank=4096
    )
    with MODULE.offload_host_backing_budget_scope(plan):
        account = namespace["HostMemoryAllocationAccount"](state.remaining)
        account.charge("memfabric", plan.aligned_reserve_bytes // plan.shared_ranks)
    target = SimpleNamespace(host_memory_account=account)
    draft = SimpleNamespace(host_memory_account=account)
    MODULE.bind_host_memory_budget_plan(target, plan)
    MODULE.bind_host_memory_budget_plan(draft, plan)
    assert account.remaining_bytes == plan.sidecar_budget_bytes
    with MODULE.offload_host_memory_budget_scope(
        SimpleNamespace(decode_host_backing=target)
    ):
        assert state.remaining == plan.sidecar_budget_bytes
        state.remaining -= 8192
    assert account.remaining_bytes == plan.sidecar_budget_bytes - 8192
    assert set(account.reservations) == {"memfabric", "offload_planner"}


if __name__ == "__main__":
    run_pytest(__file__)
