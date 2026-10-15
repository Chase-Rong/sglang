"""Offload service and target/draft admission contracts; execution is remote-only."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).parents[4]
    / "python/sglang/srt/hardware_backend/npu/offload_config.py"
)
SPEC = importlib.util.spec_from_file_location("offload_config_under_test", SOURCE)
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
        {"enable_hisparse": True},
        {"disable_radix_cache": False},
        {"optimistic_prefill_attempts": 1},
    ],
)
def test_incompatible_prefill_lifecycle_rejected_before_pool_creation(changes):
    with pytest.raises(ValueError):
        MODULE.validate_offload_service(service(**changes), "prefill")


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
