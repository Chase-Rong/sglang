"""CPU contracts for current-main PD cancellation, restore and ownership."""

import logging
import sys
from contextlib import nullcontext
from enum import Enum
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import test_layer_receive_protocol as protocol_tests
import torch
from test_hisparse_pd_dispatch import SRT, production_definitions

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

POLL = NS(Failed=0, Bootstrapping=1, WaitingForInput=2, Transferring=3, Success=4)
RESTORE = NS(FAILED="failed", PENDING="pending", READY="ready")


@pytest.mark.parametrize("offload", [False, True])
def test_real_state_component_retains_explicit_layer_map(offload):
    from sglang.srt.disaggregation.base.conn import StateType
    from sglang.srt.disaggregation.utils import append_state_component

    args = NS(
        prefill_layerwise_offload=offload,
        kv_data_ptrs=[4096, 8192],
        kv_layer_ids=[3, 7],
        state_types=[],
        state_data_ptrs=[],
        state_data_lens=[],
        state_item_lens=[],
        state_dim_per_tensor=[],
        state_conv_shard_groups=[],
        state_slice_outer_counts=[],
        state_layer_ids=[],
    )
    append_state_component(
        args, StateType.DSA, [8192, 4096], [128, 128], [64, 64], layer_ids=[1, 2]
    )
    assert args.state_types == [StateType.DSA]
    assert args.state_layer_ids == ([[7, 3]] if offload else [[1, 2]])
    append_state_component(args, StateType.MAMBA, [12288], [256], [128], layer_ids=[5])
    assert args.state_layer_ids[-1] == [5]


def definitions(filename, classes, **namespace):
    return production_definitions(
        SRT / "disaggregation" / filename,
        classes,
        KVPoll=POLL,
        HiCacheRestoreResult=RESTORE,
        logger=logging.getLogger(__name__),
        torch=torch,
        np=np,
        **namespace,
    )


@pytest.fixture
def layer_modules(monkeypatch):
    return protocol_tests.modules.__wrapped__(monkeypatch)


def test_force_arm_is_forwarded_without_a_layer_lease():
    forwarded = Mock()

    class ReceiverBase:
        def _send_abort_notification(self, *, force_arm=False):
            forwarded(force_arm=force_arm)

    module = definitions(
        "mooncake/conn.py",
        {"MooncakeKVReceiver": {"_send_abort_notification"}},
        bases={"MooncakeKVReceiver": ReceiverBase},
    )
    receiver = module.MooncakeKVReceiver()
    receiver.bootstrap_room = 5
    receiver.kv_mgr = NS(_layer_leases={})
    receiver._send_abort_notification(force_arm=True)
    forwarded.assert_called_once_with(force_arm=True)


def test_partial_metadata_force_arm_retains_the_published_writer(layer_modules):
    module = definitions(
        "mooncake/conn.py", {"MooncakeKVReceiver": {"_send_abort_notification"}}
    )
    receiver = module.MooncakeKVReceiver()
    lease, _, _, _, _ = protocol_tests.lease_inputs(layer_modules)
    lease.layer_writers = {4: ("first", 1), 8: ("second", 2)}
    staging = NS(abort=Mock())
    manager = NS(
        _layer_leases={5: lease},
        _layer_staging=staging,
        enable_deferred_decode_kv_release=True,
        register_deferred_abort_room=Mock(return_value=9),
        local_ip="decode",
        rank_port=3,
    )
    sock = NS(send_multipart=Mock())
    receiver.bootstrap_room, receiver.session_id = 5, lease.session_id
    receiver.init_time = None
    receiver.kv_mgr = manager
    receiver.bootstrap_infos = [
        {"rank_ip": "first", "rank_port": 1, "is_dummy": False},
        {"rank_ip": "second", "rank_port": 2, "is_dummy": False},
    ]
    receiver._layer_published_peers = {("first", 1)}
    receiver._connect_to_bootstrap_server = Mock(return_value=(sock, nullcontext()))
    receiver._send_abort_notification(force_arm=True)
    manager.register_deferred_abort_room.assert_called_once_with(5)
    assert lease.abort_generation == 9 and lease.failed and not lease.drained
    assert lease.drained_writers == {("second", 2)}
    staging.abort.assert_not_called()
    header, message = sock.send_multipart.call_args.args[0]
    assert header == b"FSA_LAYER_ABORT"
    assert protocol_tests.wire_module().decode(message)["abort_generation"] == 9


def test_unpublished_lease_is_drained_without_sending_abort(layer_modules):
    module = definitions(
        "mooncake/conn.py", {"MooncakeKVReceiver": {"_send_abort_notification"}}
    )
    receiver = module.MooncakeKVReceiver()
    lease, _, _, _, _ = protocol_tests.lease_inputs(layer_modules)
    lease.layer_writers = {4: ("first", 1)}
    receiver.bootstrap_room, receiver.session_id = 5, lease.session_id
    receiver.init_time = None
    receiver._layer_published_peers = set()
    receiver.kv_mgr = NS(
        _layer_leases={5: lease},
        _layer_staging=NS(abort=Mock()),
        enable_deferred_decode_kv_release=False,
    )
    receiver._send_abort_notification(force_arm=True)
    assert lease.drained
    receiver.kv_mgr._layer_staging.abort.assert_called_once_with(
        lease, transport_drained=True
    )


def test_layer_abort_ack_requires_current_generation_and_all_writers(layer_modules):
    module = definitions(
        "mooncake/conn.py", {"MooncakeKVManager": {"_handle_layer_message"}}
    )
    manager = module.MooncakeKVManager()
    lease, _, _, _, _ = protocol_tests.lease_inputs(layer_modules)
    lease.layer_writers = {4: ("first", 1), 8: ("second", 2)}
    lease.abort_generation = 9
    manager._layer_leases = {5: lease}
    manager._layer_staging = NS(abort=Mock())
    manager.record_failure = Mock()
    manager.update_status = Mock()
    wire = protocol_tests.wire_module()
    payload = dict(
        room=5,
        session=lease.session_id,
        lease=lease.lease_id,
        reply_ip="first",
        reply_port=1,
        abort_generation=8,
    )
    manager._handle_layer_message([b"FSA_LAYER_ABORT_ACK", wire.encode(payload)])
    assert not lease.drained and not lease.drained_writers and not lease.failed
    payload["abort_generation"] = 9
    manager._handle_layer_message([b"FSA_LAYER_ABORT_ACK", wire.encode(payload)])
    assert not lease.drained
    payload.update(reply_ip="second", reply_port=2, lease="old-lease")
    manager._handle_layer_message([b"FSA_LAYER_ABORT_ACK", wire.encode(payload)])
    assert not lease.drained
    payload["lease"] = lease.lease_id
    manager._handle_layer_message([b"FSA_LAYER_ABORT_ACK", wire.encode(payload)])
    assert lease.drained
    manager._layer_staging.abort.assert_called_once_with(lease, transport_drained=True)
    manager.record_failure.assert_not_called()


def release_queue(*, strict, drained, now=100):
    module = definitions(
        "decode.py",
        {"DecodeTransferQueue": {"_defer_release", "resolve_deferred_releases"}},
        time=NS(monotonic=lambda: now),
    )
    queue = module.DecodeTransferQueue()
    manager = NS(
        requires_strict_abort_drain=lambda room: strict,
        is_abort_release_safe=Mock(return_value=drained),
    )
    request = NS(
        req=NS(bootstrap_room=5),
        metadata_buffer_index=7,
        host_staged=False,
        kv_receiver=NS(kv_mgr=manager, bootstrap_infos=[{}, {}]),
    )
    queue.enable_host_receive = False
    queue.gloo_group = object()
    queue.deferred_kv_release_timeout = 5
    queue._deferred_releases = []
    queue._do_release = Mock()
    queue._observe_deferred_kv_release = Mock()
    queue._defer_release(request)
    return queue, request, module


def test_strict_drain_timeout_keeps_all_five_deferred_fields(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    queue, request, module = release_queue(strict=True, drained=False)
    assert queue._deferred_releases == [(request, 100, 105, 7, 2)]
    module.DecodeTransferQueue.resolve_deferred_releases.__globals__["time"] = NS(
        monotonic=lambda: 200
    )
    with pytest.raises(RuntimeError, match="remain retained"):
        queue.resolve_deferred_releases()
    assert queue._deferred_releases == [(request, 100, 105, 7, 2)]
    queue._do_release.assert_not_called()


def test_strict_release_waits_for_tp_drain_consensus(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    reduce = Mock(side_effect=lambda tensor, **kwargs: tensor.zero_())
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    queue, request, _ = release_queue(strict=True, drained=True)
    queue.resolve_deferred_releases()
    assert queue._deferred_releases == [(request, 100, 105, 7, 2)]
    queue._do_release.assert_not_called()
    reduce.assert_called_once()
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **kwargs: None)
    queue.resolve_deferred_releases()
    assert queue._deferred_releases == []
    queue._do_release.assert_called_once_with(request, 7)


def test_device_timeout_keeps_main_release_behavior():
    queue, request, module = release_queue(strict=False, drained=False)
    module.DecodeTransferQueue.resolve_deferred_releases.__globals__["time"] = NS(
        monotonic=lambda: 200
    )
    queue.resolve_deferred_releases()
    queue._do_release.assert_called_once_with(request, 7)
    assert not queue._deferred_releases


@pytest.mark.parametrize(
    "restore,poll,expected",
    [
        (RESTORE.PENDING, POLL.Success, POLL.Transferring),
        (RESTORE.FAILED, POLL.Transferring, POLL.Failed),
        (RESTORE.READY, POLL.Success, POLL.Success),
    ],
)
def test_restore_result_enters_transport_consensus(restore, poll, expected):
    module = definitions(
        "decode_hicache_mixin.py", {"HiCacheRestoreGatedKVReceiver": None}
    )
    request = NS(hicache_restore_status=restore, kv_receiver=NS(poll=lambda: poll))
    assert module.HiCacheRestoreGatedKVReceiver(request).poll() == expected


def test_pending_restore_metadata_waits_for_tp_and_uses_restored_prefix(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, **kwargs: tensor.fill_(POLL.Transferring),
    )
    module = definitions(
        "decode.py",
        {"DecodeTransferQueue": {"_publish_pending_layer_metadata"}},
        kv_to_page_indices=lambda rows, page: rows.cpu().numpy()[::page] // page,
    )
    receiver = NS(
        poll=lambda: POLL.WaitingForInput,
        send_metadata=Mock(),
        kv_mgr=NS(kv_args=NS(page_size=1), update_status=Mock()),
    )
    state = [np.array([3]), [2, 1, 0]]
    request = NS(
        req=NS(rid="prefix", bootstrap_room=5, kv=NS(req_pool_idx=1)),
        kv_receiver=receiver,
        metadata_buffer_index=7,
        hicache_restore_status=RESTORE.READY,
        prefix_match=NS(l1_prefix_len=2, decode_prefix_len=4),
        hicache_restored_kv_indices=torch.tensor([20, 21]),
        npu_layer_metadata=(
            np.array([50, 51]),
            state,
            {
                "layer_token_count": 6,
                "decode_prefix_len": 4,
                "layer_device_kv_indices": np.array([10, 11, 0, 0, 50, 51]),
            },
        ),
    )
    queue = module.DecodeTransferQueue()
    queue.gloo_group = object()
    queue.queue = [request]
    queue.req_to_token_pool = NS(
        req_to_token=torch.tensor([[0, 0, 0, 0, 0, 0], [10, 11, 0, 0, 50, 51]])
    )
    queue._publish_pending_layer_metadata()
    receiver.send_metadata.assert_not_called()
    assert request.npu_layer_metadata is not None
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **kwargs: None)
    queue._publish_pending_layer_metadata()
    args, kwargs = receiver.send_metadata.call_args
    assert args[1] == 7 and args[2] is state
    np.testing.assert_array_equal(
        kwargs["layer_device_kv_indices"], [10, 11, 20, 21, 50, 51]
    )
    assert kwargs["decode_prefix_len"] == 4 and request.npu_layer_metadata is None


def test_host_backed_release_uses_checkpoint_api_after_retirement():
    events = []
    release = Mock(side_effect=lambda *args, **kwargs: events.append("release"))
    module = definitions(
        "decode.py",
        {"DecodeTransferQueue": {"_release_request"}},
        uses_native_hisparse=lambda enabled: False,
        release_kv_cache=release,
    )
    queue = module.DecodeTransferQueue()
    queue.enable_host_receive = False
    queue.tree_cache = object()
    queue.scheduler = NS(
        enable_hisparse=True,
        hisparse_coordinator=NS(request_finished=lambda req: events.append("retire")),
    )
    request = NS(req=object())
    queue._release_request(request)
    assert events == ["retire", "release"]
    release.assert_called_once_with(request.req, queue.tree_cache, checkpoint=False)


@pytest.mark.parametrize("layerwise", [False, True])
def test_metadata_publish_exception_keeps_npu_request_in_failure_cleanup(layerwise):
    module = definitions(
        "decode.py",
        {"DecodePreallocQueue": {"_send_kv_metadata"}},
        KVTransferDestination=NS(HOST="host"),
    )
    queue = module.DecodePreallocQueue()
    queue.req_to_metadata_buffer_idx_allocator = NS(alloc=lambda: 7)
    queue._transfer_page_indices = lambda req, indices, page: np.asarray(indices)
    queue.transfer_queue = NS(enable_staging=False)
    queue._num_published_destinations = 0
    manager = NS(record_failure=Mock(), update_status=Mock())
    request = NS(
        req=NS(
            bootstrap_room=5, time_stats=NS(set_decode_transfer_queue_entry_time=Mock())
        ),
        kv_receiver=NS(
            kv_mgr=manager, send_metadata=Mock(side_effect=ValueError("partial state"))
        ),
        prefix_match=None,
        is_rebootstrap=False,
    )
    kwargs = {"layer_device_kv_indices": np.array([1, 2])} if layerwise else {}
    if layerwise:
        queue._send_kv_metadata(request, [1, 2], 1, **kwargs)
        manager.record_failure.assert_called_once_with(5, "partial state")
        manager.update_status.assert_called_once_with(5, POLL.Failed)
        assert request.metadata_buffer_index == 7
    else:
        with pytest.raises(ValueError, match="partial state"):
            queue._send_kv_metadata(request, [1, 2], 1, **kwargs)


def test_prefill_retry_and_late_abort_wait_for_runtime_release_consensus():
    module = definitions(
        "prefill.py",
        {"SchedulerDisaggregationPrefillMixin": {"_defer_prefill_offload_release"}},
    )
    scheduler = module.SchedulerDisaggregationPrefillMixin()
    runtime = NS(release_request=Mock())
    scheduler.disagg_prefill_bootstrap_queue = NS(
        token_to_kv_pool=NS(prefill_offload_runtime=runtime),
        draft_token_to_kv_pool=None,
    )
    scheduler.disagg_prefill_inflight_queue = []
    request = NS(rid="retry", disagg_kv_sender=NS(abort=Mock()))
    action = "optimistic_release_and_requeue"
    assert scheduler._defer_prefill_offload_release(request, action)
    assert scheduler._defer_prefill_offload_release(request, action)
    assert scheduler.disagg_prefill_inflight_queue == [request]
    runtime.release_request.assert_not_called()
    request.disagg_kv_sender.abort.assert_not_called()
    request._npu_offload_release_ready = action
    assert not scheduler._defer_prefill_offload_release(request, action)
    runtime.release_request.assert_called_once_with("retry")
    request._npu_offload_release_ready = None
    assert scheduler._defer_prefill_offload_release(
        request, "_retire_aborted_prefill_result"
    )
    request.disagg_kv_sender.abort.assert_called_once()


def test_static_layer_registration_keeps_pp_ids_without_preparing_layers():
    module = production_definitions(
        SRT / "disaggregation/utils.py",
        {},
        functions={"_npu_layer_transfer_entries", "get_kv_transfer_buf_infos"},
    )
    storage = NS(
        start_layer=0,
        layer_num=3,
        prefill_layer_slots=1,
        decode_host_backing=None,
        k_buffer=torch.zeros(1, 4, 2, 1, 4),
        v_buffer=torch.zeros(1, 4, 2, 1, 2),
        index_k_buffer=torch.zeros(2, 4, 2, 1, 3),
        indexer_layer_ids=(0, 2),
        prepare_layer_access=Mock(
            side_effect=AssertionError("registration must be static")
        ),
    )
    pool = NS(
        full_kv_pool=storage, full_attention_layer_id_mapping={8: 0, 12: 1, 16: 2}
    )
    entries = module._npu_layer_transfer_entries(pool)
    assert [layer for _, layer in entries] == [8, 12, 16, 8, 12, 16, 8, 16]
    pointers, lengths, strides = module.get_kv_transfer_buf_infos(pool)
    assert pointers[0] == pointers[1] == pointers[2]
    assert len(pointers) == len(lengths) == len(strides) == 8
    storage.prepare_layer_access.assert_not_called()


class PrefillStartupReached(RuntimeError):
    pass


class PrefillService(Enum):
    PREFILL = "prefill"


def prefill_startup(monkeypatch, disaggregation):
    name = "sglang.srt.hardware_backend.npu.hisparse.pd"
    policy = production_definitions(
        SRT / "hardware_backend/npu/hisparse/pd.py",
        {},
        functions={"validate_prefill_bootstrap_policy"},
    )
    module = ModuleType(name)
    module.validate_prefill_bootstrap_policy = policy.validate_prefill_bootstrap_policy
    monkeypatch.setitem(sys.modules, name, module)
    parallel = Mock(side_effect=PrefillStartupReached("parallel setup reached"))
    get_disaggregation = Mock(return_value=disaggregation)
    queue = definitions(
        "prefill.py",
        {"PrefillBootstrapQueue": {"__init__"}},
        get_parallel=parallel,
        get_disagg=get_disaggregation,
    ).PrefillBootstrapQueue

    def start(pool):
        return queue(
            token_to_kv_pool=pool,
            draft_token_to_kv_pool=None,
            req_to_metadata_buffer_idx_allocator=None,
            metadata_buffers=None,
            gpu_id=0,
            bootstrap_port=0,
            max_total_num_tokens=0,
            scheduler=None,
            scheduler_stage_metrics=None,
            transfer_backend=None,
        )

    return start, parallel, get_disaggregation


@pytest.mark.parametrize("role", ["prefill", PrefillService.PREFILL])
@pytest.mark.parametrize("attempts", [0, 1, 3])
def test_layerwise_prefill_complete_rejected_before_parallel_or_transport_startup(
    monkeypatch, role, attempts
):
    disaggregation = NS(
        disaggregation_mode=role,
        disaggregation_decode_allocation_policy="prefill_complete",
        optimistic_prefill_attempts=attempts,
    )
    start, parallel, get_disaggregation = prefill_startup(monkeypatch, disaggregation)
    with pytest.raises(ValueError, match="prefill_complete is unsupported") as error:
        start(NS(prefill_layer_slots=2))
    assert "--disaggregation-decode-allocation-policy=early" in str(error.value)
    assert "--optimistic-prefill-attempts=0" in str(error.value)
    parallel.assert_not_called()
    get_disaggregation.assert_called_once_with()
    assert disaggregation.optimistic_prefill_attempts == attempts


@pytest.mark.parametrize("attempts", [1, 3])
def test_layerwise_optimistic_prefill_rejected_before_startup(monkeypatch, attempts):
    start, parallel, _ = prefill_startup(
        monkeypatch,
        NS(
            disaggregation_mode="prefill",
            disaggregation_decode_allocation_policy="early",
            optimistic_prefill_attempts=attempts,
        ),
    )
    with pytest.raises(ValueError, match="optimistic-prefill-attempts=0"):
        start(NS(prefill_layer_slots=1))
    parallel.assert_not_called()


def test_layerwise_early_prefill_without_optimism_reaches_normal_startup(monkeypatch):
    start, parallel, _ = prefill_startup(
        monkeypatch,
        NS(
            disaggregation_mode="prefill",
            disaggregation_decode_allocation_policy="early",
            optimistic_prefill_attempts=0,
        ),
    )
    with pytest.raises(PrefillStartupReached):
        start(NS(prefill_layer_slots=1))
    parallel.assert_called_once_with()


@pytest.mark.parametrize("policy,attempts", [("early", 2), ("prefill_complete", 1)])
def test_ordinary_prefill_keeps_main_bootstrap_policy(monkeypatch, policy, attempts):
    start, parallel, get_disaggregation = prefill_startup(
        monkeypatch,
        NS(
            disaggregation_mode="prefill",
            disaggregation_decode_allocation_policy=policy,
            optimistic_prefill_attempts=attempts,
        ),
    )
    with pytest.raises(PrefillStartupReached):
        start(NS())
    parallel.assert_called_once_with()
    get_disaggregation.assert_not_called()


def main_prefill_policy(**changes):
    values = dict(
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
        disaggregation_decode_allocation_policy="prefill_complete",
        optimistic_prefill_attempts=0,
        disaggregation_decode_host_receive_threshold=1,
        enable_hisparse=False,
        enable_pd_role_switch=False,
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        tp_size=1,
        dp_size=1,
        enable_dp_attention=False,
        enable_pdmux=False,
        decoupled_spec_role="null",
        encoder_only=False,
        language_only=False,
        enable_hierarchical_cache=False,
    )
    values.update(changes)
    return NS(**values)


def run_main_pd_policy(config):
    def declare_resolution(server_args, hook, **fields):
        assert server_args is config and hook == "handle_pd_disaggregation"
        config.__dict__.update(fields)

    module = production_definitions(
        SRT / "arg_groups/pd_disaggregation_hook.py",
        {},
        functions={"handle_pd_disaggregation", "_validate_prefill_complete"},
        resolving_view=lambda args: args,
        declare_resolution=declare_resolution,
        logger=logging.getLogger(__name__),
        envs=NS(
            SGLANG_RUST_SERVER=NS(get=lambda: False),
            SGLANG_DISAGG_STAGING_BUFFER=NS(get=lambda: False),
        ),
    )
    module.handle_pd_disaggregation(config)


def test_main_prefill_complete_default_is_rejected_for_bounded_runtime(monkeypatch):
    config = main_prefill_policy()
    run_main_pd_policy(config)
    assert config.optimistic_prefill_attempts == 1
    start, parallel, _ = prefill_startup(monkeypatch, config)
    with pytest.raises(ValueError, match="prefill_complete is unsupported"):
        start(NS(prefill_layer_slots=2))
    parallel.assert_not_called()
    assert config.optimistic_prefill_attempts == 1


def test_main_prefill_complete_hisparse_rejected_before_default_resolution():
    config = main_prefill_policy(
        enable_hisparse=True, disaggregation_transfer_backend="ascend"
    )
    with pytest.raises(ValueError, match="prefill_complete.*HiSparse"):
        run_main_pd_policy(config)
    assert config.optimistic_prefill_attempts == 0


if __name__ == "__main__":
    from sglang.test.npu_test_paths import run_pytest

    run_pytest(__file__)
