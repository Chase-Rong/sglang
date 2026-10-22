"""Drain service users before closing the NPU Host allocation owners."""

import time


def drain_receive_leases(manager, timeout_s):
    from sglang.srt.disaggregation.base.conn import KVPoll
    from sglang.srt.hardware_backend.npu.hisparse.layer_protocol import encode
    from sglang.srt.utils.network import NetworkAddress

    deadline = time.monotonic() + timeout_s
    for room, lease in tuple(manager._layer_leases.items()):
        success = lease.ready() and manager.check_status(room) == KVPoll.Success
        if not success and not lease.drained:
            lease.failed = True
            if not lease.abort_generation:
                if manager.enable_deferred_decode_kv_release:
                    lease.abort_generation = manager.register_deferred_abort_room(room)
                else:
                    lease.abort_generation = 1
            manager.update_status(room, KVPoll.Failed)
            endpoints = lease.writer_endpoints()
            payload = encode(
                dict(
                    room=room,
                    session=lease.session_id,
                    lease=lease.lease_id,
                    abort_generation=lease.abort_generation,
                    reply_ip=manager.local_ip,
                    reply_port=manager.rank_port,
                )
            )
            for endpoint, port in endpoints:
                address = NetworkAddress(endpoint, port)
                manager._send_multipart_locked(
                    address.to_tcp(),
                    [b"FSA_LAYER_ABORT", payload],
                    is_ipv6=address.is_ipv6,
                )
            if not endpoints:
                raise RuntimeError(
                    "Cannot close a PD lease without a negotiated writer drain"
                )
            while not lease.drained:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "PD writers have not drained during HiSparse shutdown"
                    )
                time.sleep(0.001)
        staging = manager._layer_staging
        if staging is not None:
            staging.forget(lease)
        lease.close()
        del manager._layer_leases[room]
    if manager._layer_staging is not None:
        manager._layer_staging.drain()


def _agree_on_shutdown(coordinators, error):
    groups = {}
    for coordinator in coordinators:
        backing = coordinator._backing
        runtime = coordinator.prefill_offload_runtime
        pool = (
            backing.pool
            if backing is not None
            else runtime.adapter.host_pool
            if runtime is not None and hasattr(runtime, "adapter")
            else None
        )
        shared = getattr(pool, "shared_group", None)
        if shared is not None:
            group = shared.group
            groups[id(getattr(group, "cpu_group", group))] = group
    for group in groups.values():
        errors = group.all_gather_object(str(error) if error is not None else None)
        if any(item is not None for item in errors):
            raise RuntimeError(
                f"NPU HiSparse shutdown could not drain all TP ranks: {errors}"
            ) from error
    if error is not None:
        raise error


def release_npu_hisparse_resources(scheduler):
    if getattr(scheduler, "_npu_hisparse_resources_released", False):
        return
    coordinators = [scheduler.hisparse_coordinator]
    draft = scheduler.draft_worker
    if draft is not None:
        for runner in draft._draft_model_runners():
            coordinator = runner.hisparse_coordinator
            if coordinator is not None and coordinator not in coordinators:
                coordinators.append(coordinator)
    managers = []
    for name in ("disagg_prefill_bootstrap_queue", "disagg_decode_prealloc_queue"):
        queue = getattr(scheduler, name, None)
        manager = getattr(queue, "kv_manager", None)
        if manager is not None and manager not in managers:
            managers.append(manager)
    error = None
    try:
        for coordinator in coordinators:
            coordinator.prepare_shutdown()
        for manager in managers:
            drain_receive_leases(manager, 30.0)
    except Exception as caught:
        error = caught
    _agree_on_shutdown(coordinators, error)
    error = None
    try:
        scheduler.tree_cache.release_host_resources()
    except Exception as caught:
        error = caught
    _agree_on_shutdown(coordinators, error)
    for coordinator in coordinators:
        coordinator.destroy()
    scheduler._npu_hisparse_resources_released = True
