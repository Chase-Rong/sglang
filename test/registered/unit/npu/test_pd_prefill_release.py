"""Exercise scheduler transfer completion without loading NPU workers."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("has_draft", [False, True])
def test_prefill_completion_uses_bootstrap_pools_and_releases_after_drain(
    has_draft, monkeypatch
):
    import sys
    import types
    from enum import IntEnum

    class Poll(IntEnum):
        WaitingForInput = 1
        Transferring = 2
        Success = 3
        Failed = 4

    events = []
    readiness = [False]
    runtimes = [
        NS(
            can_release_request=lambda rid: readiness[0],
            release_request=lambda rid: events.append(("target", rid)),
        )
    ]
    if has_draft:
        runtimes.append(
            NS(
                can_release_request=lambda rid: True,
                release_request=lambda rid: events.append(("draft", rid)),
            )
        )
    source = Path(__file__).parents[4] / "python/sglang/srt"
    runtime_ast = ast.parse(
        (source / "hardware_backend/npu/prefill_offload_runtime.py").read_text()
    )
    check_node = next(
        node
        for node in runtime_ast.body
        if isinstance(node, ast.ClassDef) and node.name == "PrefillReleaseCheck"
    )
    connection = types.ModuleType("sglang.srt.disaggregation.base.conn")
    connection.KVPoll = Poll
    monkeypatch.setitem(sys.modules, connection.__name__, connection)
    runtime_module = types.ModuleType(
        "sglang.srt.hardware_backend.npu.prefill_offload_runtime"
    )
    exec(
        compile(
            ast.Module(body=[check_node], type_ignores=[]), "<release-check>", "exec"
        ),
        runtime_module.__dict__,
    )
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    tree = ast.parse((source / "disaggregation/prefill.py").read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "process_disagg_prefill_inflight_queue"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace = dict(
        KVPoll=Poll,
        FINISH_ABORT=type("Abort", (), {}),
        FINISH_LENGTH=lambda length: length,
        FAKE_BOOTSTRAP_HOST="fake",
        poll_and_all_reduce_attn_cp_tp_group=lambda checks, *groups: [
            check.poll() for check in checks
        ],
        release_kv_cache=lambda req, cache: events.append(("kv", req.rid)),
        maybe_release_metadata_buffer=lambda req, alloc: events.append(
            ("metadata", req.rid)
        ),
    )
    exec(compile(ast.fix_missing_locations(module), "<prefill>", "exec"), namespace)
    operation = NS(state="transferred")
    sender = NS(
        poll=lambda: Poll.Success,
        prefill_layer_operations=lambda: [operation],
        clear=lambda: events.append(("clear", "req")),
    )
    req = NS(
        rid="req",
        disagg_kv_sender=sender,
        pending_bootstrap=False,
        finished_reason=None,
        bootstrap_host="fake",
        return_logprob=False,
        time_stats=NS(
            set_prefill_kv_transfer_finish_time=lambda: None,
            set_completion_time=lambda: None,
        ),
    )
    scheduler = NS(
        disagg_prefill_inflight_queue=[req],
        attn_cp_cpu_group=None,
        attn_tp_cpu_group=None,
        disagg_prefill_bootstrap_queue=NS(
            token_to_kv_pool=NS(prefill_offload_runtime=runtimes[0]),
            draft_token_to_kv_pool=NS(prefill_offload_runtime=runtimes[1])
            if has_draft
            else None,
        ),
        tree_cache=None,
        req_to_metadata_buffer_idx_allocator=None,
        output_streamer=NS(stream_output=lambda *args: None),
    )
    complete = namespace[method.name]
    assert complete(scheduler) == []
    assert events == []
    readiness[0] = True
    assert complete(scheduler) == []
    operation.state = "drained"
    assert complete(scheduler) == [req]
    assert scheduler.disagg_prefill_inflight_queue == []
    expected = [("target", "req")]
    if has_draft:
        expected.append(("draft", "req"))
    assert events == expected + [("kv", "req"), ("clear", "req"), ("metadata", "req")]
