"""Actual Eagle verify dispatch with CPU data and the NPU HiSparse coordinator."""

import contextlib
import sys
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Eagle framework imports require Linux dependencies"
)


class CpuOnly(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if kwargs.get("device") is not None:
            assert torch.device(kwargs["device"]).type == "cpu"
        return function(*args, **kwargs)


@pytest.fixture
def framework():
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()
    from sglang.srt.hardware_backend.npu.hisparse.runtime import NpuHiSparseCoordinator
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative import eagle_utils, eagle_worker_common, spec_utils
    from sglang.srt.speculative.eagle_info import EagleVerifyInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    with get_context().override_server_args(
        device="cpu",
        speculative_algorithm="EAGLE",
        speculative_num_steps=5,
        speculative_num_draft_tokens=6,
        speculative_eagle_topk=1,
        mamba_radix_cache_strategy="no_buffer",
    ):
        yield SimpleNamespace(
            coordinator=NpuHiSparseCoordinator,
            request=Req,
            batch=ScheduleBatch,
            request_pool=ReqToTokenPool,
            translator=KVIndexTranslator,
            params=SamplingParams,
            algorithm=SpeculativeAlgorithm.EAGLE,
            mode=ForwardMode,
            capture=CaptureHiddenMode,
            forward_batch=ForwardBatch,
            verify_input=EagleVerifyInput,
            utils=eagle_utils,
            common=eagle_worker_common,
            spec_utils=spec_utils,
        )


@pytest.mark.parametrize("accepted_drafts", [0, 2, 5])
def test_eagle_mtp5_keeps_canonical_slots_without_scratch_or_layer_backup(
    framework, monkeypatch, accepted_drafts
):
    from sglang.kernels.ops.speculative import cache_locs

    width = 6
    forbidden = Mock(
        side_effect=AssertionError("unexpected scratch, remap or KV backup")
    )
    host = torch.zeros(768)
    kv_pool = SimpleNamespace(
        decode_host_backing=SimpleNamespace(backup_from_device_all_layer=forbidden),
        move_kv_cache=forbidden,
    )
    allocator = SimpleNamespace(
        get_kvcache=lambda: kv_pool,
        alloc=forbidden,
        alloc_extend=forbidden,
        alloc_decode=forbidden,
        hisparse_attn_allocator=SimpleNamespace(alloc=forbidden),
    )
    pool = framework.request_pool(2, 512, "cpu", False)
    model = SimpleNamespace(vocab_size=32, is_encoder_decoder=False)
    runner = SimpleNamespace(
        device="cpu",
        req_to_token_pool=pool,
        token_to_kv_pool=kv_pool,
        token_to_kv_pool_allocator=allocator,
        prefill_offload_runtime=None,
        spec_algorithm=framework.algorithm,
        model_config=model,
        decode_cuda_graph_runner=None,
        watermark_state=None,
        attn_backend=SimpleNamespace(verify_mask=None),
    )
    coordinator = framework.coordinator(runner)
    requests = [
        framework.request(str(row), "", array("q", [1, 2]), framework.params())
        for row in range(2)
    ]
    slots = pool.alloc(requests)
    batch = framework.batch.init_new(
        requests,
        pool,
        allocator,
        SimpleNamespace(),
        model,
        False,
        framework.algorithm,
        hisparse_coordinator=coordinator,
    )
    batch.forward_mode = framework.mode.DECODE
    batch.req_pool_indices = torch.tensor(slots)
    batch.seq_lens = torch.tensor([127, 255], dtype=torch.int32)
    batch.seq_lens_cpu = batch.seq_lens.clone()
    batch.input_ids = torch.tensor([1, 2])
    batch.out_cache_loc = None
    canonical = torch.tensor(
        [127, 256, 257, 258, 259, 260, 511, 640, 641, 642, 643, 644]
    )
    for row, request in enumerate(requests):
        length = batch.seq_lens[row].item()
        request.kv.kv_committed_len = length
        request.kv.kv_allocated_len = length + width
        pool.write(
            (slots[row], slice(length, length + width)), canonical.view(2, width)[row]
        )
    translator = framework.translator(
        req_to_token=pool.req_to_token,
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=kv_pool,
        page_size=128,
        device="cpu",
    )
    plan = translator.plan(
        req_pool_indices=batch.req_pool_indices,
        seq_lens=batch.seq_lens,
        seq_lens_cpu=batch.seq_lens_cpu,
        write_virtual=canonical,
        read_extent=width,
    )
    verify = framework.verify_input(
        draft_token=torch.arange(12),
        custom_mask=torch.ones(12, dtype=torch.bool),
        positions=torch.cat((torch.arange(127, 133), torch.arange(255, 261))),
        retrieve_index=torch.arange(12).view(2, width),
        retrieve_next_token=torch.full((2, width), -1),
        retrieve_next_sibling=torch.full((2, width), -1),
        retrieve_cum_len=None,
        spec_steps=5,
        topk=1,
        draft_token_num=width,
        capture_hidden_mode=framework.capture.FULL,
        seq_lens_sum=382,
        seq_lens_cpu=batch.seq_lens_cpu,
        prepared_out_cache_loc=plan.write_ids(translator),
        kv_loc_plan=plan,
    )
    batch.spec_info = verify
    snapshot = pool.req_to_token.clone()
    prepare = Mock(wraps=coordinator.prepare_spec_verify)
    commit = Mock(wraps=coordinator.commit_spec_accept_tokens)
    monkeypatch.setattr(coordinator, "prepare_spec_verify", prepare)
    monkeypatch.setattr(coordinator, "commit_spec_accept_tokens", commit)
    prepare_forward = Mock(wraps=framework.utils.eagle_prepare_for_verify)
    monkeypatch.setattr(framework.common, "eagle_prepare_for_verify", prepare_forward)
    monkeypatch.setattr(framework.utils, "_PREPARE_FOR_VERIFY_DEPS", None)
    monkeypatch.setattr(cache_locs, "assign_extend_cache_locs_uniform_func", forbidden)
    monkeypatch.setattr(framework.common, "prepare_verify_commit_outputs", forbidden)
    monkeypatch.setattr(framework.common, "_finalize_accept_tree_path", forbidden)
    monkeypatch.setattr(framework.spec_utils, "prepare_mamba_track_for_verify", Mock())
    monkeypatch.setattr(framework.common, "commit_mamba_states_after_verify", Mock())
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device: SimpleNamespace(current_stream=lambda: None),
    )

    def init_forward(schedule, model_runner, **kwargs):
        assert schedule is batch and model_runner is runner
        assert kwargs["kv_loc_plan"] is plan
        assert batch.out_cache_loc is canonical
        forward = SimpleNamespace(
            input_ids=batch.input_ids, forward_mode=batch.forward_mode
        )
        plan.bind(forward, translator)
        return forward

    init_new = Mock(side_effect=init_forward)
    monkeypatch.setattr(framework.forward_batch, "init_new", init_new)
    values = torch.arange(1, 13, dtype=torch.float32)

    def forward(**kwargs):
        assert prepare.call_count == 1 and commit.call_count == 0
        assert kwargs["is_verify"] and kwargs["batch"] is None
        current = kwargs["forward_batch"]
        assert current.kv_loc_plan is plan and current.out_cache_loc is canonical
        host[current.out_cache_loc] = values
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=torch.zeros(12, 32)),
            routed_experts_output=None,
            indexer_topk_output=None,
        )

    target = SimpleNamespace(
        model_runner=runner, forward_batch_generation=Mock(side_effect=forward)
    )
    predicted = torch.arange(12)
    lengths = torch.full((2,), accepted_drafts + 1, dtype=torch.int32)
    indices = torch.full((2, width), -1, dtype=torch.int64)
    for row in range(2):
        indices[row, : accepted_drafts + 1] = row * width + torch.arange(
            accepted_drafts + 1
        )
    sampler = Mock(return_value=(predicted, lengths, indices))
    monkeypatch.setattr(framework.common, "eagle_sample", sampler)

    def bonus(accepted, counts, output, stride, batch_size):
        output.copy_(accepted.gather(1, (counts - 1).long()[:, None]).flatten())

    monkeypatch.setattr(framework.common, "fill_bonus_tokens_func", bonus)
    with CpuOnly():
        result = framework.common.run_eagle_verify(
            batch,
            target_worker=target,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            plan_stream=None,
            plan_stream_ctx=contextlib.nullcontext(),
            topk=1,
            num_draft_tokens=width,
            device="cpu",
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
        )
    prepare_forward.assert_called_once_with(verify, pool, batch, target)
    assert prepare.call_count == 2
    assert all(call.args == (batch,) for call in prepare.call_args_list)
    commit.assert_called_once_with(batch=batch, accept_indices=indices)
    target.forward_batch_generation.assert_called_once()
    sampler.assert_called_once()
    init_new.assert_called_once()
    assert batch.forward_mode == framework.mode.TARGET_VERIFY
    assert batch.out_cache_loc is verify.prepared_out_cache_loc is canonical
    assert verify.kv_loc_plan is plan and plan.write_ids(translator) is canonical
    assert result.extra_keep_alive_refs[0].kv_loc_plan is plan
    assert result.extra_keep_alive_refs[0].out_cache_loc is canonical
    assert not result.can_run_cuda_graph and not hasattr(coordinator, "spec_swap")
    torch.testing.assert_close(result.accept_lens, lengths)
    torch.testing.assert_close(result.new_seq_lens, batch.seq_lens + lengths)
    torch.testing.assert_close(result.accept_index, indices)
    torch.testing.assert_close(
        result.next_draft_input.bonus_tokens,
        predicted[torch.arange(2) * width + accepted_drafts].to(torch.int32),
    )
    torch.testing.assert_close(host[canonical], values)
    torch.testing.assert_close(pool.req_to_token, snapshot)
    for row in range(2):
        accepted = indices[row, : accepted_drafts + 1]
        torch.testing.assert_close(host[canonical[accepted]], values[accepted])
    forbidden.assert_not_called()


if __name__ == "__main__":
    run_pytest(__file__)
