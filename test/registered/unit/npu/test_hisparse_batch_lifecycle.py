"""CPU integration using complete framework classes, without AST extraction."""

import sys
from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.npu_test_paths import run_pytest

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="SGLang scheduler requires Linux runtime dependencies",
)


@pytest.fixture
def framework():
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()
    from sglang.srt.hardware_backend.npu.hisparse.runtime import NpuHiSparseCoordinator
    from sglang.srt.managers import schedule_batch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.runtime_context import get_context
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    request_pool = ReqToTokenPool(2, 16, "cpu", False)
    runner = SimpleNamespace(
        device="cpu",
        req_to_token_pool=request_pool,
        token_to_kv_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=object(),
        prefill_offload_runtime=None,
    )
    coordinator = NpuHiSparseCoordinator(runner)
    with get_context().override_server_args(
        device="cpu", speculative_algorithm=None, mamba_radix_cache_strategy="no_buffer"
    ):
        yield SimpleNamespace(
            schedule=schedule_batch,
            pool=request_pool,
            coordinator=coordinator,
            params=SamplingParams,
            algorithm=SpeculativeAlgorithm.NONE,
        )


def make_batch(framework, name, coordinator):
    request = framework.schedule.Req(
        name, "", array("q", [1, 2]), framework.params(max_new_tokens=4)
    )
    framework.pool.alloc([request])
    request.kv.kv_committed_len = request.kv.kv_allocated_len = 2
    batch = framework.schedule.ScheduleBatch.init_new(
        [request],
        framework.pool,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(vocab_size=32, is_encoder_decoder=False),
        False,
        framework.algorithm,
        hisparse_coordinator=coordinator,
    )
    batch.req_pool_indices = torch.tensor([request.kv.req_pool_idx])
    batch.req_pool_indices_cpu = batch.req_pool_indices.clone()
    batch.seq_lens = batch.orig_seq_lens = batch.seq_lens_cpu = torch.tensor([2])
    batch.sampling_info = framework.schedule.SamplingBatchInfo.from_schedule_batch(
        batch, 32
    )
    return batch


def test_real_batch_retract_readmit_finish_updates_generation(framework, monkeypatch):
    coordinator = framework.coordinator
    batch = make_batch(framework, "a", coordinator)
    request = batch.reqs[0]
    slot = request.kv.req_pool_idx
    initial = framework.pool.req_generation[slot].item()
    monkeypatch.setattr(
        framework.schedule,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_mode="null"),
    )
    monkeypatch.setattr(
        framework.schedule,
        "release_kv_cache",
        lambda req, *args, **kwargs: framework.pool.free(req),
    )
    monkeypatch.setattr(framework.schedule, "evict_from_tree_cache", lambda *args: None)
    assert batch.release_req(0, 0)
    assert request.retraction_count == 1
    assert request.kv.req_pool_idx is None
    assert framework.pool.req_generation[slot] == initial + 1
    assert framework.pool.alloc([request]) == [slot]
    coordinator.request_finished(request)
    coordinator.request_finished(request)
    assert framework.pool.req_generation[slot] == initial + 3


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_pd_completion_recycles_capacity_unless_prefix_cache_is_enabled(
    framework, cache_enabled
):
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.common import release_kv_cache
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.runtime_context import get_context

    allocator = TokenToKVPoolAllocator(
        size=16, dtype=torch.bfloat16, device="cpu", kvcache=None, need_sort=False
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=framework.pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            tree_components=(ComponentType.FULL,),
        )
    )
    with get_context().override_server_args(
        disaggregation_mode="decode",
        disaggregation_decode_enable_radix_cache=cache_enabled,
    ):
        for request_index in range(1 if cache_enabled else 3):
            request = framework.schedule.Req(
                f"pd-{request_index}",
                "",
                array("q", range(10)),
                framework.params(max_new_tokens=1),
            )
            assert framework.pool.alloc([request]) is not None
            rows = allocator.alloc(10)
            assert rows is not None
            framework.pool.write((request.kv.req_pool_idx, slice(0, 10)), rows)
            request.kv.kv_committed_len = request.kv.kv_allocated_len = 10
            request.kv.cache_protected_len = 0
            framework.coordinator.request_finished(request)
            release_kv_cache(request, cache, checkpoint=True)
            assert request.kv.req_pool_idx is None
            assert allocator.available_size() == (6 if cache_enabled else 16)
            assert cache.evictable_size() == (10 if cache_enabled else 0)


@pytest.mark.parametrize("inherited", [True, False])
def test_real_batch_merge_and_snapshot_preserve_coordinator(framework, inherited):
    coordinator = framework.coordinator
    first = make_batch(framework, "a", None if inherited else coordinator)
    second = make_batch(framework, "b", coordinator)
    first.merge_batch(second)
    assert first.hisparse_coordinator is coordinator
    assert first.copy().hisparse_coordinator is coordinator
    assert len(first.reqs) == 2


def test_real_batch_rejects_foreign_coordinator_before_merging(framework):
    first = make_batch(framework, "a", framework.coordinator)
    second = make_batch(framework, "b", object())
    with pytest.raises(ValueError, match="different HiSparse"):
        first.merge_batch(second)
    assert len(first.reqs) == len(second.reqs) == 1


def test_real_eagle_wrapper_shutdown_includes_target_and_draft(framework):
    from sglang.srt.hardware_backend.npu.hisparse.shutdown import (
        release_npu_hisparse_resources,
    )
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

    order = []
    draft = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
    draft_coordinator = SimpleNamespace(
        prepare_shutdown=lambda: order.append("draft drain"),
        destroy=lambda: order.append("draft close"),
        _backing=None,
        prefill_offload_runtime=None,
    )
    draft._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            spec_algorithm=SimpleNamespace(
                is_ngram=lambda: False,
                is_frozen_kv_mtp=lambda: False,
                is_dflash_family=lambda: False,
            )
        )
    )
    draft._draft_worker = SimpleNamespace(
        draft_runners=[SimpleNamespace(hisparse_coordinator=draft_coordinator)]
    )
    target = framework.coordinator
    service = SimpleNamespace(
        hisparse_coordinator=target,
        draft_worker=draft,
        tree_cache=SimpleNamespace(
            release_host_resources=lambda: order.append("cache close")
        ),
    )
    release_npu_hisparse_resources(service)
    assert order == ["draft drain", "cache close", "draft close"]
    assert target._destroyed


def test_real_prefill_and_spec_reservation_use_npu_physical_allocator(
    framework, monkeypatch
):
    from sglang.srt.hardware_backend.npu.allocator_npu import (
        NPUPagedTokenToKVPoolAllocator,
    )
    from sglang.srt.mem_cache import allocation
    from sglang.srt.mem_cache.allocator import alloc_extend_naive
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    class CpuExtendKernel:
        def __getitem__(self, grid):
            def launch(
                prefixes, lengths, last, free, output, batch_size, page_size, max_tokens
            ):
                alloc_extend_naive(
                    prefixes, lengths, last, free, output, page_size, "cpu"
                )

            return launch

    monkeypatch.setitem(
        sys.modules,
        "sgl_kernel_npu.mem_cache.allocator",
        SimpleNamespace(alloc_extend_kernel=CpuExtendKernel()),
    )

    allocator = NPUPagedTokenToKVPoolAllocator(
        32, 4, torch.bfloat16, "cpu", None, False
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=framework.pool,
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            tree_components=(ComponentType.FULL,),
        )
    )
    request = framework.schedule.Req(
        "allocation", "", array("q", [1, 2, 3]), framework.params(max_new_tokens=4)
    )
    request.extend_end = 3
    batch = SimpleNamespace(
        reqs=[request],
        req_to_token_pool=framework.pool,
        tree_cache=cache,
        hisparse_coordinator=framework.coordinator,
        device="cpu",
        prefix_lens=[0],
        extend_lens=[3],
        extend_num_tokens=3,
        seq_lens=torch.tensor([3]),
        seq_lens_cpu=torch.tensor([3]),
        maybe_evict_swa=lambda: None,
        is_dllm=lambda: False,
    )

    def write_slots(locations, device_requests, cpu_requests, *args):
        framework.pool.req_to_token[int(cpu_requests[0]), : locations.numel()] = (
            locations.int()
        )

    def assign_slots(requests, table, starts, ends, locations, count):
        table[int(requests[0]), int(starts[0]) : int(ends[0])] = locations.int()

    monkeypatch.setattr(allocation, "write_cache_indices", write_slots)
    locations, requests, _ = allocation.alloc_for_extend(batch)
    assert locations.numel() == 3
    assert allocator.available_size() == 28
    assert not hasattr(framework.coordinator, "allocate_spec_scratch")
    monkeypatch.setattr(allocation, "get_last_loc", lambda *args: locations[-1:])
    monkeypatch.setattr(allocation, "assign_req_to_token_pool_func", assign_slots)
    allocation.alloc_for_spec_decode(
        cache,
        framework.pool,
        reqs=[request],
        req_pool_indices=requests,
        cur_kv_lens=torch.tensor([3]),
        cur_kv_lens_cpu=torch.tensor([3]),
        nxt_kv_lens=torch.tensor([9]),
        nxt_kv_lens_cpu=torch.tensor([9]),
        num_needed_tokens=6,
        batch=batch,
    )
    row = framework.pool.req_to_token[int(requests[0]), :9]
    assert row.unique().numel() == 9 and torch.all(row > 0)
    assert allocator.available_size() == 20
    assert request.kv.kv_allocated_len == 9


if __name__ == "__main__":
    run_pytest(__file__)
