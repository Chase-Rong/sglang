"""Layer-plan fusion preserves the complete ABI and replay-dependent state."""

import importlib.util
import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_FSA_TEST_ATTENTION") != "1",
    reason="requires an idle qualified NPU",
)


@pytest.fixture
def device():
    import torch_npu  # noqa: F401

    device = torch.device("npu", int(os.environ["SGLANG_FSA_TEST_DEVICE_ID"]))
    torch.npu.set_device(device)
    return device


def fused():
    name = "sglang.srt.hardware_backend.npu.fsa_layer_plan"
    assert importlib.util.find_spec(name) is not None, "fused layer plan is missing"
    from sglang.srt.hardware_backend.npu.fsa_layer_plan import build_layer_plan

    return build_layer_plan


def state(rows, capacity, device="cpu"):
    applied = torch.full((rows, capacity + 1), -1, dtype=torch.int64, device=device)
    return SimpleNamespace(
        selection_kv_cache=torch.empty((rows, capacity, 0), device=device),
        max_tokens=rows,
        applied_keys=applied[:, :-1],
        applied_keys_with_padding=applied,
    )


def test_plan_literal_encoding_and_clamped_invalid_slots(device):
    build = fused()
    keys = torch.tensor([[7, 7, -1, 8, 9, 10, 1 << 40]], device=device)
    slots = torch.tensor([[0, 1, 2, -1, 8, 7, 6]], device=device)
    applied = torch.tensor([[7, 7, -1, -1, -1, -1, 0, 10]], device=device)
    stable = torch.tensor([[True, False, True, True, True, True, True]], device=device)
    membership = torch.full((1, 16400), 123, dtype=torch.int16, device=device)
    safe, miss = build(keys, slots, applied, membership, capacity=8, cacheable_mask=stable)
    assert safe.cpu().tolist() == [[0, 1, 2, 0, 7, 7, 6]]
    assert miss.cpu().tolist() == [[False, True, False, False, False, False, True]]
    assert membership[0, 16377:16392].cpu().tolist() == [
        1, -2, 0, 0, 0, 8, -7, 0x5A4D, 0x5A45, 7, 16377, 0, 0x5A44, 8, 0
    ]
    assert torch.all(membership[:, :16377].cpu() == 123)
    assert torch.all(membership[:, 16392:].cpu() == 123)


@pytest.mark.parametrize("rows,width,capacity", [(1, 1, 1), (6, 2051, 4096), (19, 17, 32767), (120, 2051, 4096)])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_plan_eager_and_graph_match_cpu(device, rows, width, capacity, masked, inference):
    build = fused()
    from sglang.srt.hardware_backend.npu.external_sparse_planner import materialize_layer_plan

    torch.manual_seed(372)
    cache = state(rows, capacity)
    gpu_cache = state(rows, capacity, device)
    # Step-two views exercise both row and column strides, including membership.
    keys = torch.empty((rows, width * 2), dtype=torch.int64)[:, ::2]
    slots = torch.empty((rows, width * 2), dtype=torch.int64)[:, ::2]
    stable = torch.empty((rows, width * 2), dtype=torch.bool)[:, ::2] if masked else None
    gpu_keys = torch.empty((rows, width * 2), dtype=torch.int64, device=device)[:, ::2]
    gpu_slots = torch.empty((rows, width * 2), dtype=torch.int64, device=device)[:, ::2]
    gpu_stable = torch.empty((rows, width * 2), dtype=torch.bool, device=device)[:, ::2] if masked else None
    storage = torch.full((rows, 32800), 123, dtype=torch.int16, device=device)
    membership = storage[:, ::2]
    expected_membership = torch.full((rows, 16400), 123, dtype=torch.int16)

    def update(step):
        keys.copy_(torch.randint(-2, capacity, (rows, width)))
        keys[:, -1] = (1 << 40) + step
        slots.copy_(torch.randint(-2, capacity + 2, (rows, width)))
        cache.applied_keys.copy_(torch.randint(-2, capacity, (rows, capacity)))
        cache.applied_keys[:, -1] = (1 << 40) + step
        slots[:, -1] = capacity - 1
        gpu_keys.copy_(keys)
        gpu_slots.copy_(slots)
        gpu_cache.applied_keys.copy_(cache.applied_keys)
        if masked:
            stable.copy_(torch.rand(rows, width) > 0.4)
            gpu_stable.copy_(stable)

    graph = torch.npu.NPUGraph()
    with torch.inference_mode(inference):
        update(0)
        kwargs = dict(capacity=capacity, cacheable_mask=gpu_stable)
        eager = build(gpu_keys, gpu_slots, gpu_cache.applied_keys, membership, **kwargs)
        expected = materialize_layer_plan(cache, keys, slots, expected_membership, cacheable_mask=stable)
        for actual, want in zip(eager, (expected.slots, expected.miss_mask)):
            torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
        torch.testing.assert_close(membership.cpu(), expected_membership, rtol=0, atol=0)
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            output = materialize_layer_plan(gpu_cache, gpu_keys, gpu_slots, membership, cacheable_mask=gpu_stable)
        pointers = (output.slots.data_ptr(), output.miss_mask.data_ptr())
        try:
            for step in range(4):
                update(step)
                expected = materialize_layer_plan(cache, keys, slots, expected_membership, cacheable_mask=stable)
                graph.replay()
                for actual, want in ((output.slots, expected.slots), (output.miss_mask, expected.miss_mask), (membership, expected_membership), (gpu_cache.applied_keys, cache.applied_keys)):
                    torch.testing.assert_close(actual.cpu(), want, rtol=0, atol=0)
                assert pointers == (output.slots.data_ptr(), output.miss_mask.data_ptr())
                assert torch.all(storage[:, 1::2].cpu() == 123)
        finally:
            torch.npu.synchronize()
            graph.reset()


def test_plan_per_layer_commit_and_mutable_tail(device):
    fused()
    from sglang.srt.hardware_backend.npu.external_sparse_planner import commit_layer_plan, materialize_layer_plan

    cache, other = state(2, 8, device), state(2, 8, device)
    keys = torch.tensor([[7, 7, 8, -1], [17, 17, 18, -1]], device=device)
    slots = torch.arange(4, device=device).expand(2, -1)
    stable = torch.tensor([[True, True, False, False]] * 2, device=device)
    membership = torch.empty((2, 16400), dtype=torch.int16, device=device)
    plan = materialize_layer_plan(cache, keys, slots, membership, cacheable_mask=stable)
    assert torch.all(cache.applied_keys.cpu() == -1)
    commit_layer_plan(cache, plan)
    expected = [[False, False, True, False]] * 2
    assert materialize_layer_plan(cache, keys, slots, membership, cacheable_mask=stable).miss_mask.cpu().tolist() == expected
    assert materialize_layer_plan(other, keys, slots, membership, cacheable_mask=stable).miss_mask.cpu().tolist() == [[True, True, True, False]] * 2
    assert cache.applied_keys[:, :4].cpu().tolist() == [[7, 7, -1, -1], [17, 17, -1, -1]]


def test_plan_transposed_keys_and_slots(device):
    build = fused()
    from sglang.srt.hardware_backend.npu.external_sparse_planner import materialize_layer_plan

    rows, width, capacity = 3, 7, 8
    keys = torch.arange(rows * width, dtype=torch.int64).reshape(width, rows).t()
    slots = (keys % capacity).clone()
    cache = state(rows, capacity)
    cache.applied_keys.copy_(torch.arange(capacity).expand(rows, -1))
    stable = torch.ones((rows * 2, width * 2), dtype=torch.bool)[::2, ::2]
    stable[:, 1::2] = False
    expected_membership = torch.full((rows, 16400), 123, dtype=torch.int16)
    expected = materialize_layer_plan(cache, keys, slots, expected_membership, cacheable_mask=stable)
    gpu_keys, gpu_slots = keys.to(device), slots.to(device)
    assert not gpu_keys.is_contiguous() and not gpu_slots.is_contiguous()
    gpu_stable = torch.empty((rows * 2, width * 2), dtype=torch.bool, device=device)[::2, ::2]
    gpu_stable.copy_(stable)
    gpu_applied = torch.empty((rows * 2, capacity * 2), dtype=torch.int64, device=device)[::2, ::2]
    gpu_applied.copy_(cache.applied_keys)
    membership = torch.full((rows, 16400), 123, dtype=torch.int16, device=device)
    actual = build(gpu_keys, gpu_slots, gpu_applied, membership, capacity=capacity, cacheable_mask=gpu_stable)
    for got, want in zip(actual, (expected.slots, expected.miss_mask)):
        torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
    torch.testing.assert_close(membership.cpu(), expected_membership, rtol=0, atol=0)

    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    try:
        with torch.npu.graph(graph):
            replayed = build(gpu_keys, gpu_slots, gpu_applied, membership, capacity=capacity, cacheable_mask=gpu_stable)
        pointers = tuple(t.data_ptr() for t in replayed)
        for step in range(3):
            keys.add_(step + 1)
            slots.copy_((slots + 1) % capacity)
            stable.logical_not_()
            cache.applied_keys.add_(1)
            gpu_keys.copy_(keys)
            gpu_slots.copy_(slots)
            gpu_stable.copy_(stable)
            gpu_applied.copy_(cache.applied_keys)
            expected = materialize_layer_plan(cache, keys, slots, expected_membership, cacheable_mask=stable)
            graph.replay()
            for got, want in zip(replayed, (expected.slots, expected.miss_mask)):
                torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
            torch.testing.assert_close(membership.cpu(), expected_membership, rtol=0, atol=0)
            assert pointers == tuple(t.data_ptr() for t in replayed)
    finally:
        torch.npu.synchronize()
        graph.reset()


def test_plan_graph_commit_order_and_layer_isolation(device):
    fused()
    from sglang.srt.hardware_backend.npu.external_sparse_planner import commit_layer_plan, materialize_layer_plan

    cache, other = state(2, 8), state(2, 8)
    gpu_cache, gpu_other = state(2, 8, device), state(2, 8, device)
    keys = torch.tensor([[7, 8, 9, -1], [17, 18, 19, -1]])
    slots = torch.tensor([[0, 1, 2, 3]] * 2)
    stable = torch.tensor([[True, True, False, False]] * 2)
    gpu_keys, gpu_slots, gpu_stable = (t.to(device) for t in (keys, slots, stable))
    memberships = [torch.full((2, 16400), 123, dtype=torch.int16) for _ in range(3)]
    gpu_memberships = [t.to(device) for t in memberships]

    def sequence(layer, other_layer, current_keys, current_slots, current_stable, maps):
        before = materialize_layer_plan(layer, current_keys, current_slots, maps[0], cacheable_mask=current_stable)
        commit_layer_plan(layer, before)
        after = materialize_layer_plan(layer, current_keys, current_slots, maps[1], cacheable_mask=current_stable)
        isolated = materialize_layer_plan(other_layer, current_keys, current_slots, maps[2], cacheable_mask=current_stable)
        return before, after, isolated

    sequence(gpu_cache, gpu_other, gpu_keys, gpu_slots, gpu_stable, gpu_memberships)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            replayed = sequence(gpu_cache, gpu_other, gpu_keys, gpu_slots, gpu_stable, gpu_memberships)
        # Capture can execute the commit; the first replay must start cold.
        gpu_cache.applied_keys_with_padding.fill_(-1)
        gpu_other.applied_keys_with_padding.fill_(-1)
        for step, before_miss in enumerate((
            [True, True, True, False],
            [False, False, True, False],
            [False, True, True, False],
            [False, True, True, False],
        )):
            if step == 2:
                keys[:, 1] = torch.tensor([80, 180])
                slots[:, 1] = 4
                stable.copy_(torch.tensor([[True, False, True, False]] * 2))
            elif step == 3:
                stable.copy_(torch.tensor([[True, True, False, False]] * 2))
            for target, source in ((gpu_keys, keys), (gpu_slots, slots), (gpu_stable, stable)):
                target.copy_(source)
            expected = sequence(cache, other, keys, slots, stable, memberships)
            graph.replay()
            assert replayed[0].miss_mask.cpu().tolist() == [before_miss] * 2
            after_miss = [False, True, False, False] if step == 2 else [False, False, True, False]
            assert replayed[1].miss_mask.cpu().tolist() == [after_miss] * 2
            assert replayed[2].miss_mask.cpu().tolist() == [[True, True, True, False]] * 2
            for got, want in zip(replayed, expected):
                torch.testing.assert_close(got.slots.cpu(), want.slots, rtol=0, atol=0)
                torch.testing.assert_close(got.miss_mask.cpu(), want.miss_mask, rtol=0, atol=0)
            for got, want in zip(gpu_memberships, memberships):
                torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
            torch.testing.assert_close(gpu_cache.applied_keys_with_padding.cpu(), cache.applied_keys_with_padding, rtol=0, atol=0)
            assert torch.all(gpu_other.applied_keys_with_padding.cpu() == -1)
    finally:
        torch.npu.synchronize()
        graph.reset()

    for rows in (1, 2, 1):
        expected = materialize_layer_plan(cache, keys[:rows], slots[:rows], memberships[0][:rows], cacheable_mask=stable[:rows])
        actual = materialize_layer_plan(gpu_cache, gpu_keys[:rows], gpu_slots[:rows], gpu_memberships[0][:rows], cacheable_mask=gpu_stable[:rows])
        torch.testing.assert_close(actual.slots.cpu(), expected.slots, rtol=0, atol=0)
        torch.testing.assert_close(actual.miss_mask.cpu(), expected.miss_mask, rtol=0, atol=0)
        torch.testing.assert_close(gpu_memberships[0].cpu(), memberships[0], rtol=0, atol=0)
        torch.testing.assert_close(gpu_cache.applied_keys_with_padding.cpu(), cache.applied_keys_with_padding, rtol=0, atol=0)


def test_plan_rejects_membership_with_too_few_rows(device):
    build = fused()
    keys = torch.zeros((2, 1), dtype=torch.int64, device=device)
    storage = torch.full((2, 16400), 123, dtype=torch.int16, device=device)
    with pytest.raises(ValueError):
        build(keys, keys, keys, storage[:1], capacity=1)
    assert torch.all(storage.cpu() == 123)


def test_plan_zero_rows_leaves_membership_untouched(device):
    build = fused()
    keys = torch.empty((0, 7), dtype=torch.int64, device=device)
    applied = torch.empty((0, 8), dtype=torch.int64, device=device)
    membership = torch.full((1, 16400), 123, dtype=torch.int16, device=device)
    safe, miss = build(keys, keys, applied, membership, capacity=8)
    assert safe.shape == miss.shape == (0, 7)
    assert torch.all(membership.cpu() == 123)
