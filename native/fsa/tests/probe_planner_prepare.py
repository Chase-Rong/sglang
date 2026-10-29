"""Real NPU planner ABI3 probe, including mutable-union0/hash collision/prefix.

Run with --library /absolute/sgl_fsa_extension.so. No Host registration or SFA
payload is involved: this establishes planner evidence only. The caller owns
remote/device scheduling and must retain the JSON report.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from planner_reference import key_hash, reference

DEVICE = "npu:7"


def tensors(keys, *, ends=None, owners=None, capacity=128, width=16):
    T, K = len(keys), len(keys[0])
    ends = [T] if ends is None else ends
    owners = [1] if owners is None else owners
    B, Q = len(ends), 5
    dev = DEVICE

    def full(shape, value, dtype):
        return torch.full(shape, value, dtype=dtype, device=dev)

    inp = [
        torch.tensor(keys, dtype=torch.int64, device=dev),
        torch.tensor(ends, dtype=torch.int32, device=dev),
        torch.tensor(owners, dtype=torch.int64, device=dev),
        full((T, K), True, torch.bool),
        torch.arange(T * K, dtype=torch.int32, device=dev).view(T, 1, K),
        full((B,), 128, torch.int32),
        full((Q, capacity), -1, torch.int64),
        full((Q, capacity), False, torch.bool),
        full((Q, capacity), False, torch.bool),
        full((Q, capacity), False, torch.bool),
        full((Q,), 0, torch.int32),
        full((Q,), False, torch.bool),
        full((Q,), 0, torch.int8),
        full((), 0, torch.int32),
    ]
    out = [full((T, K), -9, torch.int64)]
    out += [full((T, K), True, torch.bool) for _ in range(3)]
    out += [
        full((T,), -9, torch.int64),
        full((T,), -9, torch.int32),
        full((T, K), -9, torch.int32),
    ]
    out += [full((B, width), -9, torch.int64)]
    out += [full((B, width), -9, torch.int32) for _ in range(2)]
    out += [full((B, width), True, torch.bool) for _ in range(2)]
    out += [
        full((B,), -9, torch.int32),
        full((B,), True, torch.bool),
        full((B, capacity), -9, torch.int64),
        full((B, width), -9, torch.int32),
        full((B,), -9, torch.int32),
        full((B, 9), -9, torch.int64),
        full((B, 8), -9, torch.int64),
        full((T, 1, K), -9, torch.int32),
        full((T, 16400), -9, torch.int16),
        full((T, capacity // 128), -9, torch.int32),
        full((T, 1, (K + 8) // 8 * 8), -9, torch.int32),
    ]
    return inp, out


def run(ops, inp, out, *, routes=2, host_tokens=1 << 42):
    ops.external_plan_prepare(inp, out, inp[6].shape[1], routes, host_tokens, 3)
    torch.npu.synchronize()


def check_single(ops, keys, *, mutable=False, duplicate_alias=False, occupied=False):
    inp, out = tensors(keys)
    if mutable:
        inp[3][0, 0] = False
    if occupied:
        inp[6][1] = torch.arange(500, 628, device=DEVICE)
        inp[7][1] = True
        inp[8][1] = True
        inp[9][1] = True
    inp[6][1, 3] = keys[0][0]
    inp[7][1, 3] = inp[8][1, 3] = inp[9][1, 3] = True
    if duplicate_alias:
        inp[6][1, 5] = keys[0][0]
        inp[7][1, 5] = inp[8][1, 5] = inp[9][1, 5] = True
    before = [t.cpu() for t in inp]
    expected = reference(
        keys,
        before[3].tolist(),
        before[6][1].tolist(),
        before[7][1].tolist(),
        before[8][1].tolist(),
        before[9][1].tolist(),
        length=128,
    )
    run(ops, inp, out)
    count = len(expected["keys"])
    assert out[12].cpu().tolist() == [count]
    for index, name in [
        (7, "keys"),
        (8, "slots"),
        (9, "owner"),
        (10, "miss"),
        (11, "cacheable"),
    ]:
        assert out[index][0, :count].cpu().tolist() == expected[name], (
            name,
            out[index].cpu(),
            expected,
        )
    assert out[6].flatten().cpu().tolist() == expected["row_union"]
    assert inp[9][1].cpu().tolist() == expected["reusable"]
    assert torch.equal(inp[8].cpu(), before[8]), "planner must preserve payload_valid"
    assert torch.equal(inp[9][2:].cpu(), before[9][2:]), "unrelated owners changed"
    assert out[18].cpu().eq(0).all() and out[17].cpu().eq(-1).all()
    assert out[16].cpu().eq(0).all() and not out[13].cpu().any()
    assert out[2].sum().item() == sum(expected["miss"])
    member = out[20].cpu()
    for row in range(len(keys)):
        assert member[row, 16384:16393].tolist() == [
            0x5A4D,
            0x5A45,
            len(keys[0]),
            16384 - len(keys[0]),
            1,
            0x5A53,
            128,
            3,
            16,
        ]
    seq = member.view(torch.int32).flatten()
    assert seq[:2].tolist() == [len(keys), 128]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("physical device remapping must be unset")
    global DEVICE
    DEVICE = f"npu:{args.device}"
    torch.npu.set_device(args.device)
    torch.ops.load_library(args.library)
    ops = torch.ops.sgl_fsa
    assert list(ops.external_plan_prepare_capabilities()) == [
        3,
        16384,
        16384,
        2051,
        1024,
        8,
        221184,
        197632,
        1,
        1,
        1,
    ]
    results = []

    def case(name, fn):
        fn()
        results.append({"case": name, "pass": True})

    collide = next(k for k in range(1, 10000) if key_hash(k) % 32 == key_hash(0) % 32)
    case(
        "mutable_union0_collision_all_aliases",
        lambda: check_single(
            ops, [[0, collide, 0, -1]], mutable=True, duplicate_alias=True
        ),
    )
    case(
        "full64_physical_alias_hits",
        lambda: check_single(ops, [[5, (1 << 40) + 5, 5, -1]], duplicate_alias=True),
    )
    case(
        "protected_hit_occupied_victims",
        lambda: check_single(ops, [[500, 501, 700, 701]], occupied=True),
    )
    case(
        "crossrow_one_writer", lambda: check_single(ops, [[4, 2, 4, -1], [2, 3, 4, -1]])
    )

    def empty_prefix():
        inp, out = tensors([[0, 1, -1, -1], [1, 2, -1, -1]], ends=[0, 2], owners=[0, 1])
        for _ in range(8):
            run(ops, inp, out)
            assert out[20].cpu().view(torch.int32).flatten()[:4].tolist() == [
                0,
                2,
                128,
                128,
            ]
            assert out[12].cpu().tolist() == [0, 3]
            assert out[4].cpu().tolist() == [1, 1]

    case("empty_request0_other_block_owns_row0_prefix", empty_prefix)

    def malformed():
        inp, out = tensors([[0, 1, -1, -1], [1, 2, -1, -1]], ends=[2, 1], owners=[1, 2])
        run(ops, inp, out)
        assert out[12].cpu().eq(0).all() and out[19].cpu().eq(-1).all()
        assert out[16].cpu().ne(0).all() and inp[13].item() != 0

    case("malformed_partition_failclosed", malformed)

    def alias_reject():
        inp, out = tensors([[0, 1, -1, -1]])
        inp[9] = inp[8]
        try:
            run(ops, inp, out)
        except RuntimeError as error:
            assert "mutable input alias" in str(error)
        else:
            raise AssertionError("mutable input alias accepted")

    case("mutable_input_alias_rejected_before_launch", alias_reject)

    def maximum_ub():
        keys = torch.arange(6 * 2051, dtype=torch.int64).view(6, 2051).tolist()
        inp, out = tensors(keys, capacity=16384, width=16384)
        inp[5].fill_(20000)
        run(ops, inp, out, routes=6)
        assert out[12].item() == 6 * 2051
        assert out[8][0, : 6 * 2051].cpu().tolist() == list(range(6 * 2051))
        assert out[16].item() == 0 and inp[13].item() == 0

    case("maximum193KiB_UB_R6_K2051_C16384_W16384", maximum_ub)

    def graph_fixed_inputs():
        inp, out = tensors([[0, 1, 0, -1]])
        run(ops, inp, out)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            ops.external_plan_prepare(inp, out, 128, 2, 1 << 42, 3)
        inp[0][0, 1] = 2
        inp[3][0, 0] = False
        graph.replay()
        torch.npu.synchronize()
        assert out[7][0, :2].cpu().tolist() == [0, 2]
        assert out[11][0, :2].cpu().tolist() == [False, True]
        assert out[12].item() == 2 and out[16].item() == 0
        graph.reset()

    case("Graph_fixed_address_updated_key_and_cacheability", graph_fixed_inputs)
    Path(args.report).write_text(
        json.dumps({"hardware": "NPU", "cases": results}, indent=2) + "\n"
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
