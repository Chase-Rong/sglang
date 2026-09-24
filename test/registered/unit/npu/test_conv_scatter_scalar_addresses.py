"""CPU interpreter regression for Ascend's eager scalar masked loads.

Run with TRITON_INTERPRET=1. Every scalar address must be in bounds even
when masked off: the Ascend adapter lowers scalar loads before the select.
Vector accesses retain their ordinary masked semantics. No NPU is used.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "CPU interpreter only")
class TestConvScatterScalarAddresses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from triton.runtime.interpreter import interpreter_builder

        cls.builder = interpreter_builder
        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py"
        )
        spec = importlib.util.spec_from_file_location("conv_scatter_under_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def check_scatter(self, n1, n2, num_types=1):
        module = self.module
        module._conv_multi_meta_cache.clear()
        ranges = []

        def register(tensor):
            storage = tensor.untyped_storage()
            ranges.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
            return tensor

        pairs, expected = [], []
        batch = max(n1, n2, 1)
        for t in range(num_types):
            # Last type crosses a vector block boundary. Src has nontrivial strides.
            dim = 7 if t == 0 else 345
            src = register(
                torch.arange(2 * batch * 4 * 3 * dim, dtype=torch.float32)
                .reshape(2, batch, 4, 3, dim)
                .transpose(-1, -2)
                .to(torch.bfloat16)
            )
            dst = register(
                torch.full((2, n1 + n2 + 2, dim, 3), -5, dtype=torch.bfloat16)
            )
            pairs.append((dst, src))
            expected.append(dst.clone())

        idx1 = register(torch.arange(n1, dtype=torch.int32))
        idx2 = register(torch.arange(n1, n1 + n2, dtype=torch.int64))
        st1 = register(torch.tensor([i % 4 for i in range(n1)], dtype=torch.int32))
        st2 = register(
            torch.tensor(
                [-1 if i % 2 == 0 else 3 for i in range(n2)], dtype=torch.int32
            )
        )
        if n1 > 2:
            st1[1] = -1
            st1[2] = 4  # Out-of-range step must leave its destination unchanged.
        for want, (_, src) in zip(expected, pairs):
            for indices, steps in ((idx1, st1), (idx2, st2)):
                for i, step in enumerate(steps.tolist()):
                    if 0 <= step < 4:
                        want[:, int(indices[i])] = src[:, i, step]

        original_load = self.builder.create_masked_load
        original_meta = module._conv_multi_build_meta

        def checked_load(ptrs, mask, other, *args):
            addresses = np.asarray(ptrs.data).reshape(-1)
            # Simulate the observed scalar lowering, including masked-off loads.
            active = (
                np.ones(addresses.size, dtype=bool)
                if addresses.size == 1
                else np.asarray(mask.data).reshape(-1)
            )
            width = ptrs.get_element_ty().primitive_bitwidth // 8
            for address in addresses[active]:
                address = int(address)
                if not any(
                    lo <= address and address + width <= hi for lo, hi in ranges
                ):
                    raise AssertionError("scalar/vector read outside tensor storage")
            return original_load(ptrs, mask, other, *args)

        def tracked_meta(*args):
            meta, blocks = original_meta(*args)
            register(meta)
            return meta, blocks

        with (
            patch.object(self.builder, "create_masked_load", checked_load),
            patch.object(module, "_conv_multi_build_meta", tracked_meta),
        ):
            module.fused_conv_window_scatter_multi(
                pairs, idx1, st1, idx2 if n2 else None, st2 if n2 else None
            )
        for (actual, _), want in zip(pairs, expected):
            torch.testing.assert_close(actual, want, rtol=0, atol=0)

    def test_commit_only(self):
        self.check_scatter(3, 0)

    def test_track_only(self):
        self.check_scatter(0, 3)

    def test_both_empty(self):
        self.check_scatter(0, 0)

    def test_commit_and_track(self):
        self.check_scatter(3, 3)

    def test_unequal_groups_and_multiple_types(self):
        self.check_scatter(3, 5, num_types=2)

    def test_full_batch(self):
        self.check_scatter(24, 24, num_types=2)


if __name__ == "__main__":
    unittest.main()
