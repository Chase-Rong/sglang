from collections.abc import Sequence
from typing import Any


def apply_torch_npu_patches(torch_npu: Any, patches: Sequence[Sequence[Any]]) -> None:
    """Apply torch_npu patches across old and new torch_npu patch APIs."""
    if hasattr(torch_npu, "_apply_patches"):
        torch_npu._apply_patches(patches)
        return

    if hasattr(torch_npu, "_apply_all_patches"):
        # Recent torch_npu already applies global patches during import. Calling
        # that entry point again reinstalls its exception hook with itself as
        # the origin, and does not apply the requested profiler replacements.
        import torch

        for path, replacement in patches:
            owner = torch
            components = path.split(".")
            for component in components[:-1]:
                owner = getattr(owner, component)
            setattr(owner, components[-1], replacement)
        return

    raise AttributeError(
        "torch_npu must provide either _apply_patches or _apply_all_patches"
    )
