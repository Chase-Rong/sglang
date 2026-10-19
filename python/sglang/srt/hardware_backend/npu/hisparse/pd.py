"""Reuse Ascend's staging leases for HiSparse PD destinations."""


def get_pd_receive_pool(target_pool, draft_pool=None, *, draft_layer_base):
    if getattr(target_pool, "decode_host_backing", None) is None:
        return None
    if draft_pool is None:
        return target_pool

    from sglang.srt.hardware_backend.npu.hisparse.layer_receive_group import (
        LayerReceivePoolGroup,
    )

    return LayerReceivePoolGroup(target_pool, draft_pool, draft_layer_base)
