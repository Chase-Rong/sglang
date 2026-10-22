"""Reuse Ascend's staging leases for HiSparse PD destinations."""


def validate_prefill_bootstrap_policy(disaggregation, *, layerwise_offload):
    if not layerwise_offload:
        return
    role = disaggregation.disaggregation_mode
    if getattr(role, "value", role) != "prefill":
        return
    if disaggregation.disaggregation_decode_allocation_policy == "prefill_complete":
        raise ValueError(
            "NPU HiSparse layerwise Prefill requires layer negotiation before compute; "
            "--disaggregation-decode-allocation-policy=prefill_complete is unsupported. "
            "Use --disaggregation-decode-allocation-policy=early and "
            "--optimistic-prefill-attempts=0."
        )
    if disaggregation.optimistic_prefill_attempts:
        raise ValueError(
            "NPU HiSparse layerwise Prefill requires layer negotiation before compute; "
            "use --optimistic-prefill-attempts=0."
        )


def get_pd_receive_pool(target_pool, draft_pool=None, *, draft_layer_base):
    if getattr(target_pool, "decode_host_backing", None) is None:
        return None
    if draft_pool is None:
        return target_pool

    from sglang.srt.hardware_backend.npu.hisparse.layer_receive_group import (
        LayerReceivePoolGroup,
    )

    return LayerReceivePoolGroup(target_pool, draft_pool, draft_layer_base)
