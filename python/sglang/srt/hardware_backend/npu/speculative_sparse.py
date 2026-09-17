"""Supported GLM MTP layout and conservative speculative-cache bounds."""


def sparse_mtp_tail(algorithm, spec, model_config):
    if algorithm.is_none():
        return 0
    name = getattr(algorithm, "name", "")
    if name not in ("EAGLE", "EAGLE3") or int(spec.speculative_eagle_topk or 0) != 1:
        raise ValueError(
            "NPU sparse Host offload supports EAGLE MTP chains with topk=1"
        )
    if not getattr(model_config, "num_nextn_predict_layers", 0):
        raise ValueError(
            "NPU sparse Host offload requires a matching GLM/DeepSeek MTP draft"
        )
    if getattr(spec, "enable_multi_layer_eagle", False):
        raise ValueError(
            "NPU sparse Host offload does not support widened multi-layer draft windows"
        )
    if getattr(spec, "speculative_adaptive", False):
        raise ValueError(
            "NPU sparse Host offload requires fixed MTP width for its cache and memory budget"
        )
    if getattr(spec, "speculative_draft_window_size", None) is not None:
        raise ValueError("NPU sparse Host offload requires full-context draft KV")
    steps = int(spec.speculative_num_steps or 0)
    rows = int(spec.speculative_num_draft_tokens or 0)
    if steps <= 0 or rows <= 0 or rows > steps + 1:
        raise ValueError("invalid MTP chain verification width")
    return rows + steps + 1


def sparse_mtp_rows(algorithm, spec, model_config):
    if not sparse_mtp_tail(algorithm, spec, model_config):
        return 1
    return max(int(spec.speculative_num_draft_tokens), 1)


def sparse_mtp_draft_layers(kvc):
    if kvc.spec_algorithm.is_none() or kvc.is_draft_worker:
        return 0
    count = kvc.spec_aux_config.eagle_draft_num_layers
    if count is None or int(count) <= 0:
        raise ValueError("offload sizing requires the actual MTP draft layer count")
    return int(count)
