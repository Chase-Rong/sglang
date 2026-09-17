"""One PD lease covering distinct target and draft model KV namespaces."""

from sglang.srt.hardware_backend.npu.layer_protocol import LayerReceiveLease
from sglang.srt.hardware_backend.npu.offload_config import validate_mtp_pool_pair


class LayerReceivePoolGroup:
    def __init__(self, target_pool, draft_pool, draft_layer_base):
        validate_mtp_pool_pair(target_pool, draft_pool)
        self.pools = ((0, target_pool), (draft_layer_base - draft_pool.start_layer, draft_pool))
        self.page_size = target_pool.page_size
        self.host_backings = tuple(pool.decode_host_backing for _, pool in self.pools)
        if any(backing is None for backing in self.host_backings):
            raise ValueError("target and draft must both provide registered Host KV")
        self.device_id = self.host_backings[0].pool.device_id
        inventory = set()
        ownership = []
        for (offset, pool), backing in zip(self.pools, self.host_backings):
            local_ids = (tuple(pool.full_attention_layer_id_mapping)
                         if hasattr(pool, "full_attention_layer_id_mapping")
                         else range(pool.start_layer, pool.start_layer + pool.layer_num))
            layer_ids = {offset + layer for layer in local_ids}
            if not layer_ids or min(layer_ids) < 0 or inventory.intersection(layer_ids):
                raise ValueError("target and draft transfer layer namespaces overlap")
            if pool.page_size != self.page_size or backing.pool.device_id != self.device_id:
                raise ValueError("target and draft receive pools must share page geometry and device")
            group = getattr(backing.pool, "shared_group", None)
            ownership.append(None if group is None else (group.writer_rank, group.group.ranks))
            inventory.update(layer_ids)
        if ownership[0] != ownership[1]:
            raise ValueError("target and draft Host backing must have matching writer groups")

    def layer_receive_lease(self, room, session_id, host_pages, device_pages, *, prefix=0):
        layers = {}
        visibility = {}
        leases = []
        for offset, pool in self.pools:
            lease = pool.layer_receive_lease(room, session_id, host_pages, device_pages, prefix=prefix)
            leases.append(lease)
            for layer_id, regions in lease.layers.items():
                wire_layer = offset + layer_id
                layers[wire_layer] = regions
                visibility[wire_layer] = (lease.visibility, layer_id)
        first = leases[0]
        if any(lease.page_ids != first.page_ids for lease in leases[1:]):
            raise ValueError("target and draft leases disagree on physical request pages")

        def complete_visibility(layer_id, start, end):
            callback, local_layer = visibility[layer_id]
            callback(local_layer, start, end)

        return LayerReceiveLease(
            room, session_id, self.page_size, layers, first.page_ids, complete_visibility,
            prefix=prefix, shared_group_id=first.shared_group_id,
            shared_host_writer=first.shared_host_writer,
        )
