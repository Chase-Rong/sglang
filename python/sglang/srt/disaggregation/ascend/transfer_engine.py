import logging
import os
from typing import List

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.environ import envs
from sglang.srt.utils.common import run_with_deadline
from sglang.srt.utils.network import NetworkAddress

try:
    from memfabric_hybrid import TransferEngine

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


class AscendTransferEngine(MooncakeTransferEngine):
    def __init__(
        self,
        hostname: str,
        npu_id: int,
        disaggregation_mode: DisaggregationMode,
    ):
        if import_error is not None:
            logger.warning(
                "Please install memfabric_hybrid, for details, see docs/docs/advanced_features/pd_disaggregation.mdx"
            )
            raise import_error

        self.engine = TransferEngine()
        self.hostname = hostname
        self.npu_id = npu_id
        from sglang.srt.hardware_backend.npu.hisparse.offload_config import (
            resolve_transfer_device,
        )

        self.physical_npu_id = resolve_transfer_device(
            npu_id, envs.ASCEND_NPU_PHY_ID.get(), os.getenv("ASCEND_RT_VISIBLE_DEVICES")
        )
        logger.info(
            "Ascend transfer device: logical=%d physical=%d",
            self.npu_id,
            self.physical_npu_id,
        )

        # Centralized storage address of the AscendTransferEngine
        self.store_url = os.getenv("ASCEND_MF_STORE_URL")
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = "Prefill"
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.role = "Decode"
        else:
            logger.error(f"Unsupported DisaggregationMode: {disaggregation_mode}")
            raise ValueError(f"Unsupported DisaggregationMode: {disaggregation_mode}")
        rpc_port = self.engine.get_rpc_port()
        self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()
        self.initialize()
        if rpc_port == 0:
            rpc_port = self.engine.get_rpc_port()
            self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()

    def initialize(self) -> None:
        from sglang.srt.distributed.parallel_state import (
            get_world_group,
            get_world_size,
        )

        transfer_protocol = self._get_transfer_protocol()
        if transfer_protocol is None or transfer_protocol == "sdma":
            trans_op_type = TransferEngine.TransDataOpType.SDMA
        else:
            trans_op_type = (
                TransferEngine.TransDataOpType.DEVICE_URMA
                if transfer_protocol == "device_urma"
                else TransferEngine.TransDataOpType.DEVICE_RDMA
            )
            """Initialize HCCL before device transport initialization."""
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(get_world_size())
            ]
            # Initialize hccl in advance through all_gather to avoid conflicts with rdma initialization.
            torch.distributed.all_gather(
                output_tensor_list, tmp_tensor, group=get_world_group().device_group
            )
        """Initialize the ascend transfer instance."""
        engine_arguments = (
            self.store_url,
            self.session_id,
            self.role,
            self.npu_id,
            trans_op_type,
        )
        ret_value = run_with_deadline(
            lambda: self.engine.initialize(*engine_arguments),
            timeout_s=envs.SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT.get(),
            what=f"Ascend TransferEngine.initialize({self.store_url!r}, {self.session_id!r})",
        )
        if ret_value != 0:
            logger.error("Ascend Transfer Engine initialization failed.")
            raise RuntimeError("Ascend Transfer Engine initialization failed.")

    def batch_register(self, ptrs: List[int], lengths: List[int]):
        if len(ptrs) != len(lengths):
            raise ValueError("Ascend registration pointer and length counts differ")
        if getattr(self, "_registration_failed", False):
            raise RuntimeError("Ascend registration state requires engine restart")
        registered = getattr(self, "_registered_regions", {})
        pending = {}
        for pointer, length in zip(ptrs, lengths):
            if (
                type(pointer) is not int
                or type(length) is not int
                or min(pointer, length) <= 0
            ):
                raise ValueError(
                    "Ascend registration requires positive pointers and lengths"
                )
            previous = pending.get(pointer, registered.get(pointer))
            if previous is not None and previous != length:
                raise ValueError(
                    "Ascend registration changed an existing region length"
                )
            if pointer not in registered:
                pending[pointer] = length
        if not pending:
            return 0
        try:
            status = self.engine.batch_register_memory(
                list(pending), list(pending.values())
            )
            if status != 0:
                raise RuntimeError(f"Ascend memory registration failed: {status}")
        except Exception:
            self._registration_failed = True
            for pointer in pending:
                try:
                    self.engine.unregister_memory(pointer)
                except Exception:
                    logger.exception("Could not roll back Ascend registration")
            raise
        registered.update(pending)
        self._registered_regions = registered
        return 0

    def batch_deregister(self, ptrs):
        registered = getattr(self, "_registered_regions", {})
        for pointer in dict.fromkeys(ptrs):
            if pointer not in registered:
                continue
            status = self.engine.unregister_memory(pointer)
            if status != 0:
                raise RuntimeError(f"Ascend memory deregistration failed: {status}")
            del registered[pointer]
        return 0

    @staticmethod
    def _get_transfer_protocol():
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL")
        allowed_protocols = {"device_rdma", "device_urma", "sdma"}
        if protocol and protocol.lower() in allowed_protocols:
            return protocol.lower()
        else:
            logger.warning(
                "Invalid or no transfer protocol specified, using default protocol."
            )
            return None
