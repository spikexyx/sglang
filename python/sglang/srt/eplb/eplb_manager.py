import logging
import time
from typing import TYPE_CHECKING, List, Optional

import torch.cuda

import threading

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ExpertLocationMetadata

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Add logger handler for debug
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class EPLBManager:
    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self._model_runner = model_runner
        self._server_args = model_runner.server_args
        self._rebalance_layers_per_chunk = (
            self._server_args.eplb_rebalance_layers_per_chunk
        )
        self._rebalance_num_iterations = self._server_args.eplb_rebalance_num_iterations
        # Used for deepseek_comm algorithm
        self._old_experts_metadata = None
        self._comm_penalty = None
        self._comm_check = False

        # Otherwise, the circular buffer will contain stale data. If the case is needed, it can be implemented.
        assert (
            self._server_args.eplb_rebalance_num_iterations
            >= self._server_args.expert_distribution_recorder_buffer_size
        ), "eplb_rebalance_num_iterations must be greater than expert_distribution_recorder_buffer_size"

        if not get_global_expert_distribution_recorder().recording:
            get_global_expert_distribution_recorder().start_record()

        logger.info(
            f"[EPLBManager] system started, will rebalance per {self._rebalance_num_iterations} iterations."
        )

        self._main_generator = self._entrypoint()

    def _post_rebalance_handler(self):
        logger.info("[EPLBManager] post rebalance handler start")
        if self._old_experts_metadata is None:
            logger.info("[EPLBManager] post rebalance handler end with None")
            return
        
        old = self._old_experts_metadata
        world_size = old.ep_size
        num_nodes = self._server_args.nnodes
        assert world_size % num_nodes == 0, "world_size must be divisible by num_nodes"
        gpus_per_node = world_size // num_nodes
        
        num_layers = old.num_layers
        num_log_ep = old.num_logical_experts
        num_phy_ep = old.num_physical_experts

        assert num_phy_ep % world_size == 0, "num_phy_ep must be divisible by world_size"
        num_phy_ep_per_gpu = num_phy_ep // world_size

        penalty = torch.zeros((num_layers, num_log_ep, num_phy_ep), dtype=torch.float32)

        expert_gpu = torch.arange(num_phy_ep, device=old.logical_to_all_physical_map.device) // num_phy_ep_per_gpu
        expert_node = expert_gpu // gpus_per_node

        old_log2phy = old.logical_to_all_physical_map
        old_logcnt = old.logical_to_all_physical_map_num_valid

        L, X, k_max = old_log2phy.shape
        Y = expert_node.shape[0]

        # check valid
        valid = torch.arange(k_max, device=old_log2phy.device).view(1,1,k_max) \
                < old_logcnt.unsqueeze(-1)                      # (L, X, k_max)

        phys_gpus = expert_gpu[old_log2phy]
        target_gpus = expert_gpu.view(1,1,1,Y)

        in_gpu_set = (phys_gpus.unsqueeze(-1) == target_gpus) & valid.unsqueeze(-1)
        in_gpu_set = in_gpu_set.any(dim=2)

        phys_node = expert_node[old_log2phy]                    # (L, X, k_max)

        # (L, X, k_max, Y), if True: y == phys_mapped[k]
        y_idx = torch.arange(Y, device=old_log2phy.device).view(1,1,1,Y)
        in_phys = (old_log2phy.unsqueeze(-1) == y_idx) & valid.unsqueeze(-1)   # (L,X,k_max,Y)
        in_phys = in_phys.any(dim=2)                            # (L, X, Y)

        node_idx = expert_node.view(1,1,1,Y)
        in_node_set = (phys_node.unsqueeze(-1) == node_idx) & valid.unsqueeze(-1)
        in_node_set = in_node_set.any(dim=2)                    # (L, X, Y)

        penalty = torch.zeros((L, X, Y), dtype=torch.float32, device=old_log2phy.device)
        penalty[~in_gpu_set & ~in_node_set] = self._server_args.eplb_inter_node_penalty   # cross-node
        penalty[~in_gpu_set &  in_node_set] = self._server_args.eplb_intra_node_penalty  # same-node cross-gpu

        self._comm_penalty = penalty.cpu()
        logger.info("[EPLBManager] post rebalance handler end")

    def on_forward_pass_end(self):
        next(self._main_generator)

    # can be more complex if needed
    def _entrypoint(self):
        while True:
            for _ in range(self._rebalance_num_iterations):
                yield

            yield from self.rebalance()

    def rebalance(self):
        logger.info("[EPLBManager] rebalance start")

        # If recorder mode is historical_dynamic, and ep balancedness still sufficient, skip rebalancing.
        if self._server_args.expert_distribution_recorder_mode == "historical_dynamic":
            if get_global_expert_distribution_recorder().get_balancedness() >= self._server_args.eplb_rebalance_threshold:
                logger.info(
                    "[EPLBManager] rebalance skipped, balancedness is sufficient."
                )
                return

        enable_timing = self._rebalance_layers_per_chunk is None

        if enable_timing:
            torch.cuda.synchronize()
            time_start = time.time()

        logical_count = get_global_expert_distribution_recorder().dump_record(
            output_mode="object"
        )["logical_count"]
        expert_location_metadata = ExpertLocationMetadata.init_by_eplb(
            self._server_args, self._model_runner.model_config, logical_count, self._comm_penalty
        )

        logger.info("[EPLBManager] New EPLB location metadata init_by_eplb end, start to update")

        msg = f"[EPLBManager] EPLB algorithm compute time:"
        if enable_timing:
            torch.cuda.synchronize()
            time_middle = time.time()
            msg += f" ={time_middle - time_start:.3f}s"
        logger.info(msg)

        update_layer_ids_chunks = self._compute_update_layer_ids_chunks()
        for chunk_index, update_layer_ids in enumerate(update_layer_ids_chunks):
            if len(update_layer_ids_chunks) > 1:
                yield
            self._model_runner.update_expert_location(
                expert_location_metadata,
                update_layer_ids=update_layer_ids,
            )

        msg = f"[EPLBManager] EPLB communication time:"
        if enable_timing:
            torch.cuda.synchronize()
            time_middle_2 = time.time()
            msg += f" = {time_middle_2 - time_middle:.3f}s"
        logger.info(msg)

        if self._comm_check and self._server_args.eplb_algorithm in ["deepseek_comm", "deepseek_opt"]:
            self._old_experts_metadata = expert_location_metadata
            thread = threading.Thread(
                target=self._post_rebalance_handler,
                args=(),
                daemon=True,
            )
            thread.start()

        msg = f"[EPLBManager] rebalance end"
        if enable_timing:
            torch.cuda.synchronize()
            time_end = time.time()
            msg += f" time= {time_end - time_start:.3f}s"
        logger.info(msg)

        self._comm_check = True

    def _compute_update_layer_ids_chunks(self) -> List[List[int]]:
        all_layer_ids = sorted(
            list(self._model_runner.model.routed_experts_weights_of_layer.keys())
        )
        chunk_size = self._rebalance_layers_per_chunk or 1000000
        return list(_chunk_list(all_layer_ids, chunk_size=chunk_size))


def _chunk_list(items: List, chunk_size):
    for start_index in range(0, len(items), chunk_size):
        yield items[start_index : start_index + chunk_size]
