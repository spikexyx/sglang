from enum import Enum, auto
from typing import Optional

import torch

from sglang.srt.eplb.eplb_algorithms import deepseek, deepseek_vec, deepseek_comm, deepseek_comm_vec


class EplbAlgorithm(Enum):
    deepseek = auto()
    deepseek_hierarchical = auto()
    deepseek_vec = auto()
    deepseek_vec_hierarchical = auto()
    # TODO may have more algorithm later
    deepseek_comm = auto()
    deepseek_comm_vec = auto()


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: Optional[int],
    num_nodes: int,
    algorithm: EplbAlgorithm,
    intra_node_penalty_factor: float,
    inter_node_penalty_factor: float,
):
    if algorithm in [EplbAlgorithm.deepseek, EplbAlgorithm.deepseek_hierarchical]:
        return deepseek.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_hierarchical,
        )

    if algorithm in [
        EplbAlgorithm.deepseek_vec,
        EplbAlgorithm.deepseek_vec_hierarchical,
    ]:
        return deepseek_vec.rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            enable_hierarchical=algorithm == EplbAlgorithm.deepseek_vec_hierarchical,
        )
    
    if algorithm in [
        EplbAlgorithm.deepseek_comm,
    ]:
        return deepseek_comm.rebalance_experts(
            weight=tokens_per_expert.sum(dim=0),
            num_replicas=num_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            intra_node_penalty_factor = intra_node_penalty_factor,
            inter_node_penalty_factor = inter_node_penalty_factor
        )
    
    if algorithm in [
        EplbAlgorithm.deepseek_comm_vec,
    ]:
        return deepseek_comm_vec.rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_physical_experts // num_local_physical_experts,
            intra_node_penalty = intra_node_penalty_factor,
            inter_node_penalty = inter_node_penalty_factor
        )

    raise NotImplementedError


def compute_algorithm(
    raw_algorithm: str,
    num_groups: Optional[int],
    num_nodes: int,
) -> EplbAlgorithm:
    if raw_algorithm != "auto":
        return EplbAlgorithm[raw_algorithm]

    # TODO test on real scenarios and know which ones perform better
    if (num_groups is not None) and (num_groups % num_nodes == 0):
        return EplbAlgorithm.deepseek_hierarchical
    else:
        return EplbAlgorithm.deepseek
