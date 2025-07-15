# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
from typing import Tuple, Optional

import torch
import numpy as np

from typing import List

from sglang.srt.utils import get_bool_env_var


def cost_function(
    # --- Identifiers ---
    layer_id: int,
    target_gpu: int,
    expert_phy_id: int,
    # --- Cost calculation data ---
    expert_weight: float,
    current_pack_weights: List[float],
    # --- Affinity calculation data ---
    phy2mlog: torch.Tensor,
    old_log2phy: torch.Tensor,
    gpus_per_node: int,
    groups_per_pack: int,
    # --- Penalty factors (hyperparameters) ---
    intra_node_penalty_factor: float,
    inter_node_penalty_factor: float
) -> float:
    # 1. Load Cost: The current weight of the target pack.
    load_cost = current_pack_weights[target_gpu]

    # 2. Communication Affinity Cost
    communication_penalty = 0.0

    # Find the expert's logical ID
    logical_id = phy2mlog[layer_id, expert_phy_id].item()

    # Find the GPU where this expert was previously located
    old_phy_expert_indices = old_log2phy[layer_id, logical_id]

    valid_old_phy_indices = old_phy_expert_indices[old_phy_expert_indices >= 0]

    if valid_old_phy_indices.numel() == 0:
        pass
    else:
        valid_old_gpus = valid_old_phy_indices // groups_per_pack

        if (valid_old_gpus == target_gpu).any():
            communication_penalty = 0.0
        else:
            min_penalty = float('inf')
            target_node_id = target_gpu // gpus_per_node

            for old_gpu_tensor in valid_old_gpus:
                old_gpu = old_gpu_tensor.item()
                old_node_id = old_gpu // gpus_per_node

                current_penalty = 0.0
                if old_node_id == target_node_id:
                    current_penalty = intra_node_penalty_factor * expert_weight
                else:
                    current_penalty = inter_node_penalty_factor * expert_weight

                if current_penalty < min_penalty:
                    min_penalty = current_penalty

    # 3. Total Cost
    total_cost = load_cost + communication_penalty
    return total_cost

def balanced_packing_with_affinity(
    weight: torch.Tensor, 
    num_packs: int, 
    phy2mlog: torch.Tensor, 
    old_log2phy: torch.Tensor, 
    num_nodes: int,
    num_gpus: int,
    intra_node_penalty_factor: float = 0.2,
    inter_node_penalty_factor: float = 1.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs
    packs_per_node = num_packs // num_nodes

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    # pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device=weight.device)
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)

    weight_cpu = weight.cpu()
    phy2mlog_cpu = phy2mlog.cpu()
    old_log2phy_cpu = old_log2phy.cpu()

    for i in range(num_layers):
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            group = group.item()
            pack = min(
                (k for k in range(num_packs) if pack_items[k] < groups_per_pack),
                key=lambda k: cost_function(
                    i, k, group, weight_cpu[i, group].item(), pack_weights, phy2mlog_cpu, old_log2phy_cpu, packs_per_node, groups_per_pack, intra_node_penalty_factor, inter_node_penalty_factor
                )
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight_cpu[i, group].item()
            pack_items[pack] += 1
    return pack_index, rank_in_pack

def fast_balanced_packing(
    weight: torch.Tensor, 
    num_packs: int, 
    phy2mlog: torch.Tensor, 
    old_log2phy: torch.Tensor, 
    num_nodes: int,
    num_gpus: int,
    intra_node_penalty_factor: float = 0.2,
    inter_node_penalty_factor: float = 1.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    num_layers, num_groups = weight.shape
    groups_per_pack = num_groups // num_packs
    
    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1), dtype=torch.int64, device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    weight_np = weight.cpu().numpy()
    sorted_indices = weight.argsort(axis=-1)
    indices = torch.flip(sorted_indices, dims=[1]).cpu().numpy()
    # indices = weight.argsort(axis=-1)[:, ::-1].cpu().numpy()
    
    pack_index = np.full_like(weight_np, fill_value=-1, dtype=np.int64)
    rank_in_pack = np.full_like(pack_index, fill_value=-1)
    
    affinity_penalties = precompute_affinity_matrix(
        phy2mlog, old_log2phy, num_layers, num_groups, num_packs, 
        groups_per_pack, num_nodes, intra_node_penalty_factor, inter_node_penalty_factor
    )
    
    for layer_id in range(num_layers):
        pack_weights = np.zeros(num_packs)
        pack_counts = np.zeros(num_packs, dtype=np.int32)
        
        for expert_id in indices[layer_id]:
            expert_weight = weight_np[layer_id, expert_id]
            
            available_mask = pack_counts < groups_per_pack
            available_packs = np.where(available_mask)[0]
            
            load_costs = pack_weights[available_packs]
            comm_penalties = affinity_penalties[layer_id, expert_id, available_packs]
            total_costs = load_costs + comm_penalties * expert_weight
            
            best_idx = np.argmin(total_costs)
            best_pack = available_packs[best_idx]
            
            pack_index[layer_id, expert_id] = best_pack
            rank_in_pack[layer_id, expert_id] = pack_counts[best_pack]
            pack_weights[best_pack] += expert_weight
            pack_counts[best_pack] += 1
    
    return torch.from_numpy(pack_index), torch.from_numpy(rank_in_pack)


def precompute_affinity_matrix(phy2mlog, old_log2phy, num_layers, num_groups, num_packs, 
                              groups_per_pack, num_nodes, intra_factor, inter_factor):
    packs_per_node = num_packs // num_nodes
    penalties = np.zeros((num_layers, num_groups, num_packs))
    
    phy2mlog_np = phy2mlog.cpu().numpy()
    old_log2phy_np = old_log2phy.cpu().numpy()
    
    for layer_id in range(num_layers):
        for expert_id in range(num_groups):
            logical_id = phy2mlog_np[layer_id, expert_id]
            old_indices = old_log2phy_np[layer_id, logical_id]
            valid_old_indices = old_indices[old_indices >= 0]
            
            if len(valid_old_indices) == 0:
                continue
                
            old_gpus = valid_old_indices // groups_per_pack
            
            for pack_id in range(num_packs):
                if pack_id in old_gpus:
                    penalties[layer_id, expert_id, pack_id] = 0.0
                else:
                    pack_node = pack_id // packs_per_node
                    old_nodes = old_gpus // packs_per_node
                    
                    if np.any(old_nodes == pack_node):
                        penalties[layer_id, expert_id, pack_id] = intra_factor
                    else:
                        penalties[layer_id, expert_id, pack_id] = inter_factor
    
    return penalties

def balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (i for i in range(num_packs) if pack_items[i] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack

def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt

def rebalance_experts_with_affinity(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    nnodes: int,
    intra_node_penalty_factor: float = 0.2,
    inter_node_penalty_factor: float = 1.2
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`
        nnodes: number of server nodes(for use)

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    from sglang.srt.eplb.expert_location import (
        # ExpertLocationMetadata,
        get_global_expert_location_metadata,
    )
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    old_ep_metadata = get_global_expert_location_metadata()
    old_log2phy = old_ep_metadata.logical_to_all_physical_map

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = fast_balanced_packing(tokens_per_phy, num_gpus // num_nodes, phy2mlog, old_log2phy, nnodes, num_gpus, intra_node_penalty_factor, inter_node_penalty_factor)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy
    )  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    intra_node_penalty_factor: float,
    inter_node_penalty_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """

    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    
    # use global load-balance policy with affinity
    phy2log, phyrank, logcnt = rebalance_experts_with_affinity(
        weight, num_replicas, 1, 1, num_gpus, num_nodes, intra_node_penalty_factor, inter_node_penalty_factor
    )

    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts"]
