# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
from typing import Optional, Tuple

import torch


def compute_communication_penalty(
    current_phy_to_log: torch.Tensor,
    old_log_to_phy_map: torch.Tensor,
    num_chunks: int,
    num_physical_experts_per_chunk: int,
    num_nodes: int,
    num_gpus: int,
    intra_node_penalty: float,
    inter_node_penalty: float
) -> torch.Tensor:
    num_layers = current_phy_to_log.shape[0]
    num_gpus_per_node = num_gpus // num_nodes
    
    current_reshaped = current_phy_to_log.view(
        num_layers, num_chunks, num_physical_experts_per_chunk
    )
    
    penalty = torch.zeros_like(current_reshaped, dtype=torch.float32)
    
    for layer in range(num_layers):
        for chunk in range(num_chunks):
            for pos in range(num_physical_experts_per_chunk):
                logical_id = current_reshaped[layer, chunk, pos].item()
                
                if logical_id >= 0:
                    old_phy_positions = old_log_to_phy_map[layer, logical_id]
                    # pass -1
                    valid_old_positions = old_phy_positions[old_phy_positions >= 0]
                    
                    if len(valid_old_positions) > 0:
                        current_phy_pos = chunk * num_physical_experts_per_chunk + pos
                        
                        if current_phy_pos not in valid_old_positions:
                            min_penalty = float('inf')
                            
                            current_gpu = current_phy_pos % num_gpus
                            current_node = current_gpu // num_gpus_per_node
                            
                            for old_phy_pos in valid_old_positions:
                                old_gpu = old_phy_pos.item() % num_gpus
                                old_node = old_gpu // num_gpus_per_node
                                
                                if current_node != old_node:
                                    # inter node
                                    move_penalty = inter_node_penalty
                                else:
                                    # intra node
                                    move_penalty = intra_node_penalty
                                
                                min_penalty = min(min_penalty, move_penalty)
                            
                            penalty[layer, chunk, pos] = min_penalty
    
    return penalty

def compute_communication_penalty_vectorized(
    current_phy_to_log: torch.Tensor,
    old_log_to_phy_map: torch.Tensor,
    num_chunks: int,
    num_physical_experts_per_chunk: int,
    num_nodes: int,
    num_gpus: int,
    intra_node_penalty: float,
    inter_node_penalty: float
) -> torch.Tensor:
    num_layers = current_phy_to_log.shape[0]
    num_gpus_per_node = num_gpus // num_nodes
    device = current_phy_to_log.device
    
    current_reshaped = current_phy_to_log.view(
        num_layers, num_chunks, num_physical_experts_per_chunk
    )
    
    penalty = torch.zeros_like(current_reshaped, dtype=torch.float32)
    
    chunk_indices = torch.arange(num_chunks, device=device)[:, None]
    pos_indices = torch.arange(num_physical_experts_per_chunk, device=device)[None, :]
    current_phy_positions = chunk_indices * num_physical_experts_per_chunk + pos_indices
    current_phy_positions = current_phy_positions.unsqueeze(0).expand(num_layers, -1, -1)
    
    valid_mask = current_reshaped >= 0
    
    if valid_mask.any():
        valid_layers, valid_chunks, valid_positions = torch.where(valid_mask)
        valid_logical_ids = current_reshaped[valid_mask]
        valid_current_phy_pos = current_phy_positions[valid_mask]
        
        valid_old_phy_maps = old_log_to_phy_map[valid_layers, valid_logical_ids]  # [N, X]
        
        penalties = []
        for i in range(len(valid_logical_ids)):
            layer_idx = valid_layers[i]
            logical_id = valid_logical_ids[i]
            current_phy_pos = valid_current_phy_pos[i]
            
            old_positions = valid_old_phy_maps[i]
            valid_old_positions = old_positions[old_positions >= 0]
            
            if len(valid_old_positions) == 0:
                penalties.append(0.0)
                continue
                
            if current_phy_pos in valid_old_positions:
                penalties.append(0.0)
            else:
                current_gpu = current_phy_pos % num_gpus
                current_node = current_gpu // num_gpus_per_node
                
                old_gpus = valid_old_positions % num_gpus
                old_nodes = old_gpus // num_gpus_per_node
                
                inter_node_moves = old_nodes != current_node
                move_costs = torch.where(
                    inter_node_moves,
                    torch.tensor(inter_node_penalty, device=device),
                    torch.tensor(intra_node_penalty, device=device)
                )
                
                penalties.append(move_costs.min().item())
        
        penalty[valid_mask] = torch.tensor(penalties, dtype=torch.float32, device=device)
    
    return penalty

def normalize_score(score: torch.Tensor) -> torch.Tensor:
    score_flat = score.view(-1, score.shape[-1])
    min_vals = score_flat.min(dim=-1, keepdim=True)[0]
    max_vals = score_flat.max(dim=-1, keepdim=True)[0]
    
    range_vals = max_vals - min_vals
    range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
    
    normalized = (score_flat - min_vals) / range_vals
    return normalized.view_as(score)


def make_redundant_experts_chunkwise(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_physical_experts_per_chunk: int,
    # New args
    num_nodes: int,
    num_gpus: int,
    old_phy_to_log_map: Optional[torch.Tensor] = None,
    old_log_to_phy_map: Optional[torch.Tensor] = None,
    comm_weight: float = 0.2,
    intra_node_penalty: float = 0.5,
    inter_node_penalty: float = 5.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_steps, num_moe_layers, num_logical_experts = tokens_per_expert.shape
    num_redundancy_experts = num_physical_experts - num_logical_experts

    physical_to_logical_map = torch.empty(
        num_moe_layers,
        num_physical_experts,
        dtype=torch.int,
        device=tokens_per_expert.device,
    )
    logical_to_physical_map = torch.full(
        (num_moe_layers, num_logical_experts, num_redundancy_experts + 1),
        -1,
        dtype=torch.int,
        device=tokens_per_expert.device,
    )
    logical_count = torch.ones(
        num_moe_layers,
        num_logical_experts,
        dtype=torch.int,
        device=tokens_per_expert.device,
    )

    assert num_physical_experts % num_physical_experts_per_chunk == 0
    num_chunks = num_physical_experts // num_physical_experts_per_chunk
    assert num_logical_experts % num_chunks == 0
    num_logical_experts_per_group = num_logical_experts // num_chunks
    assert num_redundancy_experts % num_chunks == 0
    num_redundancy_experts_per_group = num_redundancy_experts // num_chunks

    arange_num_moe_layers_num_groups = torch.arange(
        num_moe_layers * num_chunks, dtype=torch.int, device=tokens_per_expert.device
    )
    arange_num_logical_experts = torch.arange(
        num_logical_experts, dtype=torch.int, device=tokens_per_expert.device
    )
    arange_num_logical_experts_per_group = torch.arange(
        num_logical_experts_per_group, dtype=torch.int, device=tokens_per_expert.device
    )
    arange_num_groups = torch.arange(
        num_chunks, dtype=torch.int, device=tokens_per_expert.device
    )
    physical_to_logical_map.view(
        num_moe_layers, num_chunks, num_physical_experts_per_chunk
    )[:, :, :num_logical_experts_per_group] = arange_num_logical_experts.view(
        num_chunks, num_logical_experts_per_group
    )
    logical_to_physical_map[:, :, 0] = (
        arange_num_logical_experts_per_group.expand(
            num_chunks, num_logical_experts_per_group
        )
        + arange_num_groups[:, None] * num_physical_experts_per_chunk
    ).view(num_logical_experts)

    tokens_per_expert_all_diff = tokens_per_expert + arange_num_logical_experts * 1e-4
    for i in range(num_redundancy_experts_per_group):
        score = (
            tokens_per_expert_all_diff / logical_count
        )  # NOTE: Values in score must be different from each other
        score1 = tokens_per_expert / (logical_count + 1)
        score = score.view(
            num_steps, num_moe_layers, num_chunks, num_logical_experts_per_group
        )
        score1 = score1.view_as(score)
        values, indices = score.max(-1, keepdim=True)
        values = values.expand_as(score).contiguous()
        score.scatter_(-1, indices, score1.gather(-1, indices))
        values.scatter_(-1, indices, score.max(-1, keepdim=True).values)
        redundancy_indices = values.sum(0).argmin(-1)
        physical_to_logical_map.view(
            num_moe_layers, num_chunks, num_physical_experts_per_chunk
        )[:, :, num_logical_experts_per_group + i] = (
            redundancy_indices + arange_num_groups * num_logical_experts_per_group
        )
        redundancy_count = (
            logical_count.view(
                num_moe_layers * num_chunks, num_logical_experts_per_group
            )
            .gather(-1, redundancy_indices.view(num_moe_layers * num_chunks, 1))
            .squeeze(1)
        )
        physical_redundancy_indices = (
            (
                arange_num_groups * num_physical_experts_per_chunk
                + num_logical_experts_per_group
                + i
            )
            .expand(num_moe_layers, num_chunks)
            .flatten()
        )
        logical_to_physical_map.view(
            num_moe_layers * num_chunks,
            num_logical_experts_per_group,
            num_redundancy_experts + 1,
        )[
            arange_num_moe_layers_num_groups,
            redundancy_indices.view(num_moe_layers * num_chunks),
            redundancy_count,
        ] = physical_redundancy_indices
        logical_count.view(num_moe_layers * num_chunks, num_logical_experts_per_group)[
            arange_num_moe_layers_num_groups,
            redundancy_indices.view(num_moe_layers * num_chunks),
        ] += 1

    if num_local_physical_experts > 1:
        # Load-balancing between GPUs
        physical_to_logical_map_int64 = physical_to_logical_map.to(torch.int64)
        counts = logical_count.gather(-1, physical_to_logical_map_int64)
        score = tokens_per_expert.sum(0).gather(-1, physical_to_logical_map_int64)
        # score = score / counts
        # score = score.view(num_moe_layers, num_chunks, num_physical_experts_per_chunk)
        load_score = score / counts
        load_score = load_score.view(num_moe_layers, num_chunks, num_physical_experts_per_chunk)

        # Add communication into consideration
        if old_log_to_phy_map is not None:
            comm_penalty = compute_communication_penalty_vectorized(
                physical_to_logical_map,
                old_log_to_phy_map,
                num_chunks,
                num_physical_experts_per_chunk,
                num_nodes,
                num_gpus,
                intra_node_penalty,
                inter_node_penalty
            )

            normalized_load = normalize_score(load_score)
            normalized_comm = normalize_score(-comm_penalty)

            combined_score = (1 - comm_weight) * normalized_load + comm_weight * normalized_comm
            indices = combined_score.argsort(-1, descending=True)
        else:
            indices = load_score.argsort(-1, descending=True)

        # indices = score.argsort(-1, descending=True)
        indices += torch.arange(
            0,
            num_physical_experts,
            num_physical_experts_per_chunk,
            dtype=indices.dtype,
            device=indices.device,
        )[None, :, None]

        assert num_physical_experts_per_chunk % num_local_physical_experts == 0
        num_local_groups = num_physical_experts_per_chunk // num_local_physical_experts
        indices = indices.view(
            num_moe_layers, num_chunks, num_local_physical_experts, num_local_groups
        )
        indices[:, :, 1::2, :] = indices[:, :, 1::2, :].flip(-1)
        indices = indices.transpose(2, 3)
        indices = indices.reshape(num_moe_layers, num_physical_experts)
        physical_to_logical_map = physical_to_logical_map.gather(-1, indices)
        mask = logical_to_physical_map == -1
        logical_to_physical_map[mask] = 0
        logical_to_physical_map = (
            indices.argsort(-1)
            .gather(
                -1, logical_to_physical_map.view(num_moe_layers, -1).to(torch.int64)
            )
            .view_as(logical_to_physical_map)
            .to(torch.int)
        )
        logical_to_physical_map[mask] = -1

    return physical_to_logical_map, logical_to_physical_map, logical_count


def decode_rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_nodes: int,
    num_gpus: int,
    old_phy_to_log_map: Optional[torch.Tensor] = None,
    old_log_to_phy_map: Optional[torch.Tensor] = None,
    comm_weight: float = 0.2,
    intra_node_penalty: float = 0.5,
    inter_node_penalty: float = 5.0
):
    return make_redundant_experts_chunkwise(
        tokens_per_expert,
        num_physical_experts,
        num_local_physical_experts,
        num_physical_experts,
        num_nodes,
        num_gpus,
        old_phy_to_log_map,
        old_log_to_phy_map,
        comm_weight,
        intra_node_penalty,
        inter_node_penalty
    )


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: Optional[int],
    num_nodes: int,
    num_gpus: int,
    intra_node_penalty: float,
    inter_node_penalty: float
):
    from sglang.srt.eplb.expert_location import (
        get_global_expert_location_metadata,
    )
    old_phy_to_log_map = get_global_expert_location_metadata().physical_to_logical_map
    old_log_to_phy_map = get_global_expert_location_metadata().logical_to_all_physical_map
    return decode_rebalance_experts(
        tokens_per_expert=tokens_per_expert,
        num_physical_experts=num_physical_experts,
        num_local_physical_experts=num_local_physical_experts,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        old_phy_to_log_map=old_phy_to_log_map,
        old_log_to_phy_map=old_log_to_phy_map,
        comm_weight=0.2,
        intra_node_penalty=intra_node_penalty,
        inter_node_penalty=inter_node_penalty
    )
