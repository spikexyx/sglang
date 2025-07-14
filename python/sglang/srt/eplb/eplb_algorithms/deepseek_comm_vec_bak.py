# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
from typing import Optional, Tuple

import torch


def make_redundant_experts_chunkwise(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_physical_experts_per_chunk: int,
    num_nodes: int, 
    num_gpus: int,
    old_phy_to_log_map: Optional[torch.Tensor] = None,
    old_log_to_phy_map: Optional[torch.Tensor] = None,
    intra_node_penalty: float = 0.5,
    inter_node_penalty: float = 5.0,
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

    if num_local_physical_experts > 1 and old_phy_to_log_map is not None:
        # 1. Pre-calculate physical location info for each expert slot
        gpus_per_node = num_gpus // num_nodes
        phys_expert_indices = torch.arange(num_physical_experts, device=tokens_per_expert.device)
        gpu_ids = phys_expert_indices // num_local_physical_experts
        node_ids = gpu_ids // gpus_per_node

        # 2. Calculate the base load score
        target_phy_to_log_map = physical_to_logical_map.to(torch.int64)
        counts = logical_count.gather(-1, target_phy_to_log_map)
        base_score = tokens_per_expert.sum(0).gather(-1, target_phy_to_log_map)
        base_score = base_score / (counts + 1e-8)  # Shape: [num_moe_layers, num_physical_experts]

        # 3. Create the Communication Cost Matrix (not bonus!)
        # Expand maps for broadcasting
        target_log_experts = target_phy_to_log_map  # Shape: [L, I]
        old_log_experts_at_slot = old_phy_to_log_map  # Shape: [L, J]

        # Scale penalties to be comparable to scores (per layer)
        score_mean_per_layer = base_score.mean(dim=-1, keepdim=True) + 1e-8  # Shape: [L, 1]
        scaled_inter_penalty = inter_node_penalty * score_mean_per_layer
        scaled_intra_penalty = intra_node_penalty * score_mean_per_layer

        # Initialize communication cost matrix (starts at 0 - no penalty)
        comm_cost = torch.zeros(
            num_moe_layers, num_physical_experts, num_physical_experts, 
            device=base_score.device
        )
        
        # Calculate mask for no-move cases (no penalty)
        penalty_mask_no_move = target_log_experts.unsqueeze(2) == old_log_experts_at_slot.unsqueeze(1)
        # No penalty for no-move cases (cost remains 0)

        # Calculate penalty for intra-node moves
        if num_nodes > 1:  # Only needed when there are multiple nodes
            # Find which nodes each logical expert previously occupied
            old_log_on_node = torch.zeros(num_moe_layers, num_logical_experts, num_nodes, device=base_score.device)
            for layer_idx in range(num_moe_layers):
                for slot_idx in range(num_physical_experts):
                    log_expert = old_log_experts_at_slot[layer_idx, slot_idx]
                    if 0 <= log_expert < num_logical_experts:
                        old_log_on_node[layer_idx, log_expert, node_ids[slot_idx]] = 1.0

            # For each target expert i, check if it was on the node of physical slot j
            target_was_on_node_j = torch.zeros(num_moe_layers, num_physical_experts, num_physical_experts, device=base_score.device)
            for layer_idx in range(num_moe_layers):
                for target_idx in range(num_physical_experts):
                    target_log_expert = target_log_experts[layer_idx, target_idx]
                    if 0 <= target_log_expert < num_logical_experts:
                        for slot_idx in range(num_physical_experts):
                            slot_node = node_ids[slot_idx]
                            if old_log_on_node[layer_idx, target_log_expert, slot_node] > 0:
                                target_was_on_node_j[layer_idx, target_idx, slot_idx] = 1.0
            
            # Penalty for intra-node moves (expert moves within same node, but not to same slot)
            penalty_mask_intra_node = (target_was_on_node_j > 0) & (~penalty_mask_no_move)
            comm_cost[penalty_mask_intra_node] = scaled_intra_penalty.expand_as(comm_cost)[penalty_mask_intra_node]

            # Penalty for inter-node moves (expert moves to different node)
            penalty_mask_inter_node = (target_was_on_node_j == 0)
            comm_cost[penalty_mask_inter_node] = scaled_inter_penalty.expand_as(comm_cost)[penalty_mask_inter_node]

        else:
            # Single node case: all moves are intra-node moves (except no-move)
            penalty_mask_intra_node = ~penalty_mask_no_move
            comm_cost[penalty_mask_intra_node] = scaled_intra_penalty.expand_as(comm_cost)[penalty_mask_intra_node]

        # 4. Create the final combined benefit matrix: Load Score - Communication Cost
        benefit_matrix = base_score.unsqueeze(2) - comm_cost

        # 5. Greedy assignment using the benefit matrix (higher benefit = better)
        final_phy_to_log_map = torch.full_like(physical_to_logical_map, -1)
        indices = torch.full((num_moe_layers, num_physical_experts), -1, dtype=torch.long, device=benefit_matrix.device)
        temp_benefit = benefit_matrix.clone()
        
        # Track assigned targets and slots to avoid conflicts
        assigned_targets = torch.zeros(num_moe_layers, num_physical_experts, dtype=torch.bool, device=benefit_matrix.device)
        assigned_slots = torch.zeros(num_moe_layers, num_physical_experts, dtype=torch.bool, device=benefit_matrix.device)
        
        for assignment_step in range(num_physical_experts):
            # Find the highest remaining benefit pair
            max_val, flat_idx = temp_benefit.view(num_moe_layers, -1).max(dim=1)
            target_indices = flat_idx // num_physical_experts
            slot_indices = flat_idx % num_physical_experts

            # Safety check for valid indices
            valid_mask = (target_indices >= 0) & (target_indices < num_physical_experts) & \
                        (slot_indices >= 0) & (slot_indices < num_physical_experts)
            
            if not valid_mask.all():
                # Handle invalid cases by finding next best assignment
                for layer_idx in range(num_moe_layers):
                    if not valid_mask[layer_idx]:
                        # Find first unassigned target and slot
                        unassigned_targets = (~assigned_targets[layer_idx]).nonzero(as_tuple=True)[0]
                        unassigned_slots = (~assigned_slots[layer_idx]).nonzero(as_tuple=True)[0]
                        if len(unassigned_targets) > 0 and len(unassigned_slots) > 0:
                            target_indices[layer_idx] = unassigned_targets[0]
                            slot_indices[layer_idx] = unassigned_slots[0]
                        else:
                            continue

            # Perform the assignment
            for layer_idx in range(num_moe_layers):
                target_idx = target_indices[layer_idx]
                slot_idx = slot_indices[layer_idx]
                
                # Skip if already assigned
                if assigned_targets[layer_idx, target_idx] or assigned_slots[layer_idx, slot_idx]:
                    continue
                    
                assigned_logical_expert = physical_to_logical_map[layer_idx, target_idx]
                final_phy_to_log_map[layer_idx, slot_idx] = assigned_logical_expert
                indices[layer_idx, slot_idx] = target_idx
                
                # Mark as assigned
                assigned_targets[layer_idx, target_idx] = True
                assigned_slots[layer_idx, slot_idx] = True
                
                # Mask out the used target and slot
                temp_benefit[layer_idx, target_idx, :] = -1e9
                temp_benefit[layer_idx, :, slot_idx] = -1e9

        physical_to_logical_map = final_phy_to_log_map
        
        # 6. Update logical_to_physical_map based on the new assignment
        mask = logical_to_physical_map == -1
        logical_to_physical_map[mask] = 0  # Temporarily fill for gather
        
        # Compute inverse mapping
        inverse_indices = indices.argsort(-1)
        
        logical_to_physical_map = inverse_indices.gather(
            -1, logical_to_physical_map.view(num_moe_layers, -1).to(torch.int64)
        ).view_as(logical_to_physical_map).to(torch.int)
        
        logical_to_physical_map[mask] = -1  # Restore mask

    else:
        # Fallback to original logic if cost-aware inputs are not provided
        if num_local_physical_experts > 1:
            # Load-balancing between GPUs
            physical_to_logical_map_int64 = physical_to_logical_map.to(torch.int64)
            counts = logical_count.gather(-1, physical_to_logical_map_int64)
            score = tokens_per_expert.sum(0).gather(-1, physical_to_logical_map_int64)
            score = score / counts
            score = score.view(num_moe_layers, num_chunks, num_physical_experts_per_chunk)
            indices = score.argsort(-1, descending=True)
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

# def make_redundant_experts_chunkwise(
#     tokens_per_expert: torch.Tensor,
#     num_physical_experts: int,
#     num_local_physical_experts: int,
#     num_physical_experts_per_chunk: int,
#     num_nodes: int, 
#     num_gpus: int,
#     old_phy_to_log_map: Optional[torch.Tensor] = None,
#     intra_node_penalty: float = 0.5,
#     inter_node_penalty: float = 5.0,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     num_steps, num_moe_layers, num_logical_experts = tokens_per_expert.shape
#     num_redundancy_experts = num_physical_experts - num_logical_experts

#     physical_to_logical_map = torch.empty(
#         num_moe_layers,
#         num_physical_experts,
#         dtype=torch.int,
#         device=tokens_per_expert.device,
#     )
#     logical_to_physical_map = torch.full(
#         (num_moe_layers, num_logical_experts, num_redundancy_experts + 1),
#         -1,
#         dtype=torch.int,
#         device=tokens_per_expert.device,
#     )
#     logical_count = torch.ones(
#         num_moe_layers,
#         num_logical_experts,
#         dtype=torch.int,
#         device=tokens_per_expert.device,
#     )

#     assert num_physical_experts % num_physical_experts_per_chunk == 0
#     num_chunks = num_physical_experts // num_physical_experts_per_chunk
#     assert num_logical_experts % num_chunks == 0
#     num_logical_experts_per_group = num_logical_experts // num_chunks
#     assert num_redundancy_experts % num_chunks == 0
#     num_redundancy_experts_per_group = num_redundancy_experts // num_chunks

#     arange_num_moe_layers_num_groups = torch.arange(
#         num_moe_layers * num_chunks, dtype=torch.int, device=tokens_per_expert.device
#     )
#     arange_num_logical_experts = torch.arange(
#         num_logical_experts, dtype=torch.int, device=tokens_per_expert.device
#     )
#     arange_num_logical_experts_per_group = torch.arange(
#         num_logical_experts_per_group, dtype=torch.int, device=tokens_per_expert.device
#     )
#     arange_num_groups = torch.arange(
#         num_chunks, dtype=torch.int, device=tokens_per_expert.device
#     )
#     physical_to_logical_map.view(
#         num_moe_layers, num_chunks, num_physical_experts_per_chunk
#     )[:, :, :num_logical_experts_per_group] = arange_num_logical_experts.view(
#         num_chunks, num_logical_experts_per_group
#     )
#     logical_to_physical_map[:, :, 0] = (
#         arange_num_logical_experts_per_group.expand(
#             num_chunks, num_logical_experts_per_group
#         )
#         + arange_num_groups[:, None] * num_physical_experts_per_chunk
#     ).view(num_logical_experts)

#     tokens_per_expert_all_diff = tokens_per_expert + arange_num_logical_experts * 1e-4
#     for i in range(num_redundancy_experts_per_group):
#         score = (
#             tokens_per_expert_all_diff / logical_count
#         )  # NOTE: Values in score must be different from each other
#         score1 = tokens_per_expert / (logical_count + 1)
#         score = score.view(
#             num_steps, num_moe_layers, num_chunks, num_logical_experts_per_group
#         )
#         score1 = score1.view_as(score)
#         values, indices = score.max(-1, keepdim=True)
#         values = values.expand_as(score).contiguous()
#         score.scatter_(-1, indices, score1.gather(-1, indices))
#         values.scatter_(-1, indices, score.max(-1, keepdim=True).values)
#         redundancy_indices = values.sum(0).argmin(-1)
#         physical_to_logical_map.view(
#             num_moe_layers, num_chunks, num_physical_experts_per_chunk
#         )[:, :, num_logical_experts_per_group + i] = (
#             redundancy_indices + arange_num_groups * num_logical_experts_per_group
#         )
#         redundancy_count = (
#             logical_count.view(
#                 num_moe_layers * num_chunks, num_logical_experts_per_group
#             )
#             .gather(-1, redundancy_indices.view(num_moe_layers * num_chunks, 1))
#             .squeeze(1)
#         )
#         physical_redundancy_indices = (
#             (
#                 arange_num_groups * num_physical_experts_per_chunk
#                 + num_logical_experts_per_group
#                 + i
#             )
#             .expand(num_moe_layers, num_chunks)
#             .flatten()
#         )
#         logical_to_physical_map.view(
#             num_moe_layers * num_chunks,
#             num_logical_experts_per_group,
#             num_redundancy_experts + 1,
#         )[
#             arange_num_moe_layers_num_groups,
#             redundancy_indices.view(num_moe_layers * num_chunks),
#             redundancy_count,
#         ] = physical_redundancy_indices
#         logical_count.view(num_moe_layers * num_chunks, num_logical_experts_per_group)[
#             arange_num_moe_layers_num_groups,
#             redundancy_indices.view(num_moe_layers * num_chunks),
#         ] += 1

#     if num_local_physical_experts > 1 and old_phy_to_log_map is not None:
#         # 1. Pre-calculate physical location info
#         # num_gpus = num_physical_experts // num_local_physical_experts
#         gpus_per_node = num_gpus // num_nodes
#         phys_expert_indices = torch.arange(num_physical_experts, device=tokens_per_expert.device)
#         node_ids = (phys_expert_indices // num_local_physical_experts) // gpus_per_node

#         # 2. Calculate the base load score for each potential target expert
#         target_phy_to_log_map = physical_to_logical_map.to(torch.int64)
#         counts = logical_count.gather(-1, target_phy_to_log_map)
#         base_score = tokens_per_expert.sum(0).gather(-1, target_phy_to_log_map)
#         base_score = base_score / (counts + 1e-8)  # Shape: [num_moe_layers, num_physical_experts]

#         # 3. Create the Communication Bonus Matrix directly
#         # Goal: bonus[l, i, j] = bonus for assigning target_expert_i to physical_slot_j
        
#         # Expand maps for broadcasting.
#         # target_log_experts[l, i] is the logical expert ID of the i-th target.
#         target_log_experts = target_phy_to_log_map # Shape: [L, I]
#         # old_log_experts_at_slot[l, j] is the logical expert ID that was at physical slot j.
#         old_log_experts_at_slot = old_phy_to_log_map # Shape: [L, J]

#         # Calculate bonus for "no-move" (highest reward)
#         # bonus_mask[l, i, j] is true if target_i's log_id == old log_id at slot j.
#         bonus_mask_no_move = target_log_experts.unsqueeze(2) == old_log_experts_at_slot.unsqueeze(1)
        
#         # Scale penalties to be comparable to scores
#         # score_mean = base_score.mean([-1], keepdim=True) + 1e-8
#         # scaled_inter_penalty = inter_node_penalty * score_mean
#         # scaled_intra_penalty = intra_node_penalty * score_mean

#         score_mean_per_layer = base_score.mean([-1], keepdim=True) + 1e-8 # Shape: [L, 1]
#         scaled_inter_penalty = inter_node_penalty * score_mean_per_layer.unsqueeze(-1)
#         scaled_intra_penalty = intra_node_penalty * score_mean_per_layer.unsqueeze(-1)

#         # expanded_inter_penalty = scaled_inter_penalty.unsqueeze(-1).expand(-1, -1, num_physical_experts)
#         # expanded_intra_penalty = scaled_intra_penalty.unsqueeze(-1).expand(-1, -1, num_physical_experts)

#         comm_bonus = torch.zeros(
#             num_moe_layers, num_physical_experts, num_physical_experts, 
#             device=base_score.device
#         )
#         comm_bonus[bonus_mask_no_move] = scaled_inter_penalty.expand_as(comm_bonus)[bonus_mask_no_move]
#         # comm_bonus[bonus_mask_no_move] = scaled_inter_penalty
#         # comm_bonus = torch.where(
#         #     bonus_mask_intra_node, 
#         #     expanded_intra_penalty, 
#         #     comm_bonus
#         # )
#         # comm_bonus = torch.where(
#         #     bonus_mask_no_move, 
#         #     expanded_inter_penalty, 
#         #     comm_bonus
#         # )

#         # Calculate bonus for "intra-node move" (secondary reward)
#         # First, find which nodes each logical expert previously occupied.
#         old_log_on_node = torch.zeros(num_moe_layers, num_logical_experts, num_nodes, device=base_score.device)
#         old_log_on_node.scatter_add_(
#             2,
#             node_ids.view(1, 1, -1).expand(num_moe_layers, num_logical_experts, -1),
#             (old_log_experts_at_slot.unsqueeze(1) == torch.arange(num_logical_experts, device=base_score.device).view(1, -1, 1)).float()
#         )
#         old_log_on_node = old_log_on_node > 0 # Shape: [L, num_log_exp, num_nodes]

#         # For each target expert i, check if it was on the node of physical slot j
#         target_was_on_node_j = old_log_on_node.gather(
#             1, target_log_experts.unsqueeze(2).expand(-1, -1, num_nodes).to(torch.int64)
#         ).gather(
#             2, node_ids.view(1, 1, -1).expand(num_moe_layers, num_physical_experts, -1)
#         ) # Shape: [L, I, J]
        
#         bonus_mask_intra_node = target_was_on_node_j & (~bonus_mask_no_move)
#         comm_bonus[bonus_mask_intra_node] = scaled_intra_penalty.expand_as(comm_bonus)[bonus_mask_intra_node]

#         # 4. Create the final combined benefit matrix
#         benefit_matrix = base_score.unsqueeze(2) + comm_bonus

#         # 5. Greedy assignment
#         final_phy_to_log_map = torch.full_like(physical_to_logical_map, -1)
#         indices = torch.full((num_moe_layers, num_physical_experts), -1, dtype=torch.long, device=benefit_matrix.device)
#         temp_benefit = benefit_matrix.clone()

#         for _ in range(num_physical_experts):
#             max_val, flat_idx = temp_benefit.view(num_moe_layers, -1).max(dim=1)
#             target_indices = flat_idx // num_physical_experts
#             slot_indices = flat_idx % num_physical_experts
            
#             assigned_logical_expert = physical_to_logical_map.gather(1, target_indices.unsqueeze(1)).squeeze(1)
#             final_phy_to_log_map.scatter_(1, slot_indices.unsqueeze(1), assigned_logical_expert.unsqueeze(1).int())
#             indices.scatter_(1, slot_indices.unsqueeze(1), target_indices.unsqueeze(1))
            
#             temp_benefit[:, target_indices, :] = -1e9
#             temp_benefit[:, :, slot_indices] = -1e9

#         physical_to_logical_map = final_phy_to_log_map
        
#         # 6. Update logical_to_physical_map
#         mask = logical_to_physical_map == -1
#         logical_to_physical_map[mask] = 0
#         inverse_indices = indices.argsort(-1)
#         logical_to_physical_map = inverse_indices.gather(
#             -1, logical_to_physical_map.view(num_moe_layers, -1).to(torch.int64)
#         ).view_as(logical_to_physical_map).to(torch.int)
#         logical_to_physical_map[mask] = -1
#     else:
#         # Fallback to original logic if cost-aware inputs are not provided
#         if num_local_physical_experts > 1:
#             # Load-balancing between GPUs
#             physical_to_logical_map_int64 = physical_to_logical_map.to(torch.int64)
#             counts = logical_count.gather(-1, physical_to_logical_map_int64)
#             score = tokens_per_expert.sum(0).gather(-1, physical_to_logical_map_int64)
#             score = score / counts
#             score = score.view(num_moe_layers, num_chunks, num_physical_experts_per_chunk)
#             indices = score.argsort(-1, descending=True)
#             indices += torch.arange(
#                 0,
#                 num_physical_experts,
#                 num_physical_experts_per_chunk,
#                 dtype=indices.dtype,
#                 device=indices.device,
#             )[None, :, None]

#             assert num_physical_experts_per_chunk % num_local_physical_experts == 0
#             num_local_groups = num_physical_experts_per_chunk // num_local_physical_experts
#             indices = indices.view(
#                 num_moe_layers, num_chunks, num_local_physical_experts, num_local_groups
#             )
#             indices[:, :, 1::2, :] = indices[:, :, 1::2, :].flip(-1)
#             indices = indices.transpose(2, 3)
#             indices = indices.reshape(num_moe_layers, num_physical_experts)
#             physical_to_logical_map = physical_to_logical_map.gather(-1, indices)
#             mask = logical_to_physical_map == -1
#             logical_to_physical_map[mask] = 0
#             logical_to_physical_map = (
#                 indices.argsort(-1)
#                 .gather(
#                     -1, logical_to_physical_map.view(num_moe_layers, -1).to(torch.int64)
#                 )
#                 .view_as(logical_to_physical_map)
#                 .to(torch.int)
#             )
#             logical_to_physical_map[mask] = -1

#     return physical_to_logical_map, logical_to_physical_map, logical_count

def decode_rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_nodes: int, 
    num_gpus: int,
    old_phy_to_log_map: Optional[torch.Tensor] = None,
    old_log_to_phy_map: Optional[torch.Tensor] = None,
    intra_node_penalty: float = 0.5,
    inter_node_penalty: float = 5.0,
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
    inter_node_penalty: float,
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
        intra_node_penalty=intra_node_penalty,
        inter_node_penalty=inter_node_penalty
    )
