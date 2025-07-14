# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
from typing import Optional, Tuple

import torch


def pack_groups(tokens_per_group: torch.Tensor, num_nodes: int) -> torch.Tensor:
    num_layers, num_groups = tokens_per_group.shape
    assert num_groups % num_nodes == 0
    groups_per_rank = num_groups // num_nodes

    indices = tokens_per_group.float().sort(-1, descending=True).indices.cpu()
    ret = torch.full_like(
        tokens_per_group, fill_value=-1, dtype=torch.int64, device="cpu"
    )
    for layer in range(num_layers):
        node_tokens = [0] * num_nodes
        node_groups = [0] * num_nodes
        for group in indices[layer]:

            def key_func(rank: int) -> int:
                if node_groups[rank] >= groups_per_rank:
                    return 1, 0
                else:
                    return 0, node_tokens[rank]

            rank = min(range(num_nodes), key=key_func)
            assert node_groups[rank] < groups_per_rank
            ret[layer, group] = rank * groups_per_rank + node_groups[rank]
            node_tokens[rank] += tokens_per_group[layer, group]
            node_groups[rank] += 1
    return ret


def make_redundant_experts_chunkwise(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_physical_experts_per_chunk: int,
    num_nodes: int, 
    num_gpus: int,
    old_phy_to_log_map: Optional[torch.Tensor] = None,
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
        # This is the cost-aware load-balancing logic
        
        # 1. Pre-calculate physical location info for each expert slot
        # This is done once and is very fast
        # num_gpus = num_physical_experts // num_local_physical_experts
        gpus_per_node = num_gpus // num_nodes
        
        phys_expert_indices = torch.arange(num_physical_experts, device=tokens_per_expert.device)
        gpu_ids = phys_expert_indices // num_local_physical_experts
        node_ids = gpu_ids // gpus_per_node

        # 2. Calculate the base load score, same as original code
        # `physical_to_logical_map` here is the "target map" before final placement
        target_phy_to_log_map = physical_to_logical_map.to(torch.int64)
        counts = logical_count.gather(-1, target_phy_to_log_map)
        base_score = tokens_per_expert.sum(0).gather(-1, target_phy_to_log_map)
        base_score = base_score / counts  # Shape: [num_moe_layers, num_physical_experts]

        # 3. Create the Communication Cost/Benefit Matrix
        # We want to calculate Benefit[l, i, j]: 
        # the benefit of assigning the i-th target expert to the j-th physical slot.
        # Shape: [num_moe_layers, num_physical_experts, num_physical_experts]
        
        # Expand old and new maps for broadcasting
        # target_log_experts[l, i, j] is the logical expert of the i-th target
        target_log_experts = target_phy_to_log_map.unsqueeze(2).expand(-1, -1, num_physical_experts)
        # old_log_experts_at_slot[l, i, j] is the logical expert that was at physical slot j
        old_log_experts_at_slot = old_phy_to_log_map.unsqueeze(1).expand(-1, num_physical_experts, -1)
        
        # Find where the target expert *used to be*
        # old_slots_of_target_expert[l, i, k] is the k-th old physical slot of the i-th target expert
        old_slots_of_target_expert = (old_phy_to_log_map.unsqueeze(1) == target_phy_to_log_map.unsqueeze(2)).int()

        # Calculate node locations for all old slots
        old_node_ids_of_slots = node_ids.expand(num_moe_layers, num_physical_experts)
        
        # For each target expert, what nodes was it on previously?
        # A bit of einsum magic to check presence: 1 if target_i was on node_k, 0 otherwise
        target_on_old_node = torch.einsum(
            'lij,lj->lik', 
            old_slots_of_target_expert.float(), 
            torch.nn.functional.one_hot(node_ids, num_nodes).float()
        ) > 0 # Shape: [num_layers, num_targets, num_nodes]

        # Calculate communication penalty
        # penalty[l, i, j] = penalty for moving target_i to physical_slot_j
        current_node_of_slot_j = node_ids.view(1, 1, -1) # Shape: [1, 1, num_physical_experts]
        is_on_different_node = (1.0 - target_on_old_node.gather(2, current_node_of_slot_j.expand(num_moe_layers, num_physical_experts, -1)).float())
        # Note: This simple model assumes if an expert needs to move to a new node,
        # it will incur the inter_node_penalty. A more complex model could calculate min distance.
        # For simplicity and performance, we check if the target node is one of the source nodes.
        
        # A simpler check: does the assignment match exactly?
        is_same_logical_expert = (target_log_experts == old_log_experts_at_slot)
        
        # Let's use a clear penalty definition:
        # If logical expert is the same, penalty is 0.
        # If different, but a copy of the target expert exists on the destination node, penalty is intra_node.
        # If no copy exists on the destination node, penalty is inter_node.
        
        # We can approximate this with a bonus for staying put
        comm_bonus = torch.zeros_like(base_score).unsqueeze(1).expand(-1, num_physical_experts, -1)
        
        # Bonus for exact match (staying in the same slot)
        # Find where target expert `i` matches an old expert `j`
        bonus_mask = (target_phy_to_log_map.unsqueeze(2) == old_phy_to_log_map.unsqueeze(1))
        # We give the max penalty (inter_node_penalty) as a bonus for staying put
        comm_bonus[bonus_mask] = inter_node_penalty * base_score.mean() # Scale bonus by avg score

        # 4. Create the final combined benefit matrix
        # benefit[i, j] = benefit of assigning target i to physical slot j
        # We normalize score to be comparable to penalties
        normalized_score = base_score / (base_score.mean() + 1e-6)
        
        # The benefit matrix combines load and communication bonus
        # Benefit[l, i, j] = score of target i + bonus of placing target i in slot j
        benefit_matrix = normalized_score.unsqueeze(2) + comm_bonus

        # 5. Greedy assignment using the benefit matrix
        # We iteratively assign the best remaining target to the best remaining slot.
        final_phy_to_log_map = torch.full_like(physical_to_logical_map, -1)
        
        # To make this fast and vectorized, we find the max benefit pairs and assign them,
        # then mask them out and repeat. This is a greedy approximation.
        temp_benefit = benefit_matrix.clone()
        
        # `indices` will map from the new physical slot to the *index* of the target expert
        indices = torch.full((num_moe_layers, num_physical_experts), -1, dtype=torch.long, device=benefit_matrix.device)
        
        for _ in range(num_physical_experts):
            # Find the highest remaining benefit pair (target_idx, slot_idx)
            max_val, flat_idx = temp_benefit.view(num_moe_layers, -1).max(dim=1)
            target_indices = flat_idx // num_physical_experts
            slot_indices = flat_idx % num_physical_experts

            # Perform the assignment
            # Get the actual logical expert ID from the original target map
            assigned_logical_expert = physical_to_logical_map.gather(1, target_indices.unsqueeze(1)).squeeze(1)
            final_phy_to_log_map.scatter_(1, slot_indices.unsqueeze(1), assigned_logical_expert.unsqueeze(1).int())
            
            # Record which target was assigned to which new slot, for log_to_phy_map update
            indices.scatter_(1, slot_indices.unsqueeze(1), target_indices.unsqueeze(1))
            
            # Mask out the used target and slot to prevent re-assignment
            temp_benefit.scatter_(1, target_indices.unsqueeze(1).expand(-1, -1, num_physical_experts), -1e9)
            temp_benefit.scatter_(2, slot_indices.unsqueeze(1).unsqueeze(2).expand(-1, num_physical_experts, -1, -1).squeeze(2), -1e9)

        physical_to_logical_map = final_phy_to_log_map
        
        # 6. Update logical_to_physical_map based on the new assignment
        # The original `indices` from argsort is now our computed `indices`.
        # We need to compute its inverse to update log_to_phy_map correctly.
        mask = logical_to_physical_map == -1
        logical_to_physical_map[mask] = 0 # Temporarily fill for gather
        
        # `indices` tells us: new_slot -> old_target_index
        # We need `argsort` for the inverse: old_target_index -> new_slot
        inverse_indices = indices.argsort(-1)
        
        logical_to_physical_map = inverse_indices.gather(
            -1, logical_to_physical_map.view(num_moe_layers, -1).to(torch.int64)
        ).view_as(logical_to_physical_map).to(torch.int)
        
        logical_to_physical_map[mask] = -1 # Restore mask

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


def decode_rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_nodes: int, 
    num_gpus: int,
    old_phy_to_log_map: Optional[torch.Tensor] = None,
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
        intra_node_penalty,
        inter_node_penalty
    )


def prefill_rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: int,
    num_nodes: int,
):
    tokens_per_expert = tokens_per_expert.float().cpu()

    num_steps, _, num_logical_experts = tokens_per_expert.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0, f"{num_groups=} {num_nodes=}"

    tokens_per_group = tokens_per_expert.sum(0).unflatten(-1, (num_groups, -1)).sum(-1)
    group_perm = pack_groups(
        tokens_per_group, num_nodes
    )  # [num_moe_layers, num_groups] => [num_moe_layers, num_nodes]

    # log2mlog [layers, #logexp] -> [layers, #logexp]
    log2mlog = (
        (group_perm * group_size).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_perm.device)
    ).flatten(-2)

    # mlog2log [layers, #logexp] -> [layers, #logexp], inverse of log2mlog
    mlog2log = torch.empty_like(log2mlog)
    arange = torch.arange(
        num_logical_experts, dtype=torch.int64, device=mlog2log.device
    )
    mlog2log.scatter_(1, log2mlog, arange.expand(log2mlog.size(0), -1))

    # tokens_per_mlog[i][j][k] = tokens_per_expert[i][j][mlog2log[j][k]]
    tokens_per_mlog = tokens_per_expert.gather(
        2, mlog2log.unsqueeze(0).expand(num_steps, -1, -1)
    )

    phy2mlog, mlog2phy, mlog_count = make_redundant_experts_chunkwise(
        tokens_per_mlog,
        num_physical_experts,
        num_local_physical_experts,
        num_physical_experts // num_nodes,
    )

    # phy2log[i][j] = mlog2log[i][phy2mlog[i][j]]
    phy2log = mlog2log.gather(1, phy2mlog.to(torch.int64))

    # mlog2phy: [num_moe_layers, num_logical_experts, ...]
    # log2phy[i][j][k] = mlog2phy[i][log2mlog[i][j]][k]
    log2phy = mlog2phy.gather(
        1, log2mlog.unsqueeze(-1).expand(-1, -1, mlog2phy.size(-1)).to(torch.int64)
    )

    # log_count[i][j] = mlog_count[i][log2mlog[i][j]]
    log_count = mlog_count.gather(1, log2mlog)
    return phy2log, log2phy, log_count


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: Optional[int],
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
    intra_node_penalty: float,
    inter_node_penalty: float,
):
    if enable_hierarchical:
        return prefill_rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_groups=num_groups,
            num_nodes=num_nodes,
        )
    else:
        from sglang.srt.eplb.expert_location import (
            ExpertLocationMetadata,
            get_global_expert_location_metadata,
        )
        old_phy_to_log_map = get_global_expert_location_metadata().physical_to_logical_map
        return decode_rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=num_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
            old_phy_to_log_map=old_phy_to_log_map,
            intra_node_penalty=intra_node_penalty,
            inter_node_penalty=inter_node_penalty
        )
