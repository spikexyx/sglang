from typing import Tuple, Optional

import torch

from sglang.srt.utils import get_bool_env_var

def rebalance_experts_with_affinity(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    comm_penalty: Optional[torch.Tensor] = None,
):
    num_layers, num_logical_experts = weight.shape
    assert num_physical_experts % num_local_physical_experts == 0
    num_gpus = num_physical_experts // num_local_physical_experts
    num_redundancy_experts = num_physical_experts - num_logical_experts

    physical_to_logical_map = torch.empty(
        num_layers,
        num_physical_experts,
        dtype=torch.int,
        device=weight.device,
    )
    logical_to_physical_map = torch.full(
        (num_layers, num_logical_experts, num_redundancy_experts + 1),
        -1,
        dtype=torch.int,
        device=weight.device,
    )
    logical_count = torch.ones(
        num_layers,
        num_logical_experts,
        dtype=torch.int,
        device=weight.device,
    )

    arange_num_moe_layers = torch.arange(
        num_layers, dtype = torch.int, device=weight.device
    )
    arange_num_logical_experts = torch.arange(
        num_logical_experts, dtype = torch.int, device=weight.device
    )
    
    physical_to_logical_map[:, :num_logical_experts] = arange_num_logical_experts[None, :]
    logical_to_physical_map[:, :, 0] = arange_num_logical_experts[None, :]

    # Replicate experts
    weight_all_diff = weight + arange_num_logical_experts * 1e-4
    for i in range(num_redundancy_experts):
        score = weight_all_diff / logical_count
        score1 = weight / (logical_count + 1)
        
        score1 = score1.view_as(score)
        values, indices = score.max(-1, keepdim=True)
        values = values.expand_as(score).contiguous()
        score.scatter_(-1, indices, score1.gather(-1, indices))
        values.scatter_(-1, indices, score.max(-1, keepdim=True).values)
        redundancy_indices = values.argmin(-1)
        physical_to_logical_map[:, num_logical_experts + i] = redundancy_indices
        redundancy_count = (
            logical_count.gather(-1, redundancy_indices.view(num_layers, 1)).squeeze(1)
        )
        
        physical_redundancy_indices = torch.full(
            (num_layers,),
            num_logical_experts + i,
            dtype=torch.int,
            device=weight.device
        )
        logical_to_physical_map[
            arange_num_moe_layers,
            redundancy_indices,
            redundancy_count,
        ] = physical_redundancy_indices
        logical_count[
            arange_num_moe_layers,
            redundancy_indices,
        ] += 1

    # Load-balance between devices
    if num_gpus > 1:
        if comm_penalty is not None:
            comm_penalty = comm_penalty.to(weight.device)

        physical_to_logical_map_int64 = physical_to_logical_map.to(torch.int64)
        counts = logical_count.gather(-1, physical_to_logical_map_int64)
        score = weight.gather(-1, physical_to_logical_map_int64)
        score = score / counts
        
        sorted_scores, sorted_indices = score.sort(-1, descending=True)

        gpu_loads = torch.zeros(num_layers, num_gpus, dtype=score.dtype, device=weight.device)
        gpu_ep_counts = torch.zeros(num_layers, num_gpus, dtype=torch.long, device=weight.device)

        # balanced_indices = torch.full_like(score, -1, dtype=torch.long, device=weight.device)
        sorted_expert_final_pos = torch.full_like(sorted_indices, -1)

        for i in range(num_physical_experts):
            expert_score = sorted_scores[:, i]
            # logic_idx = sorted_indices[:, i]
            logic_idx = physical_to_logical_map_int64[
                torch.arange(num_layers, device=weight.device),
                sorted_indices[:, i]
            ]

            masked_gpu_loads = gpu_loads.clone()
            full_gpus_mask = (gpu_ep_counts >= num_local_physical_experts)
            masked_gpu_loads[full_gpus_mask] = torch.finfo(score.dtype).max

            # calculate move penalty
            g = torch.arange(num_gpus, device=weight.device).view(1, -1)
            y = g * num_local_physical_experts + gpu_ep_counts.unsqueeze(1)
            y = torch.clamp(y, 0, num_physical_experts - 1)

            if comm_penalty is not None:
                move = comm_penalty[
                    torch.arange(num_layers, device=weight.device).view(-1, 1),
                    logic_idx.view(-1, 1),
                    y
                ]
                new_load = masked_gpu_loads + (masked_gpu_loads + 1.0) * move
            else:
                new_load = masked_gpu_loads

            target_gpu = new_load.argmin(dim=1)
            slot_on_gpu = gpu_ep_counts.gather(1, target_gpu.unsqueeze(1)).squeeze(1)

            final_pos = target_gpu * num_local_physical_experts + slot_on_gpu

            sorted_expert_final_pos[:, i] = final_pos

            gpu_loads.scatter_add_(1, target_gpu.unsqueeze(1), expert_score.unsqueeze(1))
            gpu_ep_counts.scatter_add_(1, target_gpu.unsqueeze(1), torch.ones_like(target_gpu.unsqueeze(1)))

        balanced_indices = torch.full_like(sorted_indices, -1)
        balanced_indices.scatter_(-1, sorted_expert_final_pos, sorted_indices)

        physical_to_logical_map = physical_to_logical_map.gather(-1, balanced_indices)

        mask = logical_to_physical_map == -1
        logical_to_physical_map[mask] = 0

        inverse_balanced_indices = balanced_indices.argsort(-1)
        logical_to_physical_map = inverse_balanced_indices.gather(
            -1, logical_to_physical_map.view(num_layers, -1).to(torch.int64)
        ).view_as(logical_to_physical_map).to(torch.int)
        
        logical_to_physical_map[mask] = -1

    return physical_to_logical_map, logical_to_physical_map, logical_count

def rebalance_experts(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    comm_penalty: Optional[torch.Tensor] = None,
):
    weight = weight.float().cpu()
    phy2log, log2phy, logcnt = rebalance_experts_with_affinity(
        weight, num_physical_experts, num_local_physical_experts, comm_penalty
    )
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts"]