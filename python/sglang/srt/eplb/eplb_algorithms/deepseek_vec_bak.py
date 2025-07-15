# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
from typing import Optional, Tuple

import torch


def make_redundant_experts_chunkwise(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_physical_experts_per_chunk: int,
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
):
    return make_redundant_experts_chunkwise(
        tokens_per_expert,
        num_physical_experts,
        num_local_physical_experts,
        num_physical_experts,
    )


def rebalance_experts(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
    num_groups: Optional[int],
    num_nodes: int,
    enable_hierarchical: bool,
):
    return decode_rebalance_experts(
        tokens_per_expert=tokens_per_expert,
        num_physical_experts=num_physical_experts,
        num_local_physical_experts=num_local_physical_experts,
    )
