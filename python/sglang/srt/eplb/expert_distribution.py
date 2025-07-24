# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import os
import time
from abc import ABC
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import einops
import torch
import torch.distributed

from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import Withable, get_bool_env_var

logger = logging.getLogger(__name__)

# 添加控制台 方便调试输出
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --------------------------------------- Entrypoint -----------------------------------------

_OutputMode = Literal["file", "object"]


class ExpertDistributionRecorder(ABC):
    """Global expert distribution recording"""

    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        if server_args.expert_distribution_recorder_mode is not None:
            return _ExpertDistributionRecorderReal(
                server_args, expert_location_metadata, rank
            )
        else:
            return _ExpertDistributionRecorderNoop()

    @contextmanager
    def with_current_layer(self, layer_idx):
        yield

    @contextmanager
    def with_debug_name(self, debug_name):
        yield

    @contextmanager
    def disable_this_region(self):
        yield

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        yield

    def on_select_experts(self, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def start_record(self):
        self._on_not_implemented()

    def stop_record(self):
        self._on_not_implemented()

    def dump_record(self, output_mode: _OutputMode = "file"):
        self._on_not_implemented()

    def get_balancedness(self) -> float:
        return 0.0

    @property
    def recording(self):
        return False

    def _on_not_implemented(self):
        raise Exception(
            "Please set ServerArgs.expert_distribution_recorder_mode to use ExpertDistributionRecorder."
        )


class _ExpertDistributionRecorderNoop(ExpertDistributionRecorder):
    pass


class _ExpertDistributionRecorderReal(ExpertDistributionRecorder):
    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata

        self._recording = False
        self._disable_all = False
        self._current_forward_pass_id = Withable()
        self._current_layer_idx = Withable()
        self._current_debug_name = Withable()
        self._accumulator = _Accumulator.init_new(
            server_args, expert_location_metadata, rank
        )
        self._single_pass_gatherers = {
            k: _SinglePassGatherer.init_new(server_args, expert_location_metadata, rank)
            for k in self._accumulator.get_single_pass_gatherer_keys()
        }

        if server_args.enable_expert_distribution_metrics:
            logger.info(
                "ExpertDistributionRecorder auto start record since enable_expert_distribution_metrics"
            )
            self.start_record()

    def with_current_layer(self, layer_idx):
        return self._current_layer_idx.with_value(layer_idx)

    def with_debug_name(self, debug_name):
        return self._current_debug_name.with_value(debug_name)

    @contextmanager
    def with_forward_pass(self, forward_pass_id: int, forward_batch: ForwardBatch):
        with self._current_forward_pass_id.with_value(forward_pass_id):
            self._on_forward_pass_start(forward_batch)
            try:
                yield
            finally:
                self._on_forward_pass_end(forward_pass_id)

    @contextmanager
    def disable_this_region(self):
        """Context manager to temporarily disable recording."""
        previous_disable_all = self._disable_all
        self._disable_all = True
        try:
            yield
        finally:
            self._disable_all = previous_disable_all

    def _on_forward_pass_start(self, forward_batch: ForwardBatch):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            gatherer.reset()
            gatherer.on_forward_pass_start(forward_batch)

    def _on_forward_pass_end(self, forward_pass_id: int):
        if not self._recording:
            return
        for gatherer_key, gatherer in self._single_pass_gatherers.items():
            single_pass_data = gatherer.collect()
            self._accumulator.append(forward_pass_id, gatherer_key, single_pass_data)

    def on_select_experts(self, topk_ids: torch.Tensor):
        self._on_hook("on_select_experts", topk_ids=topk_ids)

    def on_deepep_dispatch_normal(
        self,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        self._on_hook(
            "on_deepep_dispatch_normal",
            local_physical_count_of_layer=local_physical_count_of_layer,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

    def on_deepep_dispatch_low_latency(
        self, local_physical_count_of_layer: torch.Tensor
    ):
        self._on_hook(
            "on_deepep_dispatch_low_latency",
            local_physical_count_of_layer=local_physical_count_of_layer,
        )

    def _on_hook(self, hook_name: str, **kwargs):
        if self._disable_all:
            return
        if not (self._recording or torch.cuda.is_current_stream_capturing()):
            return
        gatherer = self._single_pass_gatherers[
            self._accumulator.get_single_pass_gatherer_key(
                self._current_debug_name.value
            )
        ]
        getattr(gatherer, hook_name)(layer_idx=self._current_layer_idx.value, **kwargs)

    def _reset(self):
        """Reset the expert distribution recorder."""
        logger.info("Resetting ExpertDistributionRecorder...")
        assert (
            self._current_layer_idx.value is None
        ), f"{self._current_layer_idx.value=}"
        for gatherer in self._single_pass_gatherers.values():
            gatherer.reset()
        self._accumulator.reset()

    def start_record(self):
        """Start recording the expert distribution."""
        if self._recording:
            logger.warning(
                "SGLang server is already recording expert ids. Did you forget to dump the expert ids recorded so far by sending requests to the `/stop_expert_distribution_record` and `/dump_expert_distribution_record` endpoints?"
            )
        self._reset()
        self._recording = True

    def stop_record(self):
        """Stop recording the expert distribution."""
        if not self._recording:
            logger.warning(
                "SGLang server has not been recording expert ids. Did you forget to start recording by sending request to the `/start_expert_distribution_record` endpoint?"
            )
        self._recording = False

    def dump_record(self, output_mode: _OutputMode = "file"):
        """Dump the expert distribution record and reset the recorder after dumping."""
        output = self._accumulator.dump(output_mode=output_mode)
        self._reset()
        return output
    
    def get_balancedness(self) -> float:
        """get balancedness of the expert distribution.
        Returns:
            float: returns the average balancedness value of the last 100 recorded steps.
        """
        return self._accumulator.get_balancedness()

    @property
    def recording(self):
        return self._recording


_global_expert_distribution_recorder: Optional[ExpertDistributionRecorder] = (
    _ExpertDistributionRecorderNoop()
)


def get_global_expert_distribution_recorder():
    return _global_expert_distribution_recorder


def set_global_expert_distribution_recorder(value):
    global _global_expert_distribution_recorder
    _global_expert_distribution_recorder = value


# --------------------------------------- SinglePassGatherer -----------------------------------------


class _SinglePassGatherer(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ) -> "_SinglePassGatherer":
        if server_args.expert_distribution_recorder_mode == "per_token":
            return _DetailSinglePassGatherer(
                server_args, expert_location_metadata, rank
            )

        if server_args.expert_distribution_recorder_mode == "stat_approx":
            if server_args.enable_deepep_moe and (server_args.deepep_mode == "normal"):
                return _DeepepNormalSinglePassGatherer(expert_location_metadata, rank)
            else:
                raise NotImplementedError

        if server_args.enable_deepep_moe:
            if server_args.deepep_mode == "normal":
                return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)
            elif server_args.deepep_mode == "low_latency":
                return _DeepepLowLatencySinglePassGatherer(
                    expert_location_metadata, rank
                )
            else:
                raise NotImplementedError

        return _SelectExpertsSinglePassGatherer(expert_location_metadata, rank)

    def __init__(self, expert_location_metadata: "ExpertLocationMetadata", rank: int):
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        pass

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        pass

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        pass

    def reset(self):
        raise NotImplementedError

    def collect(self) -> Dict:
        raise NotImplementedError


class _DetailSinglePassGatherer(_SinglePassGatherer):
    # DeepSeek V3 has this value; should generalize later
    _TOP_K_NUM = 8

    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        super().__init__(expert_location_metadata, rank)
        self._metadata: Optional[Dict[str, Any]] = None
        self._topk_ids_of_layer = torch.zeros(
            (
                expert_location_metadata.num_layers,
                # TODO determine the max number
                server_args.chunked_prefill_size * 8,
                self._TOP_K_NUM,
            ),
            dtype=torch.int32,
            device=server_args.device,
        )
        self._misc_objects: List[Dict[str, Any]] = []
        assert (
            not server_args.enable_two_batch_overlap
        ), "DetailSinglePassGatherer does not support TBO yet"
        # TODO assert shared experts fusion is disabled, o/w data is wrong

    def on_forward_pass_start(self, forward_batch: ForwardBatch):
        assert self._metadata is None
        self._metadata = dict(
            # TODO pr-chain
            # rids=forward_batch.rids,
            input_ids=forward_batch.input_ids.cpu().tolist(),
            positions=forward_batch.positions.cpu().tolist(),
            extend_seq_lens=forward_batch.extend_seq_lens_cpu,
            forward_mode=forward_batch.forward_mode.value,
        )

    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        self._topk_ids_of_layer[layer_idx, : topk_ids.shape[0], : topk_ids.shape[1]] = (
            topk_ids
        )

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        self._misc_objects.append(
            dict(
                layer_id=layer_idx,
                num_tokens_per_rank=num_tokens_per_rank.cpu().tolist(),
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank.cpu().tolist(),
                num_tokens_per_expert=num_tokens_per_expert.cpu().tolist(),
            )
        )

    def reset(self):
        self._topk_ids_of_layer[...] = -1
        self._misc_objects.clear()
        self._metadata = None

    def collect(self) -> Dict:
        num_tokens = len(self._metadata["input_ids"])
        return dict(
            **self._metadata,
            topk_ids_of_layer=self._topk_ids_of_layer[:, :num_tokens, :].clone().cpu(),
            misc_objects=self._misc_objects,
        )


class _LayerBasedCpuSinglePassGatherer(_SinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._objects_of_layer = {}

    def _on_layer_data(self, layer_idx: int, objects: List[int]):
        assert 0 <= layer_idx < self._expert_location_metadata.num_layers
        if layer_idx in self._objects_of_layer:
            self._objects_of_layer[layer_idx] = _list_sum(
                self._objects_of_layer[layer_idx], objects
            )
        else:
            self._objects_of_layer[layer_idx] = objects

    def reset(self):
        self._objects_of_layer.clear()

    def _collect_objects(self, pad_len: int) -> torch.Tensor:
        data = [
            self._objects_of_layer.get(layer_index) or ([0] * pad_len)
            for layer_index in range(self._expert_location_metadata.num_layers)
        ]
        return torch.tensor(data)


def _list_sum(a: List, b: List) -> List:
    return [x + y for x, y in zip(a, b, strict=True)]


class _LayerBasedGpuSinglePassGatherer(_SinglePassGatherer):
    def __init__(self, *args, enable_global_physical_experts: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_global_physical_experts = enable_global_physical_experts
        self._data = torch.zeros(
            (
                self._expert_location_metadata.num_layers,
                (
                    self._expert_location_metadata.num_physical_experts
                    if enable_global_physical_experts
                    else self._expert_location_metadata.num_local_physical_experts
                ),
            ),
            dtype=torch.int,
            device="cuda",
        )

    def reset(self):
        self._data[...] = 0

    def collect(self) -> Dict:
        if self._enable_global_physical_experts:
            global_physical_count = self._data
        else:
            # Can optimize if bottleneck
            global_physical_count = _convert_local_to_global_physical_count(
                self._data,
                rank=self._rank,
                num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
                num_physical_experts=self._expert_location_metadata.num_physical_experts,
            )

        return dict(global_physical_count=global_physical_count)


class _SelectExpertsSinglePassGatherer(_LayerBasedGpuSinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, enable_global_physical_experts=True)

    # can optimize (e.g. fuse / compile)
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        topk_ids = topk_ids.flatten()
        mask = topk_ids != -1
        self._data[layer_idx, :].scatter_add_(
            dim=0, index=topk_ids.masked_fill(~mask, 0).long(), src=mask.int()
        )


class _DeepepNormalSinglePassGatherer(_LayerBasedCpuSinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.distributed.get_rank() == 0:
            logger.info(
                "DeepepNormalSinglePassGatherer gathers approximate statistics. "
                "If used with small batch size, consider using expert_distribution_recorder_mode=stat."
            )

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        assert isinstance(local_physical_count_of_layer, list)
        self._on_layer_data(layer_idx, local_physical_count_of_layer)

    def collect(self) -> Dict:
        local_physical_count = super()._collect_objects(
            pad_len=self._expert_location_metadata.num_local_physical_experts
        )
        global_physical_count = _convert_local_to_global_physical_count(
            local_physical_count,
            rank=self._rank,
            num_local_physical_experts=self._expert_location_metadata.num_local_physical_experts,
            num_physical_experts=self._expert_location_metadata.num_physical_experts,
        )
        return dict(global_physical_count=global_physical_count)


class _DeepepLowLatencySinglePassGatherer(_LayerBasedGpuSinglePassGatherer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, enable_global_physical_experts=False)

    def on_deepep_dispatch_low_latency(
        self, layer_idx: int, local_physical_count_of_layer: torch.Tensor
    ):
        # Most naive implementation, can optimize later
        self._data[layer_idx, :] += local_physical_count_of_layer


def _convert_local_to_global_physical_count(
    local_physical_count: torch.Tensor,
    rank: int,
    num_local_physical_experts: int,
    num_physical_experts: int,
) -> torch.Tensor:
    dtype = local_physical_count.dtype
    device = local_physical_count.device
    num_layers, _ = local_physical_count.shape

    ans = torch.zeros((num_layers, num_physical_experts), dtype=dtype, device=device)
    ans[
        :, num_local_physical_experts * rank : num_local_physical_experts * (rank + 1)
    ] = local_physical_count
    return ans


# --------------------------------------- Accumulator -----------------------------------------

_SINGLE_PASS_GATHERER_KEY_PRIMARY = "primary"


class _Accumulator(ABC):
    @staticmethod
    def init_new(
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ) -> "_Accumulator":
        return _Accumulator.get_class(server_args)(
            server_args, expert_location_metadata, rank
        )

    @staticmethod
    def get_class(server_args: ServerArgs) -> Type["_Accumulator"]:
        return {
            "stat": _StatAccumulator,
            "stat_approx": _StatAccumulator,
            "per_pass": _DetailAccumulator,
            "per_token": _DetailAccumulator,
            "historical": _SlidingWindowAccumulator,
            "historical_stat": _SlidingWindowAccumulator,
            "historical_dynamic": _SlidingWindowAccumulator,
        }[server_args.expert_distribution_recorder_mode]

    def __init__(
        self,
        server_args: ServerArgs,
        expert_location_metadata: "ExpertLocationMetadata",
        rank: int,
    ):
        self._server_args = server_args
        self._expert_location_metadata = expert_location_metadata
        self._rank = rank

    def get_single_pass_gatherer_keys(self):
        return [_SINGLE_PASS_GATHERER_KEY_PRIMARY]

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        return _SINGLE_PASS_GATHERER_KEY_PRIMARY

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        pass

    def reset(self):
        pass

    def dump(self, output_mode: _OutputMode):
        pass

    def get_balancedness(self) -> float:
        return 0.0

class _UtilizationRateAccumulatorMixin(_Accumulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._enable = self._server_args.enable_expert_distribution_metrics

        if self._enable:
            window_sizes = [10, 100, 1000]
            self._history = _DequeCollection(maxlens=window_sizes)
            self._rank = torch.distributed.get_rank()

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)
        if self._enable:
            self._append_utilization_rate(
                forward_pass_id, single_pass_data["global_physical_count"]
            )

    def reset(self):
        super().reset()
        if self._enable and self._server_args.expert_distribution_recorder_mode != "historical_dynamic":
            self._history.clear()

    def get_balancedness(self) -> float:
        """Get the average balancedness of the last 100 recorded steps.
        
        Returns:
            float: The average balancedness value. Returns 0.0 if expert distribution 
            metrics are not enabled or if there's no data available.
        """
        if not self._enable:
            return 0.0
        if self._rank == 0:
            mean_values = self._history.mean()
            value = mean_values.get(100, 0.0)  # Return 0.0 if no data for window size 10
        else:
            value = 0.0
        value_list = [value]
        torch.distributed.broadcast_object_list(value_list, src=0)
        return value_list[0]

    def _append_utilization_rate(
        self, forward_pass_id: int, single_pass_global_physical_count: torch.Tensor
    ):
        gpu_physical_count = compute_gpu_physical_count(
            single_pass_global_physical_count,
            num_gpu=self._expert_location_metadata.ep_size,
        )
        gpu_physical_count = gpu_physical_count.to(self._server_args.device)
        torch.distributed.reduce(
            gpu_physical_count, dst=0, op=torch.distributed.ReduceOp.SUM
        )

        if self._rank == 0:
            utilization_rate_tensor = compute_utilization_rate(gpu_physical_count)
            utilization_rate = torch.mean(utilization_rate_tensor).item()
            self._history.append(utilization_rate)

            gpu_physical_count_sum = gpu_physical_count.sum().item()

            logger.info(
                f"[Expert Balancedness] "
                f"forward_pass_id={forward_pass_id} "
                f"current_pass_balancedness={utilization_rate:.03f} "
                f"{''.join(f'last_{size}_average_balancedness={value:.03f} ' for size, value in self._history.mean().items())} "
                f"gpu_physical_count_sum={gpu_physical_count_sum}"
                # f"current_pass_per_layer={[round(x, 2) for x in utilization_rate_tensor.cpu().tolist()]}"
            )


class _DequeCollection:
    def __init__(self, maxlens: List[int]):
        self._dequeues = [deque(maxlen=maxlen) for maxlen in maxlens]

    def append(self, value):
        for d in self._dequeues:
            d.append(value)

    def clear(self):
        for d in self._dequeues:
            d.clear()

    def mean(self) -> Dict[int, float]:
        return {d.maxlen: sum(d) / len(d) for d in self._dequeues}


class _DetailAccumulator(_UtilizationRateAccumulatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._records = []

    def get_single_pass_gatherer_keys(self):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return [_SINGLE_PASS_GATHERER_KEY_PRIMARY, "child_a", "child_b"]
        return super().get_single_pass_gatherer_keys()

    def get_single_pass_gatherer_key(self, debug_name: Optional[str]):
        if False:  # TODO `server_args.enable_two_batch_overlap`
            return debug_name or _SINGLE_PASS_GATHERER_KEY_PRIMARY
        return super().get_single_pass_gatherer_key(debug_name)

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)

        def _process_object(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().clone()
            return obj

        single_pass_data_processed = {
            k: _process_object(v) for k, v in single_pass_data.items()
        }

        self._records.append(
            dict(
                forward_pass_id=forward_pass_id,
                rank=self._rank,
                gatherer_key=gatherer_key,
                **single_pass_data_processed,
            )
        )

    def reset(self):
        super().reset()
        self._records.clear()

    def dump(self, output_mode: _OutputMode):
        assert output_mode == "file"
        output = dict(
            records=self._records,
            # NOTE: This may change during recording, so here we say it is the "last" one
            last_physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )
        _dump_to_file(
            f"expert_distribution_recorder_{time.time()}_{self._rank}.pt", output
        )


class _StatAccumulator(_UtilizationRateAccumulatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_physical_count_of_buffered_step = _Buffer.init_new(
            item_shape=(
                self._expert_location_metadata.num_layers,
                # Cannot use local_physical_count to support select_experts
                self._expert_location_metadata.num_physical_experts,
            ),
            buffer_size=self._server_args.expert_distribution_recorder_buffer_size,
            dtype=torch.int32,
            device=self._server_args.device,
        )
        self._first_dump = True

    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)
        # Can optimize if overhead here is large
        self._global_physical_count_of_buffered_step.append(
            single_pass_data["global_physical_count"]
        )

    def reset(self):
        super().reset()
        self._global_physical_count_of_buffered_step.reset()

    def dump(self, output_mode: _OutputMode):
        logical_count_of_buffered_step = _convert_global_physical_count_to_logical_count(
            self._global_physical_count_of_buffered_step.get_all(),
            num_layers=self._expert_location_metadata.num_layers,
            num_logical_experts=self._expert_location_metadata.num_logical_experts,
            physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
        )

        if self._first_dump:
            self._first_dump = False
            torch.cuda.empty_cache()

        torch.distributed.all_reduce(
            logical_count_of_buffered_step, op=torch.distributed.ReduceOp.SUM
        )

        output = dict(
            rank=self._rank,
            logical_count=logical_count_of_buffered_step,
        )

        if output_mode == "file":
            if self._rank == 0:
                _dump_to_file(f"expert_distribution_recorder_{time.time()}.pt", output)
        elif output_mode == "object":
            return output
        else:
            raise NotImplementedError

class _SlidingWindowAccumulator(_UtilizationRateAccumulatorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._window_size = self._server_args.eplb_window_size
        self._priority_window_size = self._server_args.eplb_priority_window_size
        self._current_window_size = self._server_args.eplb_rebalance_num_iterations
        # 根据recorder mode选择不同的window size
        if self._server_args.expert_distribution_recorder_mode == "historical":
            buffer_window_size = self._window_size
        elif self._server_args.expert_distribution_recorder_mode in ["historical_stat", "historical_dynamic"]:
            buffer_window_size = self._current_window_size
        else:
            # 默认使用window_size
            buffer_window_size = self._window_size
        self._global_physical_count_of_buffered_step = _Buffer.init_new_historical(
            item_shape=(
                self._expert_location_metadata.num_layers,
                # Cannot use local_physical_count to support select_experts
                self._expert_location_metadata.num_physical_experts,
            ),
            window_size=buffer_window_size,
            dtype=torch.int32,
            device=self._server_args.device,
        )
    def append(
        self,
        forward_pass_id: int,
        gatherer_key: str,
        single_pass_data: Dict,
    ):
        super().append(forward_pass_id, gatherer_key, single_pass_data)
        # Can optimize if overhead here is large
        self._global_physical_count_of_buffered_step.append(
            single_pass_data["global_physical_count"]
        )
    def reset(self):
        super().reset()
        self._global_physical_count_of_buffered_step.reset()
    def dump(self, output_mode: _OutputMode):
        # 根据recorder mode选择不同的转换函数
        if self._server_args.expert_distribution_recorder_mode in ["historical_stat", "historical_dynamic"]:
            if self._server_args.eplb_window_decay_mode == "exp":
                cur_decay_mode = 0
            elif self._server_args.eplb_window_decay_mode == "linear":
                cur_decay_mode = 1
            elif self._server_args.eplb_window_decay_mode == "none":
                cur_decay_mode = 2
            else:
                cur_decay_mode = 2
            logical_count_of_buffered_step = _convert_global_physical_count_to_logical_count_historical_stat(
                self._global_physical_count_of_buffered_step.get_all(),
                num_layers=self._expert_location_metadata.num_layers,
                num_logical_experts=self._expert_location_metadata.num_logical_experts,
                decay_mode=cur_decay_mode,
                physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
            )
        else:
            # 默认使用historical
            logical_count_of_buffered_step = _convert_global_physical_count_to_logical_count_historical(
                self._global_physical_count_of_buffered_step.get_all(),
                num_layers=self._expert_location_metadata.num_layers,
                num_logical_experts=self._expert_location_metadata.num_logical_experts,
                priority_window_size=self._priority_window_size,
                current_window_size=self._current_window_size,
                total_window_size=self._window_size,
                middle_window_weight= self._server_args.eplb_middle_window_weight,
                last_window_weight= self._server_args.eplb_last_window_weight,
                physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
            )
        torch.distributed.all_reduce(
            logical_count_of_buffered_step, op=torch.distributed.ReduceOp.SUM
        )
        output = dict(
            rank=self._rank,
            logical_count=logical_count_of_buffered_step,
        )
        if output_mode == "file":
            if self._rank == 0:
                _dump_to_file(f"expert_distribution_recorder_{time.time()}.pt", output)
        elif output_mode == "object":
            return output
        else:
            raise NotImplementedError

def _dump_to_file(name, data):
    save_dir = Path(os.environ.get("SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR", "/tmp"))
    path_output = save_dir / name
    logger.info(f"Write expert distribution to {path_output}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(path_output))


class _Buffer:
    @staticmethod
    def init_new(item_shape: Tuple, buffer_size: int, dtype, device):
        if buffer_size < 0:
            return _InfiniteBuffer(item_shape, dtype=dtype, device=device)
        else:
            return _CircularBuffer(item_shape, buffer_size, dtype=dtype, device=device)
        
    @staticmethod
    def init_new_historical(item_shape: Tuple, window_size: int, dtype, device):
        if window_size < 0:
            return _InfiniteBuffer(item_shape, dtype=dtype, device=device)
        else:
            return _DequeBuffer(item_shape, window_size, dtype=dtype, device=device)

    def append(self, value: torch.Tensor):
        raise NotImplementedError

    def get_all(self) -> torch.Tensor:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class _CircularBuffer(_Buffer):
    def __init__(self, item_shape: Tuple, buffer_size: int, dtype, device):
        self._buffer = torch.zeros(
            (buffer_size, *item_shape), dtype=dtype, device=device
        )
        self._curr_index = 0

    def append(self, value: torch.Tensor):
        self._buffer[self._curr_index] = value
        self._curr_index = (self._curr_index + 1) % len(self._buffer)

    def get_all(self) -> torch.Tensor:
        return self._buffer

    def reset(self):
        self._buffer[...] = 0


class _InfiniteBuffer(_Buffer):
    def __init__(self, item_shape: Tuple, dtype, device):
        self._item_shape = item_shape
        self._buffer = torch.zeros((128, *item_shape), dtype=dtype, device=device)
        self._size = 0

    def append(self, value: torch.Tensor):
        curr_buffer_size = len(self._buffer)
        dtype = self._buffer.dtype
        device = self._buffer.device

        if self._size == curr_buffer_size:
            new_buffer = torch.zeros(
                (2 * curr_buffer_size, *self._item_shape), dtype=dtype, device=device
            )
            new_buffer[:curr_buffer_size] = self._buffer
            self._buffer = new_buffer

        self._buffer[self._size] = value
        self._size += 1

    def get_all(self) -> torch.Tensor:
        return self._buffer[: self._size]

    def reset(self):
        self._buffer[...] = 0
        self._size = 0

class _DequeBuffer(_Buffer):
    def __init__(self, item_shape: Tuple, buffer_size: int, dtype, device):
        # 使用deque作为基础数据结构,实现FIFO
        self._buffer_size = buffer_size
        self._device = device
        self._dtype = dtype
        self._item_shape = item_shape
        # 初始化一个空的deque
        self._deque = deque(maxlen=buffer_size)
    def append(self, value: torch.Tensor):
        # 如果deque达到最大长度,会自动从左侧(最早的数据)移除
        self._deque.append(value.clone())
    def get_all(self) -> torch.Tensor:
        if not self._deque: # 如果deque为空
            return torch.zeros((self._buffer_size, *self._item_shape), 
                            dtype=self._dtype, device=self._device)
        
        # 将deque中的所有tensor堆叠成一个tensor
        tensor_list = list(self._deque)
        actual_size = len(tensor_list)
        
        # 如果deque还未填满,需要填充零tensor
        result = torch.zeros((self._buffer_size, *self._item_shape), 
                           dtype=self._dtype, device=self._device)
        result[-actual_size:] = torch.stack(tensor_list)
        return result
    def reset(self):
        # self._deque.clear() # 清空deque
        pass

def _convert_global_physical_count_to_logical_count(
    # (whatever, num_layers, num_physical_experts)
    global_physical_count: torch.Tensor,
    num_layers: int,
    num_logical_experts: int,
    physical_to_logical_map: torch.Tensor,
):
    dim_extra, _, _ = global_physical_count.shape
    dtype = global_physical_count.dtype
    device = global_physical_count.device
    logical_count = torch.zeros(
        (dim_extra, num_layers, num_logical_experts), dtype=dtype, device=device
    )
    logical_count.scatter_add_(
        dim=2,
        index=physical_to_logical_map.unsqueeze(0)
        .expand(dim_extra, -1, -1)
        .to(torch.int64),
        src=global_physical_count,
    )
    return logical_count

def _convert_global_physical_count_to_logical_count_historical(
    global_physical_count: torch.Tensor,
    num_layers: int,
    num_logical_experts: int,
    priority_window_size: int,
    current_window_size: int,
    total_window_size: int,
    middle_window_weight: float,
    last_window_weight: float,
    physical_to_logical_map: torch.Tensor,
):
    """Convert global physical count to logical count for historical data with weight.
    
    Args:
        global_physical_count: Input tensor with shape (window_size, num_layers, num_physical_experts)
        priority_window_size: Size of the highest priority window (most recent)
        current_window_size: Size of the current window (includes priority window)
        total_window_size: Total size of buffer window
        
    The function applies different weights to different time windows:
    - Recent (priority) window: highest weight (1.0)
    - Current window (excluding priority): medium weight (middle_window_weight)
    - Historical window: lowest weight (last_window_weight)
    """
    dim_extra, num_layer, num_physical_expert = global_physical_count.shape
    assert dim_extra == total_window_size, (
        f"Expected eplb_window_buffer_size to be same as {total_window_size}, but got {dim_extra}"
    )
    dtype = global_physical_count.dtype
    device = global_physical_count.device
    
    # Create weight tensor
    weights = torch.ones(dim_extra, dtype=dtype, device=device) * last_window_weight  # last weight for historical data
    if current_window_size > 0:
        weights[-current_window_size:] = middle_window_weight  # medium weight for current window
    if priority_window_size > 0:
        weights[-priority_window_size:] = 1.0  # highest weight for priority window
        
    # Apply weights to input tensor
    weighted_physical_count = global_physical_count * weights.view(-1, 1, 1)
    
    # Convert to logical experts count with weights
    # Ensure weighted_physical_count has the correct dtype
    weighted_physical_count = weighted_physical_count.to(dtype)
    
    logical_count_historical = torch.zeros(
        (dim_extra, num_layers, num_logical_experts), dtype=dtype, device=device
    )
    logical_count_historical.scatter_add_(
        dim=2,
        index=physical_to_logical_map.unsqueeze(0)
        .expand(dim_extra, -1, -1)
        .to(torch.int64),
        src=weighted_physical_count,
    )
    
    return logical_count_historical

def _convert_global_physical_count_to_logical_count_historical_stat(
    global_physical_count: torch.Tensor,
    num_layers: int,
    num_logical_experts: int,
    decay_mode: int,
    physical_to_logical_map: torch.Tensor,
):
    """Convert global physical count to logical count for historical data with weight.
    
    Args:
        global_physical_count: Input tensor with shape (window_size, num_layers, num_physical_experts)
        current_window_size: Size of the current window
        decay_mode: Weight decay mode
            - 0: Reverse exponential decay (newest data has weight 1.0)
            - 1: Linear decay (weights decrease linearly from newest to oldest)
            - 2: No decay (all weights are 1.0)
    """
    dim_extra, num_layer, num_physical_expert = global_physical_count.shape
    dtype = global_physical_count.dtype
    device = global_physical_count.device
    
    # 设置衰减参数
    alpha = 0.98  # 指数衰减因子
    
    if decay_mode == 0:
        # 反向指数衰减：最新的数据权重为1，往前依次为alpha^1, alpha^2, ...
        weights = torch.tensor(
            [alpha ** i for i in range(dim_extra)][::-1],
            dtype=torch.float32,
            device=device
        )
    elif decay_mode == 1:
        # 线性衰减：权重从1线性递减到0
        weights = torch.linspace(0, 1, dim_extra, dtype=torch.float32, device=device)
    elif decay_mode == 2:
        # 无衰减：所有权重都是1
        weights = torch.ones(dim_extra, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unsupported decay mode: {decay_mode}")
    
    # 归一化权重
    weights = weights / weights.max()  # 确保最新的数据权重为1
    
    # 将权重转换为与输入tensor相同的dtype
    weights = weights.to(dtype)
    
    # 应用权重到输入张量
    weighted_physical_count = global_physical_count * weights.view(-1, 1, 1)
    
    # 转换为逻辑专家计数
    logical_count_historical = torch.zeros(
        (dim_extra, num_layers, num_logical_experts), 
        dtype=dtype, 
        device=device
    )
    
    logical_count_historical.scatter_add_(
        dim=2,
        index=physical_to_logical_map.unsqueeze(0)
        .expand(dim_extra, -1, -1)
        .to(torch.int64),
        src=weighted_physical_count,
    )
    
    return logical_count_historical

def compute_gpu_physical_count(
    physical_count_of_whatever: torch.Tensor,  # (..., num_layer, num_physical_expert)
    num_gpu: int,
):
    """output: gpu_physical_count_of_batch (..., num_layer, num_gpu)"""
    return einops.reduce(
        physical_count_of_whatever,
        "... num_layer (num_gpu num_expert_per_gpu) -> ... num_layer num_gpu",
        "sum",
        num_gpu=num_gpu,
    )


def compute_utilization_rate(
    gpu_physical_count_of_batch: torch.Tensor,  # (..., num_layer, num_gpu)
):
    """output: utilization_rate (..., num_layer)"""
    gpu_physical_count_of_batch = gpu_physical_count_of_batch.float()
    max_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "max",
    )
    avg_gpu_physical_count = einops.reduce(
        gpu_physical_count_of_batch,
        "... num_layer num_gpu -> ... num_layer",
        "mean",
    )
    return (avg_gpu_physical_count + 1e-5) / (max_gpu_physical_count + 1e-5)
