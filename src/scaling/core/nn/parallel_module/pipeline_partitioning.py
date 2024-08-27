# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from Microsoft Corporation
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

from bisect import bisect_left
from typing import List, NamedTuple, Optional, Sequence

import numpy as np

from .layer_spec import LayerSpec


class PipePartitionCoordinates(NamedTuple):
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def pipe_partition_from_indices(
    partition_array: Sequence[int], num_layers: Optional[int] = None
) -> List[PipePartitionCoordinates]:
    if num_layers is not None and partition_array[-1] != num_layers:
        raise ValueError(
            f"Last entry of partition_array needs to match num_layers; got"
            f"partition_array={partition_array}, num_layers={num_layers}."
        )
    return [
        PipePartitionCoordinates(start=start, end=end) for start, end in zip(partition_array[:-1], partition_array[1:])
    ]


def pipe_partition_uniform(item_count: int, partition_count: int) -> List[PipePartitionCoordinates]:
    """
    computes indices identifying `partition_count` uniform partitions in a list of length `item_count`
    """
    assert (
        item_count >= partition_count
    ), f"cannot partition {item_count} layers on {partition_count} pipe parallel stages"

    # compute length
    minimal_part_length = item_count // partition_count
    residual = item_count - (minimal_part_length * partition_count)

    # determine start indices with base length
    partition_array = np.arange(0, (partition_count + 1) * minimal_part_length, minimal_part_length)

    # spread the residual as evenly as possible
    for i in range(residual):
        partition_array[i + 1 :] += 1

    return pipe_partition_from_indices(partition_array.tolist())


def pipe_partition_balanced(
    layer_specs: List[LayerSpec], partition_count: int, eps: float = 1e-3
) -> List[PipePartitionCoordinates]:
    weights = _count_layer_params(layer_specs=layer_specs)
    weights_cumulated = np.cumsum(weights)

    # Find the smallest bottleneck (weight of heaviest partition)
    bottleneck = _rb_partition_balanced(weights_cumulated, partition_count, eps=eps)

    # Now compute that partitioning
    partition_array, success = _lprobe(weights_cumulated, partition_count, bottleneck)
    assert success

    return pipe_partition_from_indices(partition_array)


def _count_layer_params(layer_specs: List[LayerSpec]) -> list[int]:
    """Count the trainable parameters in individual layers.

    This routine will only build one layer at a time.

    Returns:
        A list of the number of parameters in each layer.
    """
    param_counts = [0] * len(layer_specs)
    for idx, layer_spec in enumerate(layer_specs):
        layer = layer_spec.initialize()
        params = filter(lambda p: p.requires_grad, layer.parameters())
        param_counts[idx] = sum(p.numel() for p in params)
    return param_counts


def _rb_partition_balanced(weights: np.ndarray, num_parts: int, eps: float) -> float:
    total_weight: int = weights[-1]
    lower: float = total_weight / num_parts  # best case heaviest partition
    upper: float = total_weight  # worst case heaviest partition

    # Do a binary search for the best partitioning
    while upper > lower + eps:
        mid = lower + ((upper - lower) / 2)
        parts, success = _lprobe(weights, num_parts, mid)
        if success:
            upper = mid
        else:
            lower = mid + eps
    return upper


def _lprobe(weights: np.ndarray, num_parts: int, bottleneck: float) -> tuple[list[int], bool]:
    num_items = len(weights)
    total_weight = weights[-1]

    # initialize partitioning
    parts = [0] * (num_parts + 1)
    for p in range(1, num_parts + 1):
        parts[p] = num_items

    bsum = bottleneck  # running sum of target weight for pth partition
    chunksize = num_items // num_parts
    step = chunksize
    for p in range(1, num_parts):
        # Jump to the next bucket
        while (step < num_items) and (weights[step] < bsum):
            step += chunksize

        # Find the end index of partition p
        parts[p] = bisect_left(weights, bsum, lo=step - chunksize, hi=min(step, num_items))
        # Nothing more to partition, return early
        if parts[p] == num_items:
            # See if the current partition is overweight.
            part_size = weights[-1] - weights[parts[p - 1]]
            return parts, part_size < bottleneck

        # Next partition target
        bsum = weights[parts[p] - 1] + bottleneck

    return parts, bsum >= total_weight
