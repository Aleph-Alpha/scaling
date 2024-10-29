# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from NVIDIA CORPORATION
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import torch
from torch.distributed.distributed_c10d import ProcessGroup

from scaling.core.topology import Topology

# -----------------
# Helper functions/classes for column and row parallel linear layers.
# -----------------


class _CopyToModelParallelRegion(torch.autograd.Function):
    """
    Pass the input to the model parallel region.
    """

    @staticmethod
    def symbolic(graph: Any, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor

    @staticmethod
    def forward(  # type: ignore[override] # noqa
        ctx: Any,
        input_tensor: torch.Tensor,
        model_parallel_size: int,
        model_parallel_rank: int,
        model_parallel_group: ProcessGroup,
    ) -> torch.Tensor:
        ctx.model_parallel_size = model_parallel_size
        ctx.model_parallel_rank = model_parallel_rank
        ctx.model_parallel_group = model_parallel_group
        return input_tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:  # type: ignore[override] # noqa
        output = _all_reduce(grad_output, ctx.model_parallel_size, ctx.model_parallel_rank, ctx.model_parallel_group)
        return output, None, None, None


class _AllConcat(torch.autograd.Function):
    """
    Gather the input from model parallel region and concatenate.
    """

    @staticmethod
    def symbolic(graph: Any, input_tensor: torch.Tensor) -> torch.Tensor:
        return _all_concat(input_tensor)  # type: ignore[call-arg]

    @staticmethod
    def forward(  # type: ignore[override] # noqa
        ctx: Any,
        input_tensor: torch.Tensor,
        dim: int,
        model_parallel_size: int,
        model_parallel_rank: int,
        model_parallel_group: ProcessGroup,
    ) -> torch.Tensor:
        ctx.model_parallel_size = model_parallel_size
        ctx.model_parallel_rank = model_parallel_rank
        ctx.model_parallel_group = model_parallel_group
        ctx.dim = dim
        return _all_concat(input_tensor, dim, model_parallel_size, model_parallel_rank, model_parallel_group)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:  # type: ignore[override] # noqa
        output = _all_shard(
            grad_output, ctx.dim, ctx.model_parallel_size, ctx.model_parallel_rank, ctx.model_parallel_group
        )
        return output, None, None, None, None


class _AllReduce(torch.autograd.Function):
    """
    All-reduce the input from the model parallel region.
    """

    @staticmethod
    def forward(  # type: ignore[override] # noqa
        ctx: Any,
        input_tensor: torch.Tensor,
        model_parallel_size: int,
        model_parallel_rank: int,
        model_parallel_group: ProcessGroup,
    ) -> torch.Tensor:
        return _all_reduce(input_tensor, model_parallel_size, model_parallel_rank, model_parallel_group)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:  # type: ignore[override] # noqa
        return grad_output, None, None, None


class _AllShard(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.
    """

    @staticmethod
    def forward(  # type: ignore[override] # noqa
        ctx: Any,
        input_tensor: torch.Tensor,
        dim: int,
        model_parallel_size: int,
        model_parallel_rank: int,
        model_parallel_group: ProcessGroup,
    ) -> torch.Tensor:
        ctx.model_parallel_size = model_parallel_size
        ctx.model_parallel_rank = model_parallel_rank
        ctx.model_parallel_group = model_parallel_group
        ctx.dim = dim
        return _all_shard(input_tensor, dim, model_parallel_size, model_parallel_rank, model_parallel_group)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:  # type: ignore[override] # noqa
        output = _all_concat(
            grad_output, ctx.dim, ctx.model_parallel_size, ctx.model_parallel_rank, ctx.model_parallel_group
        )
        return output, None, None, None, None


class _AllReduceScatterToSequenceParallel(torch.autograd.Function):
    """
    All-reduce the input from the model parallel region and scatter to sequence parallel.
    """

    @staticmethod
    def forward(  # type: ignore[override] # noqa
        ctx: Any,
        input_tensor: torch.Tensor,
        model_parallel_size: int,
        model_parallel_rank: int,
        model_parallel_group: ProcessGroup,
    ) -> torch.Tensor:
        ctx.model_parallel_size = model_parallel_size
        ctx.model_parallel_rank = model_parallel_rank
        ctx.model_parallel_group = model_parallel_group
        return _reduce_scatter_along_sequence_dim(
            input_tensor, model_parallel_size, model_parallel_rank, model_parallel_group
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:  # type: ignore[override] # noqa
        output = _gather_along_sequence_dim(
            grad_output, ctx.model_parallel_size, ctx.model_parallel_rank, ctx.model_parallel_group
        )
        return output, None, None, None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(  # type: ignore[override] # noqa
        ctx: Any,
        input_tensor: torch.Tensor,
        model_parallel_size: int,
        model_parallel_rank: int,
        model_parallel_group: ProcessGroup,
        tensor_parallel_output_grad: bool = True,
    ) -> torch.Tensor:
        ctx.model_parallel_size = model_parallel_size
        ctx.model_parallel_rank = model_parallel_rank
        ctx.model_parallel_group = model_parallel_group
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_sequence_dim(
            input_tensor, ctx.model_parallel_size, ctx.model_parallel_rank, ctx.model_parallel_group
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:  # type: ignore[override] # noqa
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad
        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            output = _reduce_scatter_along_sequence_dim(
                grad_output, ctx.model_parallel_size, ctx.model_parallel_rank, ctx.model_parallel_group
            )
        else:
            # Split by sequence length dim
            output = _reduce_scatter_along_sequence_dim(
                grad_output, ctx.model_parallel_size, ctx.model_parallel_rank, ctx.model_parallel_group
            )
        return output, None, None, None, None


def copy_to_tensor_model_parallel_region(input_tensor: torch.Tensor, topology: Topology) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(
        input_tensor.contiguous(),
        topology.config.model_parallel_size,
        topology.model_parallel_rank,
        topology.model_parallel_group,
    )


def all_concat(input_tensor: torch.Tensor, dim: int, topology: Topology) -> torch.Tensor:
    return _AllConcat.apply(
        input_tensor.contiguous(),
        dim,
        topology.config.model_parallel_size,
        topology.model_parallel_rank,
        topology.model_parallel_group,
    )


def all_reduce(input_tensor: torch.Tensor, topology: Topology) -> torch.Tensor:
    return _AllReduce.apply(
        input_tensor.contiguous(),
        topology.config.model_parallel_size,
        topology.model_parallel_rank,
        topology.model_parallel_group,
    )


def all_shard(input_tensor: torch.Tensor, dim: int, topology: Topology) -> torch.Tensor:
    return _AllShard.apply(
        input_tensor.contiguous(),
        dim,
        topology.config.model_parallel_size,
        topology.model_parallel_rank,
        topology.model_parallel_group,
    )


def all_reduce_scatter_to_sequence_parallel(input_tensor: torch.Tensor, topology: Topology) -> torch.Tensor:
    return _AllReduceScatterToSequenceParallel.apply(
        input_tensor.contiguous(),
        topology.config.model_parallel_size,
        topology.model_parallel_rank,
        topology.model_parallel_group,
    )


def gather_from_sequence_parallel_region(
    input_tensor: torch.Tensor, topology: Topology, tensor_parallel_output_grad: bool = True
) -> torch.Tensor:
    return _GatherFromSequenceParallelRegion.apply(
        input_tensor,
        topology.config.model_parallel_size,
        topology.model_parallel_rank,
        topology.model_parallel_group,
        tensor_parallel_output_grad,
    )


def _all_reduce(
    input_tensor: torch.Tensor, model_parallel_size: int, model_parallel_rank: int, model_parallel_group: ProcessGroup
) -> torch.Tensor:
    """
    All-reduce the input tensor across model parallel group.
    """

    # Bypass the function if we are using only 1 GPU.
    if model_parallel_size == 1:
        return input_tensor

    # All-reduce.
    torch.distributed.all_reduce(input_tensor, group=model_parallel_group)
    return input_tensor


def _reduce_scatter_along_sequence_dim(
    input_tensor: torch.Tensor, model_parallel_size: int, model_parallel_rank: int, model_parallel_group: ProcessGroup
) -> torch.Tensor:
    """Reduce-scatter the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if model_parallel_size == 1:
        return input_tensor

    # [batch_size, sequence_length, hidden_size]
    dim_size: list[int] = list(input_tensor.size())
    assert dim_size[1] % model_parallel_size == 0, "Sequence parallel size should be divisible by tensor parallel size"

    # Scatter along sequence dimension
    dim_size[1] = dim_size[1] // model_parallel_size

    output = torch.empty(dim_size, dtype=input_tensor.dtype, device=_get_current_cuda_device())
    torch.distributed.reduce_scatter_tensor(output, input_tensor.contiguous(), group=model_parallel_group)

    return output


def _gather_along_sequence_dim(
    input_tensor: torch.Tensor, model_parallel_size: int, model_parallel_rank: int, model_parallel_group: ProcessGroup
) -> torch.Tensor:
    """Gather tensors and concatenate along the first dimension."""

    # Bypass the function if we are using only 1 GPU.
    if model_parallel_size == 1:
        return input_tensor

    dim_size = list(input_tensor.size())
    dim_size[1] = dim_size[1] * model_parallel_size

    output_tensor = torch.empty(dim_size, dtype=input_tensor.dtype, device=_get_current_cuda_device())
    torch.distributed.all_gather_into_tensor(output_tensor, input_tensor.contiguous(), group=model_parallel_group)

    return output_tensor


def _all_shard(
    input_tensor: torch.Tensor,
    dim: int,
    model_parallel_size: int,
    model_parallel_rank: int,
    model_parallel_group: ProcessGroup,
) -> torch.Tensor:
    """
    Split the tensor along the passed dimension and keep the corresponding slice.
    """
    # Bypass the function if we are using only 1 GPU.
    if model_parallel_size == 1:
        return input_tensor

    # Split along last dimension.
    shard_dim_size = input_tensor.size()[dim] // model_parallel_size
    input_list = torch.split(input_tensor, shard_dim_size, dim=dim)

    # Note: torch.split does not create contiguous tensors by default.
    output_tensor = input_list[model_parallel_rank].contiguous()

    return output_tensor


def _all_concat(
    input_tensor: torch.Tensor,
    dim: int,
    model_parallel_size: int,
    model_parallel_rank: int,
    model_parallel_group: ProcessGroup,
) -> torch.Tensor:
    """
    Gather tensors and concatenate along the last dimension.
    """

    # Bypass the function if we are using only 1 GPU.
    if model_parallel_size == 1:
        return input_tensor

    input_tensor = input_tensor.contiguous()

    # Size and dimension.
    rank = model_parallel_rank

    tensor_list = [torch.empty_like(input_tensor) for _ in range(model_parallel_size)]
    tensor_list[rank] = input_tensor
    torch.distributed.all_gather(tensor_list, input_tensor, group=model_parallel_group)

    # Note: torch.cat already creates a contiguous tensor.
    output_tensor = torch.cat(tensor_list, dim=dim).contiguous()

    return output_tensor


def _get_current_cuda_device() -> torch.device:
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def get_device(
    topology: Topology | None = None,
    device: torch.device | None = None,
) -> torch.device:
    # Should a precedence rule be established instead?
    assert topology is None or device is None, "cannot specify both device and topology"

    # Default setting.
    result = torch.device(type="cuda", index=torch.cuda.current_device())

    if topology is not None:
        result = topology.device
    elif device is not None:
        result = device

    return result
