from typing import List

import torch
from torch.distributed.distributed_c10d import ProcessGroup


def allreduce_tensor_in_float32(tensor: torch.Tensor, process_group: ProcessGroup) -> None:
    if tensor.dtype in [torch.float16, torch.bfloat16]:
        tensor_to_allreduce = tensor.to(torch.float32)
        torch.distributed.all_reduce(tensor_to_allreduce, group=process_group)
        tensor.copy_(tensor_to_allreduce)
    else:
        torch.distributed.all_reduce(tensor, group=process_group)


def allreduce_bucket(
    bucket: List[torch.Tensor], data_parallel_group: ProcessGroup, data_parallel_size: int
) -> torch.Tensor:
    tensor = torch._C._nn.flatten_dense_tensors(bucket)
    tensor.div_(data_parallel_size)
    allreduce_tensor_in_float32(tensor=tensor, process_group=data_parallel_group)
    return tensor


def allreduce_and_copy(bucket: List[torch.Tensor], data_parallel_group: ProcessGroup, data_parallel_size: int) -> None:
    stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream):
        allreduced = allreduce_bucket(
            bucket=bucket,
            data_parallel_group=data_parallel_group,
            data_parallel_size=data_parallel_size,
        )
        for buf, synced in zip(
            bucket,
            torch._C._nn.unflatten_dense_tensors(allreduced, bucket),
        ):
            buf.copy_(synced)


def allreduce_no_retain(
    bucket: List[torch.Tensor],
    data_parallel_group: ProcessGroup,
    data_parallel_size: int,
    numel_per_bucket: int = 500000000,
) -> None:
    small_bucket = []
    numel = 0
    for tensor in bucket:
        small_bucket.append(tensor)
        numel = numel + tensor.numel()
        if numel > numel_per_bucket:
            allreduce_and_copy(
                bucket=small_bucket,
                data_parallel_group=data_parallel_group,
                data_parallel_size=data_parallel_size,
            )
            small_bucket = []
            numel = 0

    if len(small_bucket) > 0:
        allreduce_and_copy(
            bucket=small_bucket,
            data_parallel_group=data_parallel_group,
            data_parallel_size=data_parallel_size,
        )
