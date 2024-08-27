from typing import Optional, TypeAlias, cast

import torch

from ..topology import Topology

_MAX_DATA_DIM = 8  # maximum dimension of tensors that can be synced

NumbersOfElements: TypeAlias = list[int]
TensorSize: TypeAlias = list[int]
TensorSizes: TypeAlias = list[TensorSize]


def sync_sizes(tensors: list[Optional[torch.Tensor]], topology: Topology) -> tuple[TensorSizes, NumbersOfElements]:
    """
    Build the size tensor on rank 0 and broadcast it to model_parallel_group, otherwise it receives the tensor
    Returns a tuple of lists.
    The first element is a list of the sizes of the tensors.
    The second element contains a list of the number of elements (product of the sizes) of each tensor
    :param tensors: the list of tensors for which the sizes should be synced
    :param topology: topology config used to sync
    :return: the synced sizes via the different ranks in the group
    """

    tensor_sizes = _build_tensor_sizes(tensors, topology.model_parallel_rank)

    # Move to GPU and broadcast.
    sizes_cuda = torch.LongTensor(tensor_sizes).cuda()
    torch.distributed.broadcast(
        sizes_cuda,
        torch.distributed.distributed_c10d.get_global_rank(topology.model_parallel_group, 0),
        group=topology.model_parallel_group,
    )
    sizes_cpu = sizes_cuda.cpu().tolist()
    return _unpack_sizes(sizes_cpu)


def _unpack_sizes(tensor_sizes: list[int]) -> tuple[TensorSizes, NumbersOfElements]:
    """
    Unpacks the result after broadcasting, which means unflatten the tensor sizes
    :param tensor_sizes: tensor sizes in a flat representation
    :return: unflattened version of the tensor sizes, combined with the number of elements, that can be stored in a
    tensor of the given size
    """
    max_dim = _MAX_DATA_DIM
    sizes: TensorSizes = []
    number_of_element_list = []
    tensor_len = len(tensor_sizes) // max_dim
    for tensor_position in range(tensor_len):
        size = list()
        number_of_elements = 1
        for dimension_position in range(max_dim):
            this_size = tensor_sizes[tensor_position * max_dim + dimension_position]
            if this_size <= 0:
                break
            size.append(this_size)
            number_of_elements *= this_size
        sizes.append(size)
        number_of_element_list.append(number_of_elements)
    return sizes, number_of_element_list


def _build_tensor_sizes(tensors: list[Optional[torch.Tensor]], model_parallel_rank: int) -> list[int]:
    """
    Builds one list containing the sizes of the given lists of tensors
    e.g. tensors=[a,b,c] where a.size()=(2,3,9), b.size()=(5), c.size()=(7,8)
    it will return [2,3,9,-1,...,5,-1,...,7,8,-1...]
    with -1 filled up to length of _MAX_DATA_DIM
    dimension size can not exceed _MAX_DATA_DIM
    :param tensors: list of tensors
    :param model_parallel_rank:  only happens if model_parallel_rank is 0
    :return: list of sizes
    """
    max_dim = _MAX_DATA_DIM
    # build the list which will contain tensor_sizes of size max_dim*len(tensors)
    tensor_sizes = [-1] * max_dim * len(tensors)
    if model_parallel_rank != 0:
        return tensor_sizes
    # Pack the sizes on rank zero.
    tensors_mp_0 = cast(list[torch.Tensor], tensors)  # mypy type casting
    offset = 0
    for tensor in tensors_mp_0:
        assert tensor.dim() <= max_dim, "you should increase MAX_DATA_DIM"
        tensor_size = tensor.size()
        for i, s in enumerate(tensor_size):
            assert s > 0, "cannot communicate tensor of size 0"
            tensor_sizes[i + offset] = s
        offset += max_dim
    return tensor_sizes


def _check_data_types(tensors: list[torch.Tensor], dtype: torch.dtype) -> None:
    """Check that all the keys have the same target data type.
    :param tensors: list of tensors to be checked
    :param dtype: type which all tensors should have
    """
    for tensor in tensors:
        assert (
            tensor.dtype == dtype
        ), f"broadcast_data requires a list of tensors of the same dtype; expected {dtype} got {tensor.dtype}"


def broadcast_data(tensors: list[Optional[torch.Tensor]], dtype: torch.dtype, topology: Topology) -> list[torch.Tensor]:
    """
    Broadcast data from rank zero of each model parallel group to the members of the same model parallel group.
    :param tensors: list of tensors to broadcast
    :param dtype: the data type of the tensors. This needs to be similar in all tensors
    :param topology: topology to be used
    :return: list of tensors after broadcasting
    """
    # Build the sizes and number of elements list
    sizes, numels_per_tensor = sync_sizes(tensors=tensors, topology=topology)

    # Pack on rank zero.
    flatten_data = _flatten(tensors, dtype, numels_per_tensor, topology.model_parallel_rank)

    if orig_dtype_is_bool := flatten_data.dtype == torch.bool:
        flatten_data = flatten_data.to(torch.int8)
    # Broadcast
    torch.distributed.broadcast(
        flatten_data,
        torch.distributed.distributed_c10d.get_global_rank(topology.model_parallel_group, 0),
        group=topology.model_parallel_group,
    )
    if orig_dtype_is_bool:
        flatten_data = flatten_data.to(torch.bool)

    # Unpack
    output = list()
    offset = 0
    for size, numel in zip(sizes, numels_per_tensor):
        output.append(flatten_data.narrow(0, offset, numel).view(size))
        offset += numel

    return output


def _flatten(
    tensors: list[Optional[torch.Tensor]],
    dtype: torch.dtype,
    numels_per_tensor: NumbersOfElements,
    model_parallel_rank: int,
) -> torch.Tensor:
    """
    Flatten the tensors if all have the required dtype and the model parallel rank is 0.
    If the model parallel rank is not 0 it will return an empty tensor of size sum of numels_per_tensor,
    which is used to receive the flattened tensors from rank 0
    :param tensors: tensors to flatten
    :param dtype: dtype of the tensors should be the same over the whole list
    :param numels_per_tensor: number of elements for each tensor
    :param model_parallel_rank: current rank of the model parallel
    :return: a flattened version of the tensors
    """
    if model_parallel_rank != 0:
        return torch.empty(
            size=[sum(numels_per_tensor)],
            device=torch.device("cuda", torch.cuda.current_device()),
            dtype=dtype,
        )
    assert all(tensor is not None for tensor in tensors)
    tensors_mp_0 = cast(list[torch.Tensor], tensors)
    # Check that all keys have the same data type.
    _check_data_types(tensors=tensors_mp_0, dtype=dtype)
    # Flatten the data associated with the keys
    return torch.cat([tensor.contiguous().view(-1) for tensor in tensors_mp_0], dim=0).cuda()
