import torch

from scaling.core.nn.parameter_meta import CoreParameterMeta
from scaling.core.topology import Topology


def merge_parameter(
    parameter: torch.Tensor,
    core_parameter_meta: CoreParameterMeta,
    topology: Topology,
) -> torch.Tensor:
    # We pass cloned tensors to torch.save() to avoid checkpoint bloat which occurs because torch.save()
    # saves the underlying storage rather than the slice of the storage corresponding to individual tensors.
    # This is a problem when allocate tensors using slices of large flattened buffers.
    # Tensor cloning helps to avoid this problem because the storage of cloned tensors are closer to the true size.
    # It is expected that the garbage collector will reclaim the cloned tensor storage to avoid memory bloat.
    # See https://pytorch.org/docs/stable/notes/serialization.html#preserve-storage-sharing

    # casting to cpu requires a copy of tensor data and removes other potentially set parameters

    if topology.config.model_parallel_size == 1 or not core_parameter_meta.is_model_parallel:
        # just use the locally available tensor if there is no model parallelism or the parameter is a duplicate
        # saving just one duplicate fixes potential numeric divergence upon restart / load from a checkpoint
        return parameter.clone().cpu()

    # there is model parallel and a parameter split along model parallel ranks
    # a merge is necessary
    # Logic:
    #  - Move local parameter to cpu. this keeps the state available for later reset
    #  - Use the memory allocated for the parameters to broadcast parameter values from all mp > 0
    #  - Move received parameter values for mp > 0 to cpu and collect all
    #  - recover state of local parameter that has been overwritten due to the broadcast
    #  - merge the parameters collected on cpu and add to state dict

    local_parameter_value = parameter.clone().cpu()
    partitions = list()

    parameter = parameter.cuda()

    for sending_mp_rank in range(0, topology.config.model_parallel_size):
        # load parameter value from cpu again
        # after the first iteration the values would otherwise been changed
        parameter.data.copy_(local_parameter_value)
        torch.distributed.broadcast(
            parameter.data,
            topology.get_global_rank(model_parallel_rank=sending_mp_rank),
            group=topology.model_parallel_group,
        )

        partitions.append(parameter.clone().cpu())

    # recover parameter values on all ranks
    parameter.data.copy_(local_parameter_value)

    # merge the parameter list
    parameter_merged = torch.cat(
        partitions,
        dim=core_parameter_meta.model_parallel_dimension,
    )

    return parameter_merged


def split_parameter(
    parameter: torch.Tensor,
    core_parameter_meta: CoreParameterMeta,
    topology: Topology,
) -> torch.Tensor:
    """
    Slices tensor by the model parallel size and
    returns the split tensors at the current model parallel rank position.
    """

    model_parallel_size = topology.config.model_parallel_size
    model_parallel_dimension = core_parameter_meta.model_parallel_dimension
    assert model_parallel_dimension is not None
    assert (
        parameter.size(dim=model_parallel_dimension) % model_parallel_size == 0
    ), f"""cannot slice tensor for parallelization, the loaded {core_parameter_meta.layer_class_name}
    {core_parameter_meta.parameter_name} is of size ({parameter.size(dim=model_parallel_dimension)})
    in dimension {model_parallel_dimension},
    it needs to be divisible by the model parallel size ({model_parallel_size})"""

    tensor_size_per_partition = parameter.size(dim=model_parallel_dimension) // model_parallel_size
    tensor_start_index = topology.model_parallel_rank * tensor_size_per_partition
    tensor_end_index = tensor_start_index + tensor_size_per_partition  # subtract 1 for index_select slicing function
    # slice tensor for current mp rank
    split_tensor_for_mp_rank = torch.index_select(
        parameter,
        dim=model_parallel_dimension,
        index=(
            torch.arange(tensor_start_index, tensor_end_index)
            if tensor_start_index != tensor_end_index
            else torch.tensor([tensor_start_index])
        ),
    )
    return split_tensor_for_mp_rank.contiguous()
