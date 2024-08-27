from typing import Optional, Union

import pytest
import torch

from scaling.core import (
    ColumnParallelLinear,
    RowParallelLinear,
    Topology,
    TopologyConfig,
)
from scaling.core.runner.launch_config import LaunchConfig
from scaling.core.utils.port import find_free_port

from ..utils import dist_launcher


def run_test_parallel_linear(
    return_dict: dict,
    model_parallel_size: Optional[int],
    in_features: int,
    out_features: int,
    bias: bool,
    linear_layer_type: torch.autograd.Function,
    bitfit_bias_name: Optional[str],
):
    """
    function implementing the behavior of training for one single gpu / process
    """
    launch_config = LaunchConfig.from_launcher_args()
    topology = Topology(
        config=TopologyConfig(  # type: ignore[call-arg]
            global_rank=launch_config.global_rank,
            world_size=model_parallel_size,
            model_parallel_size=model_parallel_size,
            local_slot=launch_config.local_slot,
            pipe_parallel_size=1,
            global_batch_size=1,
            micro_batch_size=1,
        )
    )
    topology.initialize_distributed(
        master_addr=launch_config.master_addr,
        master_port=str(launch_config.master_port),
        torch_distributed_timeout_minutes=2,
    )

    merged_weights = torch.zeros((out_features, in_features), dtype=torch.float32).cuda()
    merged_bias = torch.zeros((out_features), dtype=torch.float32).cuda()

    # Test ColumnParallelLinear case
    if linear_layer_type == ColumnParallelLinear:
        parallel_linear: Union[ColumnParallelLinear, RowParallelLinear] = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            topology=topology,
            bitfit_bias_name=bitfit_bias_name,
        )

        # retrieve initialized weights of parallel linear layer
        if model_parallel_size == 1:
            merged_weights = parallel_linear.weight
        else:
            merged_weights[
                topology.model_parallel_rank * (out_features // 2) : (topology.model_parallel_rank + 1)
                * (out_features // 2),
                :,
            ].copy_(parallel_linear.weight)
        if model_parallel_size and bias and model_parallel_size > 1:
            assert parallel_linear.bias is not None
            merged_bias[
                topology.model_parallel_rank * (out_features // 2) : (topology.model_parallel_rank + 1)
                * (out_features // 2)
            ].copy_(parallel_linear.bias)
    else:  # Test RowParallelLinear case
        parallel_linear = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            topology=topology,
            bitfit_bias_name=bitfit_bias_name,
        )

        # retrieve initialized weights of parallel linear layer
        if model_parallel_size == 1:
            merged_weights = parallel_linear.weight
        else:
            merged_weights[
                :,
                topology.model_parallel_rank * (in_features // 2) : (topology.model_parallel_rank + 1)
                * (in_features // 2),
            ].copy_(parallel_linear.weight)
        if bias:
            assert parallel_linear.bias is not None
            merged_bias.copy_(parallel_linear.bias)

    torch.distributed.all_reduce(merged_weights)
    # the bias should be constant across ranks, this test only works because the bias is initialized to zero
    # torch.distributed.all_reduce(merged_bias)

    linear = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        device=topology.device,
    )
    # replace initialized weights of linear layer with the one from the parallel linear layer
    # to be able to compare the forward pass results
    linear.weight.data.copy_(merged_weights.data)
    if bias:
        linear.bias.data.copy_(merged_bias.data)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(
        (
            1,
            in_features,
        ),
        device=topology.device,
    )

    output_linear = linear(x)
    output_parallel_linear = parallel_linear(x)

    if model_parallel_size == 1:
        assert torch.isclose(
            output_parallel_linear, output_linear, atol=1e-7
        ).all(), "output from parallel implementation and default torch.nn.Linear implementation differs"
    else:
        # in the model parallel case we encounter problems with precision so that a torch.isclose() does not cover it
        delta = (output_parallel_linear - output_linear).abs().mean()
        assert (
            delta < 0.005
        ), f"output from parallel implementation and default implementation differs with delta of {delta}"


@pytest.mark.parametrize("model_parallel_size", [1, 2])
@pytest.mark.parametrize("in_features", [1, 8, 17, 32])
@pytest.mark.parametrize("out_features", [1, 8, 17, 32])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("linear_layer_type", [ColumnParallelLinear, RowParallelLinear])
@pytest.mark.parametrize("bitfit_bias_name", [None])
def test_parallel_linear(
    model_parallel_size: int,
    in_features: int,
    out_features: int,
    bias: bool,
    linear_layer_type: torch.autograd.Function,
    bitfit_bias_name,
):
    """
    tests if the output from parallel embedding implementation and the default torch.nn.Embedding do not differ
    """
    # Skip test if model parallel is not possible with specified vocab size
    if linear_layer_type == ColumnParallelLinear and model_parallel_size and out_features % model_parallel_size != 0:
        pytest.skip(
            f"cannot column parallelize, out_features ({out_features}) "
            f"needs to be divisible by model parallel size ({model_parallel_size})"
        )
    if linear_layer_type == RowParallelLinear and model_parallel_size and in_features % model_parallel_size != 0:
        pytest.skip(
            f"cannot row parallelize, in_features ({in_features}) "
            f"needs to be divisible by model parallel size ({model_parallel_size})"
        )

    world_size = model_parallel_size

    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"cannot run test with world size {world_size} with available {torch.cuda.device_count()} cuda devices"
        )

    return_dict_continuously_trained_model = dist_launcher(
        run_func=run_test_parallel_linear,
        world_size=model_parallel_size,
        master_port=find_free_port(),
        model_parallel_size=model_parallel_size,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        linear_layer_type=linear_layer_type,
        bitfit_bias_name=bitfit_bias_name,
    )
    assert return_dict_continuously_trained_model is not None
    pass
