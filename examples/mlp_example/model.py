from typing import Any, Callable, Tuple

import torch
import torch.nn.functional as F

from scaling.core import BaseLayerIO
from scaling.core import ColumnParallelLinear
from scaling.core import RowParallelLinear
from scaling.core import BaseLayer
from scaling.core import ParallelModule
from scaling.core import LayerSpec
from scaling.core import BaseOptimizer
from scaling.core import Optimizer
from scaling.core import OptimizerParamGroup
from scaling.core import OptimizerParamGroupConfig

from scaling.core.topology import Topology
from examples.mlp_example.context import MLPContext
from examples.mlp_example.data import MNISTDatasetBatch


class MLPLayerIO(BaseLayerIO):

    def __init__(self, activations: torch.Tensor):
        self.activations = activations


class MLPBaseLayer(BaseLayer[MLPLayerIO, MLPLayerIO, MLPLayerIO]):
    @staticmethod
    def input_to_tuple(input: MLPLayerIO) -> Tuple[Any, ...]:
        return (input.activations,)

    @staticmethod
    def tuple_to_input(d: Tuple[Any, ...]) -> MLPLayerIO:
        return MLPLayerIO(activations=d[0])

    @staticmethod
    def output_to_tuple(output: MLPLayerIO) -> Tuple[Any, ...]:
        return (output.activations,)

    @staticmethod
    def tuple_to_last_stage_activation(d: Tuple[Any, ...]) -> MLPLayerIO:
        return MLPLayerIO(activations=d[0])


class MLPLinearColumnParallel(MLPBaseLayer):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        topology: Topology,
        parallel_output: bool = True,
        act_fn: Callable = lambda x: x,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.linear = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            parallel_output=parallel_output,
            topology=topology,
            dtype=dtype,
            bias=True,
        )

        self.act_fn = act_fn

    def forward(self, x: MLPLayerIO | MNISTDatasetBatch) -> MLPLayerIO:
        inp = x.activations if isinstance(x, MLPLayerIO) else x.inputs
        pre_activations = torch.flatten(inp, start_dim=1)  # type: ignore
        pre_activations = self.linear(pre_activations)
        activations = self.act_fn(pre_activations)
        return MLPLayerIO(activations=activations)


class MLPLinearRowParallel(MLPBaseLayer):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        topology: Topology,
        act_fn: Callable = lambda x: x,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.linear = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            parallel_input=True,
            topology=topology,
            dtype=dtype,
            bias=True,
        )

        self.act_fn = act_fn

    def forward(self, x: MLPLayerIO | MNISTDatasetBatch) -> MLPLayerIO:
        inp = x.activations if isinstance(x, MLPLayerIO) else x.inputs
        pre_activations = torch.flatten(inp, start_dim=1)  # type: ignore
        pre_activations = self.linear(pre_activations)
        activations = self.act_fn(pre_activations)
        return MLPLayerIO(activations=activations)


def loss_function(
    output: MLPLayerIO,
    batch: MNISTDatasetBatch
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = torch.nn.functional.cross_entropy(output.activations, torch.tensor(batch.targets, dtype=torch.long))
    accuracy = sum(output.activations.argmax(dim=1) == batch.targets) / batch.targets.shape[0]  # type: ignore
    return loss, { "accuracy": accuracy }


def metrics_aggregation_fn(
    topology: Topology,
    metrics: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    def map_all_reduce_fn(t: torch.Tensor) -> torch.Tensor:
        if topology.config.data_parallel_size > 1:
            torch.distributed.all_reduce(t, group=topology.data_parallel_group)
            t = t / topology.config.data_parallel_size
        return t

    metrics_stack: dict[str, torch.Tensor] = {
        k: torch.mean(torch.tensor([m[k] for m in metrics]).cuda()) for k in metrics[0]
    }

    metrics_stack = {k: map_all_reduce_fn(v) for k, v in metrics_stack.items()}
    return metrics_stack


def init_model(context: MLPContext):
    layer_specs = []
    # +2 for input and output layers.
    n_layers = context.config.architecture.n_hidden_layers + 2
    hidden_dim = context.config.architecture.hidden_dim
    module_classes = [MLPLinearColumnParallel, MLPLinearRowParallel]
    input_dim = 784  # MNIST item flattened.
    output_dim = 10

    for i in range(n_layers):
        kwargs = {}
        kwargs.update({"in_features": hidden_dim if i != 0 else input_dim})
        kwargs.update({"out_features": hidden_dim if i != n_layers - 1 else output_dim})
        kwargs.update({"parallel_output": False} if i == n_layers - 1 else {})
        kwargs.update({"act_fn": F.relu} if i != n_layers -1 else {})  # type: ignore

        layer_specs.append(
            LayerSpec(
                module_class=module_classes[i % 2],
                topology=context.topology,
                dtype=torch.float16,
                **kwargs,
            ),
        )

    return ParallelModule(
        layer_specs=layer_specs,
        topology=context.topology,
        profiler_config=context.config.profiler,
    )


def init_optimizer(context: MLPContext, model: ParallelModule) -> BaseOptimizer:
    parameter_groups = [
        OptimizerParamGroup(
            named_parameters_with_meta=[
                (n, p, m) for n, p, m in model.named_parameters_with_meta()
            ],
            config=OptimizerParamGroupConfig(
                name="weight_decay_params",
                weight_decay=context.config.training.weight_decay,
                learning_rate_scheduler=context.config.learning_rate_scheduler,
            ),
        ),
    ]

    return Optimizer(
        config=context.config.optimizer,
        parameter_groups=parameter_groups,
        topology=context.topology,
    )
