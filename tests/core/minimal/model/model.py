from typing import Any, Dict, List, Optional, Tuple

import torch

from scaling.core import (
    BaseLayer,
    BaseLayerIO,
    BaseOptimizer,
    ColumnParallelLinear,
    LayerNorm,
    LayerNormConfig,
    LayerSpec,
    Optimizer,
    OptimizerParamGroup,
    OptimizerParamGroupConfig,
    ParallelModule,
    RowParallelLinear,
    TiedLayerSpec,
    Topology,
    VocabParallelEmbedding,
)
from scaling.core.nn.linear.utils import all_concat, copy_to_tensor_model_parallel_region
from tests.core.minimal.context import MinimalContext
from tests.core.minimal.data import MinimalBatch


class MinimalLinearIO(BaseLayerIO):
    activations: torch.Tensor

    def __init__(self, activations: torch.Tensor):
        self.activations = activations


class MinimalEmbeddingInput(BaseLayer[MinimalBatch, MinimalLinearIO, MinimalLinearIO]):
    def __init__(self, topology: Optional[Topology] = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.topology = topology
        self.embedding = VocabParallelEmbedding(
            num_embeddings=2, embedding_dim=2, topology=self.topology, dtype=dtype, finetunable_token_ids=[]
        )
        self.register_buffer("buffer_for_testing", torch.tensor([0.0, 1.0, 2.0]))

    def forward(self, x: MinimalBatch) -> MinimalLinearIO:
        activations = self.embedding(x.inputs)
        return MinimalLinearIO(activations=activations)

    @staticmethod
    def input_to_tuple(
        input: MinimalBatch,
    ) -> Tuple[Any, ...]:
        """
        convert layer input to a tuple with tensors as values for pipe communication and activation checkpointing
        this may include a split to model parallel
        tuple_to_input will be called on the tuple, here you might need to merge split tensors again
        we are using a tuple because torch requires tuples for activation checkpointing
        """
        return (input.inputs, input.targets)

    @staticmethod
    def tuple_to_input(d: Tuple[Any, ...]) -> MinimalBatch:
        """
        convert a tuple with tensors as values for pipe communication to the layer input class
        you might need to merge split tensors again
        """
        return MinimalBatch(inputs=d[0], targets=d[1])

    @staticmethod
    def output_to_tuple(
        output: MinimalLinearIO,
    ) -> Tuple[Any, ...]:
        """
        convert layer output to a tuple with tensors as values for pipe communication and activation checkpointing
        this may include a split to model parallel
        tuple_to_input will be called on the tuple, here you might need to merge split tensors again
        we are using a tuple because torch requires tuples for activation checkpointing
        """
        return (output.activations,)

    @staticmethod
    def tuple_to_last_stage_activation(d: Tuple[Any, ...]) -> MinimalLinearIO:
        """
        convert a tuple with tensors as values for pipe communication to the last layer's output class
        you might need to merge split tensors again
        """
        return MinimalLinearIO(activations=d[0])


class MinimalBaseLayer(BaseLayer[MinimalLinearIO, MinimalLinearIO, MinimalLinearIO]):
    @staticmethod
    def input_to_tuple(
        input: MinimalLinearIO,
    ) -> Tuple[Any, ...]:
        """
        convert layer input to a tuple with tensors as values for pipe communication and activation checkpointing
        this may include a split to model parallel
        tuple_to_input will be called on the tuple, here you might need to merge split tensors again
        we are using a tuple because torch requires tuples for activation checkpointing
        """
        return (input.activations,)

    @staticmethod
    def tuple_to_input(d: Tuple[Any, ...]) -> MinimalLinearIO:
        """
        convert a tuple with tensors as values for pipe communication to the layer input class
        you might need to merge split tensors again
        """
        return MinimalLinearIO(activations=d[0])

    @staticmethod
    def output_to_tuple(
        output: MinimalLinearIO,
    ) -> Tuple[Any, ...]:
        """
        convert layer output to a tuple with tensors as values for pipe communication and activation checkpointing
        this may include a split to model parallel
        tuple_to_input will be called on the tuple, here you might need to merge split tensors again
        we are using a tuple because torch requires tuples for activation checkpointing
        """
        return (output.activations,)

    @staticmethod
    def tuple_to_last_stage_activation(d: Tuple[Any, ...]) -> MinimalLinearIO:
        """
        convert a tuple with tensors as values for pipe communication to the last layer's output class
        you might need to merge split tensors again
        """
        return MinimalLinearIO(activations=d[0])


class MinimalLinearRowParallel(MinimalBaseLayer):
    def __init__(
        self,
        topology: Optional[Topology] = None,
        bitfit_bias_name: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.topology = topology
        self.linear = RowParallelLinear(
            in_features=2,
            out_features=2,
            bias=True,
            topology=self.topology,
            bitfit_bias_name=bitfit_bias_name,
            dtype=dtype,
        )

    def forward(self, x: MinimalLinearIO) -> MinimalLinearIO:
        activations = self.linear(x.activations)
        assert activations.shape == x.activations.shape
        return MinimalLinearIO(activations=activations)


class MinimalLinearColumnParallel(MinimalBaseLayer):
    def __init__(
        self,
        topology: Optional[Topology] = None,
        bitfit_bias_name: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.topology = topology
        self.linear = ColumnParallelLinear(
            in_features=2,
            out_features=2,
            bias=True,
            topology=self.topology,
            bitfit_bias_name=bitfit_bias_name,
            dtype=dtype,
        )

    def forward(self, x: MinimalLinearIO) -> MinimalLinearIO:
        activations = self.linear(x.activations)
        assert activations.shape == x.activations.shape
        return MinimalLinearIO(activations=activations)


class MinimalLayerNorm(MinimalBaseLayer):
    def __init__(
        self,
        topology: Optional[Topology] = None,
        bitfit_bias_name: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.topology = topology
        self.norm = LayerNorm(
            config=LayerNormConfig(),
            normalized_shape=2,
            device=torch.device("cpu") if self.topology is None else self.topology.device,
            bitfit_bias_name=bitfit_bias_name,
            dtype=dtype,
        )

    def forward(self, x: MinimalLinearIO) -> MinimalLinearIO:
        activations = self.norm(x.activations)
        assert activations.shape == x.activations.shape

        return MinimalLinearIO(activations=activations)


class MinimalEmbeddingTied(MinimalBaseLayer):
    def __init__(self, topology: Optional[Topology] = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.topology = topology
        self.embedding = VocabParallelEmbedding(
            num_embeddings=2, embedding_dim=2, topology=self.topology, dtype=dtype, finetunable_token_ids=[]
        )

    def forward(self, x: MinimalLinearIO) -> MinimalLinearIO:
        activations = x.activations
        if self.topology is not None:
            activations = copy_to_tensor_model_parallel_region(activations, topology=self.topology)
        activations = torch.nn.functional.linear(activations, self.embedding.weight)
        if self.topology is not None:
            activations = all_concat(activations, dim=-1, topology=self.topology)
        assert activations.shape == x.activations.shape
        return MinimalLinearIO(activations=activations)


class MinimalParallelModule(ParallelModule[MinimalLinearIO, MinimalBatch]):
    pass


def loss_function(output: MinimalLinearIO, batch: MinimalBatch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Here we are calculating the actual loss the model is going to train with.
    loss = torch.nn.functional.mse_loss(output.activations.sum(dim=0).float(), batch.targets.float())
    # Here we are calculating a metric which we find interesting to look at.
    l1_metric = torch.nn.functional.l1_loss(output.activations.sum(dim=0).float(), batch.targets.float())

    return loss, {"l1_metric": l1_metric}


def metrics_aggregation_fn(topology: Topology, metrics: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Map function to
    def map_all_reduce_fn(t: torch.Tensor):
        assert topology.config.data_parallel_size is not None

        if topology.config.data_parallel_size > 1:
            torch.distributed.all_reduce(t, group=topology.data_parallel_group)
            t = t / topology.config.data_parallel_size

        return t

    metrics_stack: Dict[str, torch.Tensor] = {
        k: torch.mean(torch.stack([m[k] for m in metrics])) for k in metrics[0].keys()
    }
    metrics_stack = {k: map_all_reduce_fn(v) for k, v in metrics_stack.items()}

    return metrics_stack


def init_model(context: MinimalContext) -> MinimalParallelModule:
    # instantiate small model

    if context.config.training.weight_tying:
        layer_specs = [
            TiedLayerSpec(
                module_class=MinimalEmbeddingInput,
                key="embedding_tying",
                tied_weight_attributes=["embedding.weight"],
                topology=context.topology,
                dtype=context.config.training.precision.dtype,
            ),
            LayerSpec(
                module_class=MinimalLinearColumnParallel,
                topology=context.topology,
                bitfit_bias_name=context.config.training.bitfit_bias_name,
                dtype=context.config.training.precision.dtype,
            ),
            TiedLayerSpec(
                module_class=MinimalEmbeddingTied,
                key="embedding_tying",
                tied_weight_attributes=["embedding.weight"],
                topology=context.topology,
                dtype=context.config.training.precision.dtype,
            ),
            LayerSpec(
                module_class=MinimalLinearRowParallel,
                topology=context.topology,
                bitfit_bias_name=context.config.training.bitfit_bias_name,
                dtype=context.config.training.precision.dtype,
            ),
            LayerSpec(
                module_class=MinimalLayerNorm,
                topology=context.topology,
                bitfit_bias_name=context.config.training.bitfit_bias_name,
                dtype=context.config.training.precision.dtype,
            ),
        ]
    else:
        layer_specs = [
            LayerSpec(module_class=MinimalEmbeddingInput, topology=context.topology),
            LayerSpec(
                module_class=MinimalLinearColumnParallel,
                topology=context.topology,
                bitfit_bias_name=context.config.training.bitfit_bias_name,
            ),
            LayerSpec(
                module_class=MinimalLinearRowParallel,
                topology=context.topology,
                bitfit_bias_name=context.config.training.bitfit_bias_name,
            ),
            LayerSpec(
                module_class=MinimalLayerNorm,
                topology=context.topology,
                bitfit_bias_name=context.config.training.bitfit_bias_name,
            ),
        ]
    return MinimalParallelModule(
        layer_specs=layer_specs,
        topology=context.topology,
        profiler_config=context.config.profiler,
    )


def init_optimizer(context: MinimalContext, model: ParallelModule) -> BaseOptimizer:
    if context.config.training.bitfit_bias_name == "" or context.config.training.bitfit_bias_name is None:
        bias_name_in_training = ".bias"
    else:
        bias_name_in_training = f".bias_{context.config.training.bitfit_bias_name}"

    parameter_groups = [
        OptimizerParamGroup(
            named_parameters_with_meta=[
                (n, p, m) for n, p, m in model.named_parameters_with_meta() if not n.endswith(bias_name_in_training)
            ],
            config=OptimizerParamGroupConfig(
                name="weight_decay_params",
                weight_decay=context.config.training.weight_decay,
                learning_rate_scheduler=context.config.learning_rate_scheduler,
            ),
        ),
        OptimizerParamGroup(
            named_parameters_with_meta=[
                (n, p, m) for n, p, m in model.named_parameters_with_meta() if n.endswith(bias_name_in_training)
            ],
            config=OptimizerParamGroupConfig(
                name="no_weight_decay_params",
                weight_decay=0.0,
                learning_rate_scheduler=context.config.learning_rate_scheduler,
            ),
        ),
    ]
    return Optimizer(
        config=context.config.optimizer,
        parameter_groups=parameter_groups,
        topology=context.topology,
    )
