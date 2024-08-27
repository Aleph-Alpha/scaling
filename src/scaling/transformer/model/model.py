import math
import re
from typing import Sequence, TypeAlias, Union

import torch

from scaling.core import (
    BaseOptimizer,
    CoreParameterMeta,
    LayerSpec,
    Optimizer,
    OptimizerParamGroup,
    OptimizerParamGroupConfig,
    ParallelModule,
    TiedLayerSpec,
    Topology,
)
from scaling.core.logging import logger
from scaling.core.nn.parallel_module.buffers import BufferType
from scaling.transformer.data.text_dataset_batch import TextDatasetBatch
from scaling.transformer.data.utils import remove_cumulative_seq_lengths_padding

from ..context import (
    LearningRateSchedulerConfig,
    OptimizerConfig,
    TransformerArchitectureConfig,
    TransformerContext,
)
from ..context.config import TrainingConfig
from .layers import (
    EmbeddingInput,
    LayerNormWrapper,
    TransformerEmbeddingHead,
    TransformerLayer,
    TransformerLayerIO,
    TransformerLMHead,
    TransformerLMHeadTied,
)

NamedParameterMeta: TypeAlias = tuple[str, torch.Tensor, CoreParameterMeta]


def loss_function(output: TransformerLayerIO, batch: TextDatasetBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Calculates cross entropy loss for every probability distribution in the output activations and the target token ids.
    Returns the average loss for this batch as well as the accuracy.
    """
    assert (
        batch.target_token_ids is not None
    ), "target_token_ids not set in batch; you may want to revisit the implementation of Batch.only_targets()"
    assert (
        batch.loss_weights is not None
    ), "loss_weights not set in batch; you may want to revisit the implementation of Batch.only_targets()"

    # Flatten Tensors
    loss_weights_flatten = batch.loss_weights.float().flatten()
    output_activations_flatten = output.activations.float().reshape(-1, output.activations.shape[-1])
    target_token_ids_flatten = batch.target_token_ids.flatten()

    # Get flattened cross entropy losses
    losses = torch.nn.functional.cross_entropy(
        output_activations_flatten,
        target=target_token_ids_flatten,
        reduction="none",
    )
    # Average of losses
    loss = torch.sum(losses * loss_weights_flatten) / loss_weights_flatten.sum()

    # Get the loss mask from the loss weights
    loss_mask_flatten: torch.Tensor = loss_weights_flatten > 0
    loss_mask_flatten = loss_mask_flatten.float()

    accuracy = (output_activations_flatten.argmax(-1) == target_token_ids_flatten).float()
    accuracy = torch.sum(accuracy * loss_mask_flatten) / loss_mask_flatten.sum()

    return loss, {"accuracy": accuracy}


def metrics_aggregation_fn(topology: Topology, metrics: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # Map function to
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


class TransformerParallelModule(ParallelModule[TransformerLayerIO, TextDatasetBatch]):
    def _execute_send_activations(
        self,
        buffer_id: int,
    ) -> None:
        # remove un-padded attention mask before sending activations to next pipe rank
        # to prevent potentially sending different shapes
        pipeline_output = self.pipe_buffer.data[BufferType.PIPELINE_STAGE_OUTPUT][buffer_id]
        assert pipeline_output is not None
        pipeline_output.cumulative_seq_lengths = None  # type: ignore[union-attr]

        super()._execute_send_activations(buffer_id=buffer_id)

    def _execute_receive_activations(self, buffer_id: int) -> None:
        super()._execute_receive_activations(buffer_id=buffer_id)

        # unpacking of the attention mask once for every pipe rank after we received the activations
        pipeline_input = self.pipe_buffer.data[BufferType.PIPELINE_STAGE_INPUT][buffer_id]
        assert pipeline_input is not None
        if pipeline_input.cumulative_seq_lengths is None:
            assert pipeline_input.cumulative_seq_lengths_padded is not None
            pipeline_input.cumulative_seq_lengths = remove_cumulative_seq_lengths_padding(
                pipeline_input.cumulative_seq_lengths_padded
            )


def get_transformer_layer_specs(
    architecture_config: TransformerArchitectureConfig, topology: Topology | None = None
) -> list[LayerSpec]:
    init_method = lambda x: torch.nn.init.normal_(  # noqa: E731
        x,
        mean=0.0,
        std=math.sqrt(2 / (architecture_config.hidden_size * 5)),
    )

    layer_specs: list[Union[TiedLayerSpec, LayerSpec]] = []

    # ---------------------embedding layer---------------------------
    tied_weight_attributes = list()
    if architecture_config.weight_tying:
        tied_weight_attributes.append("embedding.weight")

    if architecture_config.weight_tying:
        layer_specs.append(
            TiedLayerSpec(
                module_class=EmbeddingInput,
                key="embedding_tying",
                tied_weight_attributes=tied_weight_attributes,
                architecture_config=architecture_config,
                topology=topology,
                init_method=init_method,
            )
        )
    else:
        layer_specs.append(
            LayerSpec(
                module_class=EmbeddingInput,
                architecture_config=architecture_config,
                topology=topology,
                init_method=init_method,
            )
        )

    # ----------------------transformer layers------------------------

    for layer_index in range(architecture_config.num_layers):
        layer_specs.append(
            LayerSpec(
                TransformerLayer,
                architecture_config=architecture_config,
                topology=topology,
                layer_index=layer_index,
                init_method=init_method,
            )
        )

    # ----------------------final layernorm---------------------------

    layer_specs.append(
        LayerSpec(
            LayerNormWrapper,
            architecture_config=architecture_config,
            topology=topology,
            layer_index=architecture_config.num_layers,  # this will be the layer following the last transformer layer
        )
    )

    # ----------------------LM head layer-----------------------------

    if architecture_config.weight_tying:
        layer_specs.append(
            TiedLayerSpec(
                module_class=TransformerLMHeadTied,
                key="embedding_tying",
                tied_weight_attributes=tied_weight_attributes,
                architecture_config=architecture_config,
                topology=topology,
                init_method=init_method,
            )
        )
    else:
        layer_specs.append(
            LayerSpec(
                module_class=TransformerLMHead,
                architecture_config=architecture_config,
                topology=topology,
                init_method=init_method,
            )
        )
    if architecture_config.embedding_head_config is not None:
        layer_specs.append(
            LayerSpec(
                module_class=TransformerEmbeddingHead,
                architecture_config=architecture_config,
                topology=topology,
            )
        )

    # ----------------------------------------------------------------

    return layer_specs


def init_model(
    context: TransformerContext,
    use_continuous_recommunication: bool = False,
) -> TransformerParallelModule:
    # instantiate transformer model

    layer_specs = get_transformer_layer_specs(
        architecture_config=context.config.transformer_architecture, topology=context.topology
    )
    model = TransformerParallelModule(
        layer_specs=layer_specs,
        topology=context.topology,
        profiler_config=context.config.profiler,
        use_continuous_recommunication=use_continuous_recommunication,
    )

    return model


def get_parameter_groups(
    context: TransformerContext,
    model: TransformerParallelModule,
    learning_rate_scheduler_config: LearningRateSchedulerConfig | None = None,
    embedding_learning_rate_scheduler_config: LearningRateSchedulerConfig | None = None,
) -> list[OptimizerParamGroup]:
    if learning_rate_scheduler_config is None:
        learning_rate_scheduler_config = context.config.learning_rate_scheduler

    if embedding_learning_rate_scheduler_config is None:
        embedding_learning_rate_scheduler_config = context.config.embedding_learning_rate_scheduler

    embedding_weight_decay, parameters_no_weight_decay, parameters_weight_decay = _extract_parameters(
        context.config.training, model.named_parameters_with_meta()
    )

    parameter_counts = [len(parameters_weight_decay), len(parameters_no_weight_decay), len(embedding_weight_decay)]
    parameter_count_total = sum(parameter_counts)
    parameter_count_total_tensor = torch.tensor([parameter_count_total], dtype=torch.long, device="cuda")
    torch.distributed.all_reduce(parameter_count_total_tensor)
    parameter_count_total = int(parameter_count_total_tensor.item())
    assert parameter_count_total > 0, "did not specify any finetuneable parameters on any rank"

    parameter_groups = []

    parameter_set = {p[0] for p in parameters_weight_decay + parameters_no_weight_decay + embedding_weight_decay}
    logger.warning(f"training parameters: {parameter_set}")

    parameter_counts_max_tensor = torch.tensor(
        parameter_counts,
        dtype=torch.int,
        device="cuda",
    )
    # collect whether there is at least one non-empty group for weight_decay, resp. no_weight decay parameters on
    # some rank
    torch.distributed.all_reduce(parameter_counts_max_tensor, op=torch.distributed.ReduceOp.MAX)

    # if at least one rank has a non-empty group we need to add the group everywhere since it hangs otherwise
    if parameter_counts_max_tensor[0].item() > 0:
        parameter_groups.append(
            OptimizerParamGroup(
                named_parameters_with_meta=parameters_weight_decay,
                config=OptimizerParamGroupConfig(
                    name="weight_decay_params",
                    weight_decay=context.config.training.weight_decay,
                    learning_rate_scheduler=learning_rate_scheduler_config,
                ),
            )
        )
    if parameter_counts_max_tensor[1].item() > 0:
        parameter_groups.append(
            OptimizerParamGroup(
                named_parameters_with_meta=parameters_no_weight_decay,
                config=OptimizerParamGroupConfig(
                    name="no_weight_decay_params",
                    weight_decay=0.0,
                    learning_rate_scheduler=learning_rate_scheduler_config,
                ),
            )
        )
    if parameter_counts_max_tensor[2].item() > 0:
        parameter_groups.append(
            OptimizerParamGroup(
                named_parameters_with_meta=embedding_weight_decay,
                config=OptimizerParamGroupConfig(
                    name="embedding_weight_decay_params",
                    weight_decay=context.config.training.weight_decay,
                    learning_rate_scheduler=embedding_learning_rate_scheduler_config,
                ),
            )
        )

    # Safety check whether the number of optimizer groups is the same on all ranks
    len_param_groups_tensor_list = [
        torch.zeros([1], dtype=torch.int, device="cuda")
    ] * torch.distributed.get_world_size()
    torch.distributed.all_gather(
        len_param_groups_tensor_list,
        torch.tensor([len(parameter_groups)], dtype=torch.int, device="cuda"),
    )

    len_param_groups_list = [t.item() for t in len_param_groups_tensor_list]
    assert (
        len(set(len_param_groups_list)) == 1
    ), f"Got different number of optimizer groups on different ranks \n {len_param_groups_list}"

    assert len(parameter_groups) > 0, "Number of optimizer groups is zero"

    return parameter_groups


def _extract_parameters(
    training_config: TrainingConfig, named_parameters_with_meta: list[NamedParameterMeta]
) -> tuple[list[NamedParameterMeta], list[NamedParameterMeta], list[NamedParameterMeta]]:
    parameters_weight_decay = []
    parameters_no_weight_decay = []
    embeddings_weight_decay = []
    found_finetunable_parameters = set()
    for named_param_meta in named_parameters_with_meta:
        if training_config.finetune:
            match = _find_matching_param(named_param_meta, training_config.finetunable_parameters)
            if match is None:
                continue
            found_finetunable_parameters.add(match)

        name = named_param_meta[0]
        if name.endswith(".bias"):
            parameters_no_weight_decay.append(named_param_meta)
        elif training_config.use_separate_lr_on_embeddings and name == "embedding.weight":
            assert not training_config.finetune, "Can not use separate lr on embeddings with finetuning"
            embeddings_weight_decay.append(named_param_meta)
        else:
            parameters_weight_decay.append(named_param_meta)

    if unmatched_parameters := _find_global_unmatched_parameters(
        found_finetunable_parameters, training_config.finetunable_parameters
    ):
        raise ValueError(f"Unmatched finetunable parameters: {unmatched_parameters}")
    if parameters_exclude := training_config.parameters_exclude:
        parameters_weight_decay = _filter_by_param(parameters_exclude, parameters_weight_decay)
        parameters_no_weight_decay = _filter_by_param(parameters_exclude, parameters_no_weight_decay)
        embeddings_weight_decay = _filter_by_param(parameters_exclude, embeddings_weight_decay)
    return (
        embeddings_weight_decay,
        parameters_no_weight_decay,
        parameters_weight_decay,
    )


def _find_global_unmatched_parameters(
    found_finetunable_parameters: set[str], finetunable_parameters: Sequence[str]
) -> set[str]:
    """
    Returns a set of finetunable parameters that are missing on all ranks
    """
    local_unmatched_parameters = set(finetunable_parameters) - found_finetunable_parameters
    global_unmatched_parameters: list[set[str]] = [set()] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(global_unmatched_parameters, local_unmatched_parameters)
    return set.intersection(*global_unmatched_parameters)


def _filter_by_param(
    parameters_exclude: list[str], parameter_list: list[NamedParameterMeta]
) -> list[NamedParameterMeta]:
    return [param for param in parameter_list if _find_matching_param(param, parameters_exclude) is None]


def _find_matching_param(key: NamedParameterMeta, data: list[str]) -> str | None:
    return next((item for item in data if re.search(item, key[0]) is not None), None)


def init_optimizer(
    context: TransformerContext,
    model: TransformerParallelModule,
    optimizer_config: OptimizerConfig | None = None,
    learning_rate_scheduler_config: LearningRateSchedulerConfig | None = None,
    embedding_learning_rate_scheduler_config: LearningRateSchedulerConfig | None = None,
) -> BaseOptimizer:
    parameter_groups = get_parameter_groups(
        context=context,
        model=model,
        learning_rate_scheduler_config=learning_rate_scheduler_config,
        embedding_learning_rate_scheduler_config=embedding_learning_rate_scheduler_config,
    )
    optimizer = Optimizer(
        config=(optimizer_config if optimizer_config is not None else context.config.optimizer),
        parameter_groups=parameter_groups,
        topology=context.topology,
    )

    return optimizer
