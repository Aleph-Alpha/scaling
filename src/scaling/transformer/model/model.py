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
from scaling.transformer.context import (
    OptimizerConfig,
    TrainingGroupConfig,
    TransformerArchitectureConfig,
    TransformerContext,
)
from scaling.transformer.data.text_dataset_batch import TextDatasetBatch
from scaling.transformer.data.utils import remove_cumulative_seq_lengths_padding
from scaling.transformer.model.layers import (
    EmbeddingInput,
    LayerNormWrapper,
    TransformerEmbeddingHead,
    TransformerLayer,
    TransformerLayerIO,
    TransformerLMHead,
    TransformerLMHeadTied,
)

NamedParameterMeta: TypeAlias = tuple[str, torch.Tensor, CoreParameterMeta]


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

    if architecture_config.umup.enable:
        assert not architecture_config.weight_tying, "u-mup and weight tying are not compatible"

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
            umup_on_residual=False,
        )
    )

    # ----------------------LM head layer-----------------------------
    if architecture_config.lm_head:
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

    # ----------------------EmbeddingHead layer-----------------------------
    if architecture_config.embedding_head_config is not None:
        layer_specs.append(
            LayerSpec(
                module_class=TransformerEmbeddingHead,
                architecture_config=architecture_config,
                topology=topology,
            )
        )

    return layer_specs


def init_model(
    context: TransformerContext,
    use_continuous_recommunication: bool = False,
) -> TransformerParallelModule:
    # instantiate transformer model

    layer_specs = get_transformer_layer_specs(
        architecture_config=context.config.transformer_architecture,
        topology=context.topology,
    )
    model = TransformerParallelModule(
        layer_specs=layer_specs,
        topology=context.topology,
        profiler_config=context.config.profiler,
        use_continuous_recommunication=use_continuous_recommunication,
    )

    if context.config.transformer_architecture.umup.enable:
        if context.config.transformer_architecture.umup.normalize_depth_to_num_layers:
            depth = context.config.transformer_architecture.num_layers
        else:
            depth = 2 * context.config.transformer_architecture.num_layers
        model.umup_setup(
            effective_batch_size=context.config.topology.global_batch_size
            * context.config.transformer_architecture.sequence_length,
            depth=depth,
            avg_sequence_length=context.config.transformer_architecture.sequence_length,
            allow_non_umup_params=context.config.transformer_architecture.umup.allow_non_umup_params,
        )

    return model


def get_parameter_groups(
    context: TransformerContext,
    model: TransformerParallelModule,
) -> list[OptimizerParamGroup]:
    parameters_groups = _extract_parameters(
        context.config.training_groups,
        model.named_parameters_with_meta(),
        context.config.training.allow_missing_params_in_optimizer,
    )

    parameter_counts = [len(group[0]) for group in parameters_groups]
    parameter_count_total = sum(parameter_counts)
    parameter_count_total_tensor = torch.tensor([parameter_count_total], dtype=torch.long, device="cuda")
    torch.distributed.all_reduce(parameter_count_total_tensor)
    parameter_count_total = int(parameter_count_total_tensor.item())
    assert parameter_count_total > 0, "did not specify any finetuneable parameters on any rank"

    parameter_groups = []

    parameter_set = {p[0] for (parameters_group, _) in parameters_groups for p in parameters_group}
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
    for idx, (named_parameters_with_meta, training_group_config) in enumerate(parameters_groups):
        if parameter_counts_max_tensor[idx].item() > 0:
            parameter_groups.append(
                OptimizerParamGroup(
                    named_parameters_with_meta=named_parameters_with_meta,
                    config=OptimizerParamGroupConfig(
                        name=training_group_config.group_name,
                        weight_decay=training_group_config.weight_decay,
                        independent_weight_decay=training_group_config.independent_weight_decay,
                        learning_rate_scheduler=training_group_config.learning_rate_scheduler,
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
    training_groups: list[TrainingGroupConfig],
    named_parameters_with_meta: list[NamedParameterMeta],
    allow_missing_params_in_optimizer: bool,
) -> list[tuple[list[NamedParameterMeta], TrainingGroupConfig]]:
    parameters_groups = []
    parameters_groups_config = []
    found_group_parameters_list: list[set[str]] = []

    for training_group in training_groups:
        parameters: list[NamedParameterMeta] = []
        found_include = set()

        for named_param_meta in named_parameters_with_meta:
            if parameters_include := training_group.parameters_include:
                match = _find_matching_param(named_param_meta, parameters_include)
                if match is None:
                    continue

                found_include.add(match)

            parameters.append(named_param_meta)

        if training_group.parameters_include:
            if unmatched_parameters := _find_global_unmatched_parameters(
                found_include, training_group.parameters_include
            ):
                raise ValueError(f"Unmatched finetunable parameters: {unmatched_parameters}")

        if parameters_exclude := training_group.parameters_exclude:
            parameters = _filter_by_param(parameters_exclude, parameters)

        parameters_groups.append(parameters)
        parameters_groups_config.append(training_group)
        found_group_parameters_list.append({n for (n, _, _) in parameters})

    all_used_parameters: list[str] = []
    for found_group_parameters in found_group_parameters_list:
        for parameter in found_group_parameters:
            if parameter in all_used_parameters:
                raise ValueError(
                    f"Parameter '{parameter}' in more then one training_group. Use "
                    f"'parameters_include' and 'parameters_exclude' to ensure no overlap between training groups."
                )

            all_used_parameters.append(parameter)

    if not allow_missing_params_in_optimizer:
        all_params_names: set[str] = {n for (n, _, _) in named_parameters_with_meta}
        not_used_params = set(all_used_parameters).symmetric_difference(all_params_names)

        if len(not_used_params) > 0:
            raise ValueError(
                f"""The following parameters were not added to any optimizer group: {not_used_params}".
                If this is intended, add the names to 'allow_missing_params_in_optimizer`."""
            )

    return list(zip(parameters_groups, parameters_groups_config))


def _find_global_unmatched_parameters(
    found_parameters_include: set[str], parameters_include: Sequence[str]
) -> set[str]:
    """
    Returns a set of finetunable parameters that are missing on all ranks
    """
    local_unmatched_parameters = set(parameters_include) - found_parameters_include
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
) -> BaseOptimizer:
    parameter_groups = get_parameter_groups(
        context=context,
        model=model,
    )

    optimizer = Optimizer(
        config=(optimizer_config if optimizer_config is not None else context.config.optimizer),
        parameter_groups=parameter_groups,
        topology=context.topology,
        scale_backward_by_grad_acc=not context.config.transformer_architecture.umup.enable,
    )

    return optimizer
