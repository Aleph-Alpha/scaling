from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from pydantic import Field, model_validator

from scaling.core import (
    BaseConfig,
    BlendedDatasetConfig,
    LayerNormConfig,
    LearningRateSchedulerConfig,
    LoRaConfig,
    MaskedSoftmaxConfig,
    NormType,
    OptimizerConfig,
    ProfilerConfig,
    RelativePositionEmbeddingType,
    RunnerConfig,
    TopologyConfig,
    TrainerConfig,
)
from scaling.core.fp8 import FP8LinearConfig
from scaling.core.logging import LoggerConfig


class Precision(Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"

    @property
    def dtype(self) -> torch.dtype:
        if self == Precision.FLOAT16:
            return torch.float16
        elif self == Precision.BFLOAT16:
            return torch.bfloat16
        elif self == Precision.FLOAT32:
            return torch.float32
        else:
            raise NotImplementedError


class MLPType(Enum):
    DEFAULT = "default"
    SWIGLU = "swiglu"


class LossFunctionType(str, Enum):
    CROSS_ENTROPY_LOSS = "cross_entropy_loss"
    CONTRASTIVE_LOSS = "contrastive_loss"


class BaseLossFunctionConfig(BaseConfig):
    """Base configuration class for loss functions."""

    loss_type: LossFunctionType = Field(description="Type of loss function")


class CrossEntropyLossFunctionConfig(BaseLossFunctionConfig):
    loss_type: Literal[LossFunctionType.CROSS_ENTROPY_LOSS] = Field(LossFunctionType.CROSS_ENTROPY_LOSS)


class ContrastiveLossFunctionConfig(BaseLossFunctionConfig):
    loss_type: Literal[LossFunctionType.CONTRASTIVE_LOSS] = Field(LossFunctionType.CONTRASTIVE_LOSS)

    number_of_hard_negatives: int = Field(
        1,
        description="If Loss is Contrastive, this constrains the number of hard negatives per positive sample",
    )
    use_instructions: bool = Field(
        True,
        description="If True, we will prepend the sentence to embed with an instruction if available",
    )
    query_side_only: bool = Field(
        False,
        description="If use_instructions is true, this will make sure only the first query receives an instruction.",
    )
    scale: int = Field(1, description="Scale (divide) cosine scores by factor")

    log_verbose_metrics: bool = Field(
        True,
        description="Will compute and log all metrics relevant for Embedding \
        finetuning. Otherwise only loss is reported",
    )


class TrainingGroupConfig(BaseConfig):
    group_name: str = Field(description="Name of the group.")

    parameters_include: list[str] | None = Field(
        None, description="Regex pattern of parameters to be included in this param group"
    )
    parameters_exclude: list[str] = Field(
        [], description="Regex pattern of parameters to be excluded in this param group"
    )

    weight_decay: float = Field(0.0001, description="")

    independent_weight_decay: bool = Field(False, description="enables independent weight decay")

    learning_rate_scheduler: LearningRateSchedulerConfig = Field(LearningRateSchedulerConfig(), description="")

    @model_validator(mode="after")
    def check_no_overlap(self) -> "TrainingGroupConfig":
        if self.parameters_include is not None:
            include_set = set(self.parameters_include)
            exclude_set = set(self.parameters_exclude)
            overlap = include_set.intersection(exclude_set)
            if overlap:
                raise ValueError(
                    "No overlap is allowed between parameters_include and parameters_exclude. "
                    f"Found the following keys in both lists: {', '.join(overlap)}. "
                )
        return self


class TrainingConfig(BaseConfig):
    allow_missing_params_in_optimizer: bool = Field(
        False,
        description="if set to True, allows to freeze model parameters during training, e.g. for finetuning scenarios",
    )

    use_deterministic_torch_algorithms: bool = Field(
        False, description="Operations will use deterministic algorithms on torch"
    )

    loss_function_config: ContrastiveLossFunctionConfig | CrossEntropyLossFunctionConfig = Field(
        CrossEntropyLossFunctionConfig(),
        description="Defines which loss is used in the training",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_loss_function_config(cls, values: dict[str, Any]) -> dict[str, Any]:
        loss_function_config = values.get("loss_function_config")
        if loss_function_config is not None:
            loss_type = loss_function_config.get("loss_type")
            if loss_type == LossFunctionType.CONTRASTIVE_LOSS:
                values["loss_function_config"] = ContrastiveLossFunctionConfig(**loss_function_config).as_dict()
            elif loss_type == LossFunctionType.CROSS_ENTROPY_LOSS:
                values["loss_function_config"] = CrossEntropyLossFunctionConfig(**loss_function_config).as_dict()
        return values


class BitfitBiasConfig(BaseConfig):
    name: str = Field(description="")
    version: str = Field(
        default=".unknown.",
        description="""
        We use this version to report the `model_version` to our API users in the inference stack.
        Each softprompt should have a version.
        For the same transformer architecture the version should alpha-numerically increase with newer
        training states.
        When loading adapters/biases/softprompts then we resolve the version by taking the alpha-
        numerically largest version across base model and all adapters/biases/softprompts.
        That's also why we use a default version that starts with a full stop.""",
    )


class SoftpromptConfig(BaseConfig):
    name: str = Field(description="")
    n_tokens: int = Field(description="")
    version: str = Field(
        default=".unknown.",
        description="""
        We use this version to report the `model_version` to our API users in the inference stack.
        Each softprompt should have a version.
        For the same transformer architecture the version should alpha-numerically increase with newer
        training states.
        When loading adapters/biases/softprompts then we resolve the version by taking the alpha-
        numerically largest version across base model and all adapters/biases/softprompts.
        That's also why we use a default version that starts with a full stop.""",
    )


class AdapterConfig(BaseConfig):
    name: str = Field(description="")
    attention_downsampling_factor: float | None = Field(None, description="")
    mlp_downsampling_factor: float | None = Field(None, description="")
    init_std: float = Field(1.0e-5, description="")
    version: str = Field(
        default=".unknown.",
        description="""
        We use this version to report the `model_version` to our API users in the inference stack.
        Each adapter should have a version.
        For the same transformer architecture the version should alpha-numerically increase with newer
        training states.
        When loading adapters/biases/softprompts then we resolve the version by taking the alpha-
        numerically largest version across base model and all adapters/biases/softprompts.
        That's also why we use a default version that starts with a full stop.""",
    )


class PoolingMethod(str, Enum):
    MEAN = "mean"
    LAST_TOKEN = "last_token"
    WEIGHTED_MEAN = "weighted_mean"


class EmbeddingHeadConfig(BaseConfig):
    name: str = Field(description="")
    proj_layers: list[int] | None = Field(description="List of projection layers")
    pooling: PoolingMethod = Field(default=PoolingMethod.WEIGHTED_MEAN, description="")


class UMuPConfig(BaseConfig):
    enable: bool = Field(False, description="enables u-mup")
    allow_non_umup_params: bool = Field(
        False,
        description="""
                                        allows for the model to not have all params in umup,
                                        e.g. for adapter finetuning a umup model""",
    )
    normalize_depth_to_num_layers: bool = Field(
        False,
        description="""
                                                If True, uses number of transformer layers as depth to calculate
                                                depth-mup factors (actual residual depth is 2*num_layers)""",
    )
    attn_mult: float = Field(1.0, description="attention softmax temperature")
    act_mult: float = Field(1.0, description="temperature for mlp nonlinearity")
    residual_mult: float = Field(1.0, description="overall residual strength")
    residual_attn_ratio: float = Field(1.0, description="ratio of attn to mlp residual strength")
    loss_mult: float = Field(1.0, description="cross-entropy softmax temperature")


class TransformerArchitectureConfig(BaseConfig):
    """
    Transformer architecture config object containing non-mutable (constant) architecture specific configurations
    """

    vocab_size: int = Field(
        0,
        description="Size of the vocabulary before padding; this matches the vocab size of the tokenizer",
    )

    vocab_file: Path | None = Field(
        None,
        description="",
    )

    hidden_size: int = Field(
        0,
        description="Transformer hidden size.",
    )

    num_layers: int = Field(
        0,
        description="Number of transformer layers",
    )

    num_attention_heads: int = Field(
        0,
        description="Number of attention heads",
    )

    num_local_attention_heads: int = Field(
        0,
        description="Number of attention heads",
    )

    local_attention_window_size: int | None = Field(None, description="The size of the local attention window")

    rotary_embedding_base: int = Field(
        10000,
        description="",
    )

    rotary_percentage: float = Field(
        1.0,
        description="Percentage to apply rotary embeddings over. It applies the rotary embeddings to the first "
        "`(hidden_size // attention_heads) * rotary_percentage` dimensions",
    )

    sequence_length: int = Field(
        2048,
        description="Sequence length in number of tokens in one sample on which a train job is run; at inference time "
        "the sequence length of a sample should (usually) not be exceeded.",
    )

    norm_type: NormType = Field(
        NormType.LAYERNORM,
        description="choose between 'layernorm' and 'rms'",
    )

    relative_position_embedding_type: RelativePositionEmbeddingType = Field(
        RelativePositionEmbeddingType.ROTARY,
        description="choose relative position embeddings among 'none', 'rotary', 'rotary_complex'",
    )

    mlp_type: MLPType = Field(
        MLPType.DEFAULT,
        description="choose between 'default' and 'swiglu'",
    )

    mlp_factor: float = Field(
        4.0,
        description="expansion factor for mlp hidden layer",
    )

    attention_bias: bool = Field(
        True,
        description="add bias terms to attention components",
    )

    attention_qkv_in_one: bool = Field(
        True,
        description="whether query key value should be executed as one matrix multiplication at once",
    )

    attention_num_kv_heads: int | None = Field(
        None,
        description="number kv heads, if it differs from query heads",
    )

    attention_use_matmul: bool = Field(
        False,
        description="use torch.matmul instead of torch.baddbmm",
    )

    reset_attention_mask: bool = Field(True, description="resets the attention mask between documents in a sequence")

    reset_position_ids: bool = Field(True, description="resets the position ids between documents in a sequence")

    mlp_bias: bool = Field(
        True,
        description="add bias terms to mlp",
    )

    key_query_norm: bool = Field(
        False,
        description="add a norm for key and query scores",
    )

    weight_tying: bool = Field(
        True,
        description="",
    )

    masked_softmax: MaskedSoftmaxConfig = Field(MaskedSoftmaxConfig(), description="")

    layernorm: LayerNormConfig = Field(LayerNormConfig(), description="")

    precision: Precision = Field(Precision.FLOAT32, description="")

    fp8_config_attention: FP8LinearConfig | None = Field(
        None,
        description="FP8 settings for matmuls in attention layers. No FP8 is used if set to `None`.",
    )

    fp8_config_attention_dense_out: FP8LinearConfig | None = Field(
        None,
        description="FP8 settings for the dense out matmul in attention layers. No FP8 is used if set to `None`.",
    )

    fp8_config_mlp: FP8LinearConfig | None = Field(
        None,
        description="FP8 settings for matmuls in MLPs. No FP8 is used if set to `None`.",
    )

    fp8_config_mlp_dense_out: FP8LinearConfig | None = Field(
        None,
        description="FP8 settings for the dense out matmul in MLPs. No FP8 is used if set to `None`.",
    )

    fp8_config_lm_head: FP8LinearConfig | None = Field(
        None,
        description="FP8 settings for LM head matmul. No FP8 is used if set to `None`.",
    )

    dropout_embedding: float = Field(0.0, description="dropout applied after the embedding layer", ge=0.0, le=1.0)

    dropout_attention_probs: float = Field(
        0.0,
        description="dropout applied to the attention probabilities",
        ge=0.0,
        le=1.0,
    )

    dropout_after_attention: float = Field(0.0, description="dropout applied after the embedding layer", ge=0.0, le=1.0)

    dropout_after_mlp: float = Field(0.0, description="dropout applied after the embedding layer", ge=0.0, le=1.0)

    bitfit_bias_config: BitfitBiasConfig | None = Field(
        None,
        description="Config for a bias that will be finetuned.",
    )

    finetunable_token_ids: list[int] = Field(
        list(),
        description="Number of extra tokens for fine-tuned embedding. Set to 0 to deactivate fine-tuned embedding.",
    )

    image_encoder: bool = Field(
        False,
        description="add image encoder to input embedding",
    )

    dropout_image_encoder: float = Field(
        0.0,
        description="dropout applied after the image encoder projection",
        ge=0.0,
        le=1.0,
    )

    softprompt_config: SoftpromptConfig | None = Field(
        None,
        description="",
    )

    adapter_config: AdapterConfig | None = Field(
        None,
        description="",
    )

    lora_config: LoRaConfig | None = Field(None, description="creates LoRa finetuning configuration")

    embedding_head_config: EmbeddingHeadConfig | None = Field(None, description="")

    lm_head: bool = Field(
        True,
        description="Flag for including a LMHead at the end for next Token prediction. \
                        You can either set embedding_head_config for an embedding_head or lm_head, never both",
    )

    causal: bool = Field(True, description="Make attention layers causal.")

    umup: UMuPConfig = Field(
        UMuPConfig(), description="Configuration for unit-scaled maximum update parametrization (UMuP)"
    )


class DataConfig(BaseConfig):
    """
    Data config object containing non-mutable (constant) dataset specific configurations
    """

    legacy_dataset: bool = Field(
        False,
        description="Use the legacy dataset implementation",
    )

    load_mmap_index_to_memory: bool = Field(
        False,
        description="Load the memory map index of the main memory map of the dataset into RAM; this reduces the number "
        "of open file handles at the cost of increased memory use. Does not work for legacy_dataset.",
    )

    use_mmap: bool = Field(
        True,
        description="Use memory maps instead of regular file operations to read data",
    )

    load_data_item_mmap_index_to_memory: bool = Field(
        False,
        description="Load the memory map index of the memory map used to combine source documents into batch items of "
        "the dataset into RAM; this reduces the number of open file handles at the cost of increased "
        "memory use. Does not work for legacy_dataset.",
    )

    finetuning_dataset: bool = Field(
        False,
        description="Use the finetuning text dataset implementation",
    )

    finetuning_chat_dataset: bool = Field(
        False,
        description="Use the finetuning chat dataset implementation",
    )

    finetuning_dataset_memory_map: bool = Field(
        False,
        description="Indicate whether the finetuning dataset is a memory map dataset",
    )

    embedding_dataset: bool = Field(
        False,
        description="Use the finetuning text dataset implementation",
    )

    embedding_dataset_memory_map: bool = Field(
        False,
        description="Use the finetuning text dataset implementation",
    )

    data_prefixes: list[Path] | None = Field(
        None,
        description="Training data prefixes pointing to tokenized memory map",
    )

    validation_data_prefixes: list[Path] | None = Field(
        None,
        description="Validation data prefixes pointing to tokenized memory map",
    )

    blended_dataset: BlendedDatasetConfig = Field(
        BlendedDatasetConfig(),
        description="Configuration for the blended dataset",
    )

    only_full_sequences: bool = Field(
        False,
        description="Only uses sequences witch fully fill up the context."
        "This Option is only available for none legacy datasets.",
    )

    allow_incomplete_sequences_every_n: int = Field(
        0,
        description="This option is only available used when 'only_full_sequences' is set to True."
        "It will allow every n-th sequence to not be fully filled.",
    )


class TransformerConfig(BaseConfig):
    version: str = Field(
        default=".unknown.",
        description="""
        We use this version to report the `model_version` to our API users in the inference stack.
        Each checkpoint should have a version.
        The same goes for separately deployed adapters.
        For the same transformer architecture the version should alpha-numerically increase with newer
        training states.
        When loading adapters/biases/softprompts then we resolve the version by taking the alpha-
        numerically largest version across base model and all adapters/biases/softprompts.
        That's also why we use a default version that starts with a full stop.""",
    )

    runner: RunnerConfig = Field(
        RunnerConfig(),
        description="",
    )

    logger: LoggerConfig = Field(
        LoggerConfig(),
        description="",
    )

    topology: TopologyConfig = Field(
        TopologyConfig(  # type: ignore[call-arg]
            model_parallel_size=1,
            pipe_parallel_size=1,
            data_parallel_size=1,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
        ),
        description="",
    )
    optimizer: OptimizerConfig = Field(OptimizerConfig(), description="")

    training: TrainingConfig = Field(TrainingConfig(), description="")

    training_groups: list[TrainingGroupConfig] = Field(default=[], description="")

    trainer: TrainerConfig = Field(TrainerConfig(), description="")

    profiler: ProfilerConfig = Field(ProfilerConfig(), description="")

    transformer_architecture: TransformerArchitectureConfig = Field(TransformerArchitectureConfig(), description="")

    data: DataConfig = Field(DataConfig(), description="")

    determined_experiment_id: int | None = Field(
        None,
        description="determined experiment id, necessary to recover checkpoint paths",
    )

    determined_trial_id: int | None = Field(
        None,
        description="determined trial id, necessary to recover checkpoint paths",
    )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], overwrite_values: dict | None = None) -> "TransformerConfig":
        # collect separate_file_for_parameters from finetunable parameters
        if is_legacy_config(d):
            d = convert_legacy_config(d)

        separate_file_for_parameters = set()
        if "transformer_architecture" in d:
            if (
                "bitfit_bias_config" in d["transformer_architecture"]
                and d["transformer_architecture"]["bitfit_bias_config"] is not None
            ):
                separate_file_for_parameters.add(f"bias_{d['transformer_architecture']['bitfit_bias_config']['name']}")

            if (
                "adapter_config" in d["transformer_architecture"]
                and d["transformer_architecture"]["adapter_config"] is not None
            ):
                separate_file_for_parameters.add(f"adapter_{d['transformer_architecture']['adapter_config']['name']}")

            if (
                "softprompt_config" in d["transformer_architecture"]
                and d["transformer_architecture"]["softprompt_config"] is not None
            ):
                separate_file_for_parameters.add(
                    f"softprompt_{d['transformer_architecture']['softprompt_config']['name']}"
                )

        d_alt = dict(deepcopy(d))
        # add separate_file_for_parameters to config
        if len(separate_file_for_parameters) > 0:
            if "trainer" not in d_alt:
                d_alt["trainer"] = dict()
            d_alt["trainer"]["separate_file_for_parameters"] = list(separate_file_for_parameters)

        return super().from_dict(d_alt, overwrite_values=overwrite_values)

    @model_validator(mode="after")
    def check_training_group_names(self) -> "TransformerConfig":
        if len(self.training_groups) > 0:
            all_names = [group.group_name for group in self.training_groups]
            duplicate_names = set(name for name in all_names if all_names.count(name) > 1)

            if len(duplicate_names) > 0:
                raise ValueError(
                    f"Please give each training parameter group a unique name.\n"
                    f"The following names showed up more than one time: {duplicate_names}"
                )

        return self


def is_legacy_config(config: Mapping[str, Any]) -> bool:
    return "luminous_architecture" in config


def convert_legacy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    new_config = dict(config)
    if "luminous_architecture" in config:
        new_config["transformer_architecture"] = new_config["luminous_architecture"]
        del new_config["luminous_architecture"]
    return new_config
