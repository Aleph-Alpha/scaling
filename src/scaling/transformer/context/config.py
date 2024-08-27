from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import torch
from pydantic import Field, model_validator
from typing_extensions import Self

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


class TrainingConfig(BaseConfig, populate_by_name=True):
    weight_decay: float = Field(0.0001, description="")
    finetune: bool = Field(False, description="activate finetuning mode")
    finetunable_parameters: list[str] = Field([], description="pattern of parameters to be included in finetuning")
    parameters_exclude: list[str] = Field([], description="pattern of parameters to be excluded in training")
    use_separate_lr_on_embeddings: bool = Field(
        False, description="", alias="use_seperate_lr_on_embeddings"
    )  # Alias is necessary for supporting legacy configs

    use_deterministic_torch_algorithms: bool = Field(
        False, description="Operations will use deterministic algorithms on torch"
    )

    @model_validator(mode="after")
    def check_finetune(self) -> Self:
        if self.finetune and not self.finetunable_parameters:
            raise ValueError("Can not set finetune when finetunable_parameters is empty")
        elif not self.finetune and self.finetunable_parameters:
            raise ValueError("Can not set finetunable_parameters when finetune is False")
        return self


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


class EmbeddingHeadConfig(BaseConfig):
    name: str = Field(description="")
    proj_layers: list[int] = Field(description="")


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

    causal: bool = Field(True, description="Make attention layers causal.")


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

    learning_rate_scheduler: LearningRateSchedulerConfig = Field(LearningRateSchedulerConfig(), description="")

    embedding_learning_rate_scheduler: LearningRateSchedulerConfig = Field(
        LearningRateSchedulerConfig(), description=""
    )

    training: TrainingConfig = Field(TrainingConfig(), description="")

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
