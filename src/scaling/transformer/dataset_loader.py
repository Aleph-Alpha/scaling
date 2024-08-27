from pathlib import Path
from typing import Sequence

from scaling.core import (
    BaseDataset,
)
from scaling.transformer import (
    FinetuningChatDataset,
    FinetuningTextDataset,
    TextDataset,
    TransformerConfig,
)
from scaling.transformer.context import TransformerArchitectureConfig
from scaling.transformer.context.config import DataConfig
from scaling.transformer.tokenizer import Tokenizer, load_tokenizers


def load_datasets(
    data_config: DataConfig, architecture_config: TransformerArchitectureConfig, config: TransformerConfig
) -> tuple[Sequence[BaseDataset], Sequence[BaseDataset]]:
    assert data_config.data_prefixes is not None, "path to data prefix not defined in Transformer context"
    if data_config.finetuning_dataset:
        return _load_finetuning_datasets(data_config, architecture_config, config)
    if data_config.finetuning_chat_dataset:
        return _load_chat_finetuning_datasets(data_config, architecture_config, config)
    return _load_text_datasets(data_config, architecture_config, config)


def _load_text_datasets(
    data_config: DataConfig, architecture_config: TransformerArchitectureConfig, config: TransformerConfig
) -> tuple[list[TextDataset], list[TextDataset]]:
    assert data_config.data_prefixes is not None, "path to data prefix not defined in Transformer context"
    datasets = _extract_text_datasets(data_config.data_prefixes, data_config, architecture_config, config)
    validation_datasets = []
    if data_prefixes := data_config.validation_data_prefixes:
        validation_datasets = _extract_text_datasets(data_prefixes, data_config, architecture_config, config)
    return datasets, validation_datasets


def _load_chat_finetuning_datasets(
    data_config: DataConfig, architecture_config: TransformerArchitectureConfig, config: TransformerConfig
) -> tuple[list[FinetuningChatDataset], list[FinetuningChatDataset]]:
    assert data_config.data_prefixes is not None, "path to data prefix not defined in Transformer context"
    datasets = _extract_chat_finetuning_dataset(data_config.data_prefixes, architecture_config, config)
    validation_datasets = []
    if data_prefixes := data_config.validation_data_prefixes:
        validation_datasets = _extract_chat_finetuning_dataset(data_prefixes, architecture_config, config)
    return datasets, validation_datasets


def _load_finetuning_datasets(
    data_config: DataConfig, architecture_config: TransformerArchitectureConfig, config: TransformerConfig
) -> tuple[list[FinetuningTextDataset], list[FinetuningTextDataset]]:
    assert data_config.use_mmap, "Finetuning is currently only supported with use_mmap set to true."
    assert data_config.data_prefixes is not None, "path to data prefix not defined in Transformer context"
    dataset_memory_map = data_config.finetuning_dataset_memory_map
    datasets = _extract_finetuning_text_dataset(
        data_config.data_prefixes, architecture_config, dataset_memory_map, config.trainer.seed
    )
    validation_datasets = []
    if data_prefixes := data_config.validation_data_prefixes:
        validation_datasets = _extract_finetuning_text_dataset(
            data_prefixes, architecture_config, dataset_memory_map, config.trainer.seed
        )
    return datasets, validation_datasets


def _extract_chat_finetuning_dataset(
    data_prefixes: list[Path], architecture_config: TransformerArchitectureConfig, config: TransformerConfig
) -> list[FinetuningChatDataset]:
    tokenizer, tokenizer_no_prefix_space = _load_tokenizer(architecture_config)
    softprompt_n_tokens = _get_softprompt_n_tokens(architecture_config)
    return [
        FinetuningChatDataset(
            data_path=data_prefix,
            sequence_length=architecture_config.sequence_length,
            seed=config.trainer.seed,
            softprompt_n_tokens=softprompt_n_tokens,
            tokenizer=tokenizer,
            tokenizer_no_prefix_space=tokenizer_no_prefix_space,
        )
        for data_prefix in data_prefixes
    ]


def _extract_text_datasets(
    data_prefixes: list[Path],
    data_config: DataConfig,
    architecture_config: TransformerArchitectureConfig,
    config: TransformerConfig,
) -> list[TextDataset]:
    return [
        TextDataset(
            data_prefix=data_prefix,
            sequence_length=architecture_config.sequence_length,
            seed=config.trainer.seed,
            legacy_dataset=data_config.legacy_dataset,
            load_mmap_index_to_memory=data_config.load_mmap_index_to_memory,
            load_data_item_mmap_index_to_memory=data_config.load_data_item_mmap_index_to_memory,
            only_full_sequences=data_config.only_full_sequences,
            allow_incomplete_sequences_every_n=data_config.allow_incomplete_sequences_every_n,
            use_mmap=data_config.use_mmap,
        )
        for data_prefix in data_prefixes
    ]


def _extract_finetuning_text_dataset(
    data_prefixes: list[Path], architecture_config: TransformerArchitectureConfig, memory_map_dataset: bool, seed: int
) -> list[FinetuningTextDataset]:
    tokenizer, tokenizer_no_prefix_space = _load_tokenizer(architecture_config)
    softprompt_n_tokens = _get_softprompt_n_tokens(architecture_config)
    return [
        FinetuningTextDataset(
            data_prefix=data_prefix,
            sequence_length=architecture_config.sequence_length,
            seed=seed,
            softprompt_n_tokens=softprompt_n_tokens,
            tokenizer=tokenizer,
            tokenizer_no_prefix_space=tokenizer_no_prefix_space,
            memory_map_dataset=memory_map_dataset,
        )
        for data_prefix in data_prefixes
    ]


def _load_tokenizer(architecture_config: TransformerArchitectureConfig) -> tuple[Tokenizer, Tokenizer]:
    assert architecture_config.vocab_file is not None, "vocab_file needs to be to load the vocabulary file."
    return load_tokenizers(architecture_config.vocab_file)


def _get_softprompt_n_tokens(architecture_config: TransformerArchitectureConfig) -> int:
    if architecture_config.softprompt_config is None:
        return 0

    return architecture_config.softprompt_config.n_tokens
