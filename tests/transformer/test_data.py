from pathlib import Path

import pytest

from scaling.transformer.data import (
    FinetuningChatDataset,
    FinetuningTextDataset,
    TextDataset,
)
from scaling.transformer.tokenizer import load_tokenizers


@pytest.mark.transformer
@pytest.mark.parametrize("load_mmap_index_to_memory", [True, False])
@pytest.mark.parametrize("load_data_item_mmap_index_to_memory", [True, False])
def test_text_dataset_instantiation(load_mmap_index_to_memory: bool, load_data_item_mmap_index_to_memory: bool):
    dataset = TextDataset(
        data_prefix=Path(__file__).parents[0] / "files" / "dataset" / "data",
        sequence_length=8,
        seed=42,
        load_mmap_index_to_memory=load_mmap_index_to_memory,
        load_data_item_mmap_index_to_memory=load_data_item_mmap_index_to_memory,
    )

    batch_before_sync = dataset.collate(batch=[dataset[0], dataset[1]])
    _ = TextDataset.sync_batch_to_model_parallel(topology=None, batch=batch_before_sync)


@pytest.mark.transformer
@pytest.mark.parametrize("load_mmap_index_to_memory", [True, False])
@pytest.mark.parametrize("load_data_item_mmap_index_to_memory", [True, False])
@pytest.mark.parametrize("allow_incomplete_sequences_every_n", [0, 4, 8, 256])
def test_text_dataset_instantiation_max_seq(
    load_mmap_index_to_memory: bool,
    load_data_item_mmap_index_to_memory: bool,
    allow_incomplete_sequences_every_n: int,
):
    for r in (Path(__file__).parents[0] / "files" / "dataset").glob(
        "data_index_cache_decoder_dataset_seed_42_seq_len_4.*"
    ):
        r.unlink(missing_ok=True)

    for r in (Path(__file__).parents[0] / "files" / "dataset").glob(
        f"data_index_cache_decoder_dataset_seed_42_seq_len_4_only_full_sequences_allow_incomplete_sequences_every_n_{allow_incomplete_sequences_every_n}.*"
    ):
        r.unlink(missing_ok=True)

    dataset = TextDataset(
        data_prefix=Path(__file__).parents[0] / "files" / "dataset" / "data",
        sequence_length=4,
        seed=42,
        load_mmap_index_to_memory=load_mmap_index_to_memory,
        load_data_item_mmap_index_to_memory=load_data_item_mmap_index_to_memory,
    )

    batch_before_sync = dataset.collate(batch=[dataset[0], dataset[1]])
    _ = TextDataset.sync_batch_to_model_parallel(topology=None, batch=batch_before_sync)

    has_eos_in_body = False

    for idx in range(len(dataset)):
        if 0 in dataset[idx].token_ids[:-1].tolist():
            has_eos_in_body = True

    assert has_eos_in_body, "The following test assumes that this setup would have EOS token in the text body."

    dataset = TextDataset(
        data_prefix=Path(__file__).parents[0] / "files" / "dataset" / "data",
        sequence_length=4,
        seed=42,
        load_mmap_index_to_memory=load_mmap_index_to_memory,
        load_data_item_mmap_index_to_memory=load_data_item_mmap_index_to_memory,
        only_full_sequences=True,
        allow_incomplete_sequences_every_n=allow_incomplete_sequences_every_n,
    )

    batch_before_sync = dataset.collate(batch=[dataset[0], dataset[1]])
    _ = TextDataset.sync_batch_to_model_parallel(topology=None, batch=batch_before_sync)

    has_eos_in_body_count = 0

    for idx in range(len(dataset)):
        if 0 in dataset[idx].token_ids[:-1].tolist():
            has_eos_in_body_count += 1

    if allow_incomplete_sequences_every_n == 0:
        assert has_eos_in_body_count == 0
    else:
        assert (has_eos_in_body_count / len(dataset)) <= (1 / allow_incomplete_sequences_every_n)

    print("")


@pytest.mark.transformer
def test_finetuning_text_dataset_instantiation():
    tokenizer, tokenizer_no_prefix_space = load_tokenizers(Path(__file__).parents[0] / "files" / "alpha-001-128k.json")
    dataset = FinetuningTextDataset(
        data_prefix=Path(__file__).parents[0] / "files" / "dataset" / "finetuning.json",
        sequence_length=512,
        seed=42,
        softprompt_n_tokens=2,
        tokenizer=tokenizer,
        tokenizer_no_prefix_space=tokenizer_no_prefix_space,
    )
    batch_before_sync = dataset.collate(batch=[dataset[0], dataset[1]])
    _ = FinetuningTextDataset.sync_batch_to_model_parallel(topology=None, batch=batch_before_sync)


@pytest.mark.transformer
def test_finetuning_text_dataset_instantiation_indexed_dataset(tmp_path: Path):
    data_prefix = tmp_path / "dataset"

    tokenizer, tokenizer_no_prefix_space = load_tokenizers(Path(__file__).parents[0] / "files" / "alpha-001-128k.json")

    FinetuningTextDataset.convert_jsonl(
        jsonl_file=Path(__file__).parents[0] / "files" / "dataset" / "finetuning.jsonl",
        tokenizer=tokenizer,
        tokenizer_no_prefix_space=tokenizer_no_prefix_space,
        out_prefix_path=data_prefix,
    )
    dataset = FinetuningTextDataset(
        data_prefix=data_prefix,
        memory_map_dataset=True,
        sequence_length=512,
        seed=42,
        softprompt_n_tokens=2,
        tokenizer=tokenizer,
        tokenizer_no_prefix_space=tokenizer_no_prefix_space,
    )
    batch_before_sync = dataset.collate(batch=[dataset[0], dataset[1]])
    _ = FinetuningTextDataset.sync_batch_to_model_parallel(topology=None, batch=batch_before_sync)


@pytest.mark.transformer
def test_finetuning_chat_dataset_instantiation_indexed_dataset(tmp_path: Path):
    data_path = Path(__file__).parents[0] / "files" / "dataset" / "finetuning_chat.jsonl"

    tokenizer, tokenizer_no_prefix_space = load_tokenizers(Path(__file__).parents[0] / "files" / "alpha-001-128k.json")

    dataset = FinetuningChatDataset(
        data_path=data_path,
        sequence_length=512,
        seed=42,
        softprompt_n_tokens=2,
        tokenizer=tokenizer,
        tokenizer_no_prefix_space=tokenizer_no_prefix_space,
    )
    batch_before_sync = dataset.collate(batch=[dataset[0], dataset[1]])
    _ = FinetuningChatDataset.sync_batch_to_model_parallel(topology=None, batch=batch_before_sync)


@pytest.mark.transformer
def test_legacy_text_dataset_instantiation():
    # Delete index if existing
    # We want to make sure that the index can be computed on the fly
    index_files = list((Path(__file__).parents[0].absolute() / "files" / "dataset" / "legacy").glob("*_seed_*"))
    for index_file in index_files:
        index_file.unlink()

    dataset = TextDataset(
        data_prefix=Path(__file__).parents[0] / "files" / "dataset" / "legacy" / "enron_text_document_100",
        sequence_length=8,
        seed=42,
        legacy_dataset=True,
    )
    batch_before_sync = dataset.collate(batch=[dataset[0], dataset[1]])
    _ = TextDataset.sync_batch_to_model_parallel(topology=None, batch=batch_before_sync)

    # Delete index if existing
    # We want to make sure that the index can be computed on the fly
    index_files = list((Path(__file__).parents[0].absolute() / "files" / "dataset" / "legacy").glob("*_seed_*"))
    for index_file in index_files:
        index_file.unlink()
