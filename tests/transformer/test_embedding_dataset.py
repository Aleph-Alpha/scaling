import os
from pathlib import Path

import pytest
import torch

from scaling.transformer.data.embedding_dataset import EmbeddingDataset
from scaling.transformer.tokenizer import Tokenizer


def mem_map_exists(load_path):
    return (
        os.path.exists(load_path.with_suffix(".bin"))
        and os.path.exists(load_path.with_suffix(".idx"))
        and os.path.exists(load_path.with_suffix(".meta.json"))
        and os.path.exists(load_path.with_suffix(".done"))
    )


@pytest.fixture
def run_embedding_dataset_test(path_to_files: Path, unigram_02_tokenizer: Tokenizer):
    def inner(
        cache_dir: Path,
        use_instructions: bool,
        query_side_only: bool,
        from_memory_map: bool,
        sequence_length: int,
        number_of_hard_negatives: int,
    ):
        tokenizer = unigram_02_tokenizer
        if from_memory_map:
            if use_instructions:
                load_path = path_to_files / "dataset/embedding_dataset_instructed.jsonl"
            else:
                load_path = path_to_files / "dataset/embedding_dataset_non_instructed.jsonl"

            if not mem_map_exists(cache_dir / load_path.stem):
                EmbeddingDataset.jsonl_to_embedding_mmap(
                    load_path, tokenizer=tokenizer, out_prefix_path=cache_dir / load_path.stem
                )
            load_path = cache_dir / load_path.stem
        else:
            if use_instructions:
                load_path = path_to_files / "dataset/embedding_dataset_instructed.jsonl"
            else:
                load_path = path_to_files / "dataset/embedding_dataset_non_instructed.jsonl"

        dataset = EmbeddingDataset(
            data_path=load_path,
            tokenizer=tokenizer,
            memory_map_dataset=from_memory_map,
            use_instruction=use_instructions,
            number_of_hard_negatives=number_of_hard_negatives,
            seed=42,
            sequence_length=sequence_length,
            query_side_only=query_side_only,
        )

        for i in range(len(dataset)):
            sample = dataset[i]
            assert (
                sample.input_token_ids.shape[-1] == sequence_length
            ), f"Sequence length {sequence_length} did not match DatasetItems sequence \
                length {sample.input_token_ids.shape[-1]}"
            assert (
                sample.input_token_ids.shape == sample.loss_weights.shape
            ), f"Shape of input_token_ids: {sample.input_token_ids.shape} did not match up \
                with loss_weights shape {sample.input_token_ids.shape}"
            assert not torch.any(
                sample.loss_weights.sum(-1) == 0
            ), "Loss weights contained a sample of all zeros. This would leed to a nan in the loss."

        if use_instructions:
            for i in range(len(dataset)):
                sample = dataset[i]
                assert sample.loss_weights[0][0] == 0, "Loss mask did not mask query instruction"
                if not query_side_only:
                    assert sample.loss_weights[1][0] == 0, "Loss mask did not mask positive instruction"
                    if number_of_hard_negatives > 0:
                        assert sample.loss_weights[2][0] == 0, "Loss mask did not mask negative instruction"
        else:
            for i in range(len(dataset)):
                sample = dataset[i]
                assert not sample.loss_weights[0][0] == 0, "Loss mask was applied to a sample without instruction"
                if not query_side_only:
                    assert not sample.loss_weights[1][0] == 0, "Loss mask was applied to a sample without instruction"
                    if number_of_hard_negatives > 0:
                        assert (
                            not sample.loss_weights[2][0] == 0
                        ), "Loss mask was applied to a sample without instruction"

    return inner


@pytest.mark.embedding
@pytest.mark.parametrize("use_instructions", [True])
@pytest.mark.parametrize("from_memory_map", [True, False])
@pytest.mark.parametrize("query_side_only", [True, False])
@pytest.mark.parametrize("sequence_length", [128])
@pytest.mark.parametrize("number_of_hard_negatives", [1])
def test_try_load_instructed_from_non_instructed_file(
    tmp_path: Path,
    use_instructions: bool,
    query_side_only: bool,
    from_memory_map: bool,
    sequence_length: int,
    number_of_hard_negatives: int,
    unigram_02_tokenizer: Tokenizer,
    path_to_files: Path,
):
    tokenizer = unigram_02_tokenizer
    if from_memory_map:
        load_path = path_to_files / "dataset/embedding_dataset_non_instructed.jsonl"

        if not mem_map_exists(tmp_path / load_path.stem):
            EmbeddingDataset.jsonl_to_embedding_mmap(
                load_path, tokenizer=tokenizer, out_prefix_path=tmp_path / load_path.stem
            )
        load_path = tmp_path / load_path.stem
    else:
        load_path = path_to_files / "dataset/embedding_dataset_non_instructed.jsonl"

    with pytest.raises(AssertionError):
        dataset = EmbeddingDataset(
            data_path=load_path,
            tokenizer=tokenizer,
            memory_map_dataset=from_memory_map,
            use_instruction=use_instructions,
            number_of_hard_negatives=number_of_hard_negatives,
            seed=42,
            sequence_length=sequence_length,
            query_side_only=query_side_only,
        )
        dataset[0]


@pytest.mark.embedding
@pytest.mark.parametrize("use_instructions", [False])
@pytest.mark.parametrize("from_memory_map", [True, False])
@pytest.mark.parametrize("query_side_only", [False])
@pytest.mark.parametrize("sequence_length", [128])
@pytest.mark.parametrize("number_of_hard_negatives", [0, 1])
def test_load_non_instructed_from_instructed_file(
    tmp_path: Path,
    use_instructions: bool,
    query_side_only: bool,
    from_memory_map: bool,
    sequence_length: int,
    number_of_hard_negatives: int,
    unigram_02_tokenizer: Tokenizer,
    path_to_files: Path,
):
    tokenizer = unigram_02_tokenizer
    if from_memory_map:
        load_path = path_to_files / "dataset/embedding_dataset_instructed.jsonl"

        if not mem_map_exists(tmp_path / load_path.stem):
            EmbeddingDataset.jsonl_to_embedding_mmap(
                load_path, tokenizer=tokenizer, out_prefix_path=tmp_path / load_path.stem
            )
        load_path = tmp_path / load_path.stem
    else:
        load_path = path_to_files / "dataset/embedding_dataset_instructed.jsonl"

    dataset = EmbeddingDataset(
        data_path=load_path,
        tokenizer=tokenizer,
        memory_map_dataset=from_memory_map,
        use_instruction=use_instructions,
        number_of_hard_negatives=number_of_hard_negatives,
        seed=42,
        sequence_length=sequence_length,
        query_side_only=query_side_only,
    )

    for i in range(len(dataset)):
        sample = dataset[i]
        assert (
            sample.loss_weights[0][0] != 0
        ), "Loss weights for query sample masked out initial token. This should not happen in uninstructed dataset."
        assert (
            sample.loss_weights[1][0] != 0
        ), "Loss weights for positive sample masked out initial token. This should not happen in uninstructed dataset."
        if number_of_hard_negatives > 0:
            assert sample.loss_weights[2][0] != 0, """Loss weights for negative sample masked out initial token.
            This should not happen in uninstructed dataset."""


@pytest.mark.embedding
@pytest.mark.parametrize("use_instructions", [True])
@pytest.mark.parametrize("from_memory_map", [True, False])
@pytest.mark.parametrize("query_side_only", [True, False])
@pytest.mark.parametrize("sequence_length", [128])
@pytest.mark.parametrize("number_of_hard_negatives", [2, 10])
def test_try_load_with_nr_hard_negatives_greater_one(
    tmp_path: Path,
    use_instructions: bool,
    query_side_only: bool,
    from_memory_map: bool,
    sequence_length: int,
    number_of_hard_negatives: int,
    run_embedding_dataset_test,
):
    with pytest.raises(NotImplementedError):
        run_embedding_dataset_test(
            cache_dir=tmp_path,
            use_instructions=use_instructions,
            query_side_only=query_side_only,
            from_memory_map=from_memory_map,
            sequence_length=sequence_length,
            number_of_hard_negatives=number_of_hard_negatives,
        )


@pytest.mark.embedding
@pytest.mark.parametrize("query_side_only", [True, False])
@pytest.mark.parametrize("use_instructions", [True, False])
@pytest.mark.parametrize("from_memory_map", [True, False])
@pytest.mark.parametrize("sequence_length", [128, 2048])
@pytest.mark.parametrize("number_of_hard_negatives", [0, 1])
def test_embedding_dataset(
    tmp_path: Path,
    use_instructions: bool,
    query_side_only: bool,
    from_memory_map: bool,
    sequence_length: int,
    number_of_hard_negatives: int,
    run_embedding_dataset_test,
):
    if use_instructions and query_side_only:
        pytest.skip()

    if query_side_only and not use_instructions:
        pytest.skip()

    run_embedding_dataset_test(
        cache_dir=tmp_path,
        use_instructions=use_instructions,
        query_side_only=query_side_only,
        from_memory_map=from_memory_map,
        sequence_length=sequence_length,
        number_of_hard_negatives=number_of_hard_negatives,
    )


@pytest.mark.parametrize("use_instruction", [True, False])
@pytest.mark.parametrize("query_side_only", [True, False])
def test_is_valid_sample(
    unigram_02_tokenizer: Tokenizer,
    path_to_files: Path,
    use_instruction: bool,
    query_side_only: bool,
    batch_size: int = 16,
    sequence_length: int = 128,
):
    if not use_instruction and query_side_only:
        pytest.skip("Skipping test as it is an invalid combination.")

    tokenizer = unigram_02_tokenizer
    load_path = path_to_files / "dataset/embedding_dataset_instructed.jsonl"

    dataset = EmbeddingDataset(
        data_path=load_path,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        use_instruction=use_instruction,
        number_of_hard_negatives=1,
        seed=42,
        query_side_only=query_side_only,
    )

    def generate_sample(query_len_offset=0, pos_len_offset=0, neg_len_offset=0):
        return {
            "query_token_ids": torch.randint(low=0, high=100, size=(batch_size, sequence_length)),
            "positive_token_ids": torch.randint(low=0, high=100, size=(batch_size, sequence_length)),
            "negative_token_ids": [torch.randint(low=0, high=100, size=(batch_size, sequence_length))],
            "query_instruction_length": sequence_length + query_len_offset,
            "positive_instruction_length": sequence_length + pos_len_offset,
            "negative_instruction_length": [sequence_length + neg_len_offset],
        }

    invalid_lenses_sample = generate_sample(query_len_offset=0, pos_len_offset=0, neg_len_offset=0)
    valid_sample = generate_sample(query_len_offset=-1, pos_len_offset=-1, neg_len_offset=-1)
    invalid_pos_negative_lenses_sample = generate_sample(query_len_offset=-1, pos_len_offset=0, neg_len_offset=0)

    if use_instruction:
        assert not dataset._is_valid_sample(
            invalid_lenses_sample
        ), "Sample should not be valid, as instruction lengths match the sequence length."
        assert dataset._is_valid_sample(valid_sample), "Sample should be valid."

        if query_side_only:
            assert dataset._is_valid_sample(
                invalid_pos_negative_lenses_sample
            ), "Sample should be valid when only query instruction length is checked."
        else:
            assert not dataset._is_valid_sample(
                invalid_pos_negative_lenses_sample
            ), "Sample should not be valid, as positive/negative instruction lengths match the sequence length."
    else:
        assert dataset._is_valid_sample(valid_sample), "Sample should be valid without instructions."
        assert dataset._is_valid_sample(
            invalid_lenses_sample
        ), "Sample should be valid without instruction length restrictions."
        assert dataset._is_valid_sample(
            invalid_pos_negative_lenses_sample
        ), "Sample should be valid without instruction length restrictions."
