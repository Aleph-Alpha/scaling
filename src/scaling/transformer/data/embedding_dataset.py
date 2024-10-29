import hashlib
import json
import random
from pathlib import Path
from typing import Any, Union, no_type_check

import numpy as np
import torch
from tqdm import tqdm

from scaling.core.data import (
    BaseBlendedDataset,
    BaseDataset,
    BaseDatasetItem,
    MemoryMapDataset,
    MemoryMapDatasetBuilder,
    broadcast_data,
)
from scaling.core.topology.topology import Topology
from scaling.transformer.data.text_dataset_batch import TextDatasetBatch
from scaling.transformer.data.utils import get_cumulative_seq_lengths
from scaling.transformer.tokenizer import Tokenizer


class EmbeddingDatasetItem(BaseDatasetItem):
    input_token_ids: torch.Tensor
    position_ids: torch.Tensor
    loss_weights: torch.Tensor

    def __init__(
        self,
        input_token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        loss_weights: torch.Tensor,
    ):
        self.input_token_ids = input_token_ids
        self.loss_weights = loss_weights
        self.position_ids = position_ids


class EmbeddingDataset(BaseDataset[EmbeddingDatasetItem, TextDatasetBatch, TextDatasetBatch]):
    def __init__(
        self,
        data_path: Path,
        sequence_length: int,
        seed: int,
        tokenizer: Tokenizer,
        number_of_hard_negatives: int,
        memory_map_dataset: bool = False,
        use_instruction: bool = True,
        query_side_only: bool = False,
        shuffle: bool = True,
    ):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.memory_map_dataset = memory_map_dataset
        self.number_of_hard_negatives = number_of_hard_negatives
        self.use_instruction = use_instruction
        self.query_side_only = query_side_only

        if number_of_hard_negatives > 1:
            raise NotImplementedError(
                "Selecting number_of_hard_negatives > 1 has not been implemented yet. \
                                       Choose values of [0,1]."
            )

        if self.query_side_only:
            assert self.use_instruction, "query_side_only was explicitly set to TRUE but use_instructions is FALSE. \
                Make sure to also set use_instructions to TRUE"

        self.data: list[int]
        if self.memory_map_dataset:
            self.mem_map_dataset = EmbeddingMemoryMapDataset(prefix_path=self.data_path)
            self.data = list(range(len(self.mem_map_dataset)))
        else:
            with open(self.data_path, "r", encoding="UTF-8") as r:
                self.data_jsonl: list[dict[str, Any]] = [json.loads(s) for s in str(r.read()).split("\n") if s != ""]  # type: ignore[no-redef]
                self.load_dataset_from_jsonl()

            self.data = list(range(len(self.jsonl_dataset)))
        self.seed: int | None = None
        self.data_item_index: list | None = None

        super().__init__(seed=seed, shuffle=shuffle)

    def _is_valid_sample(self, sample: dict[str, torch.Tensor | None]) -> bool:
        assert sample["query_token_ids"] is not None
        assert sample["positive_token_ids"] is not None
        assert sample["negative_token_ids"] is not None

        query_len = len(sample["query_token_ids"])
        positive_len = len(sample["positive_token_ids"])
        negatives_len = [len(neg) for neg in sample["negative_token_ids"]]
        if any([query_len == 0, positive_len == 0, any(n == 0 for n in negatives_len)]):
            return False

        if not self.use_instruction:
            return True

        assert sample["query_instruction_length"] is not None
        assert sample["positive_instruction_length"] is not None
        assert sample["negative_instruction_length"] is not None

        if sample["query_instruction_length"] >= self.sequence_length:
            return False

        if self.query_side_only:
            return True

        if sample["positive_instruction_length"] >= self.sequence_length:
            return False
        if any(n >= self.sequence_length for n in sample["negative_instruction_length"]):
            return False
        return True

    def load_dataset_from_jsonl(self) -> None:
        self.jsonl_dataset = []
        for d_json in self.data_jsonl:
            encoded_sample = EmbeddingDataset.encode_sample(d_json, self.tokenizer)

            if not self._is_valid_sample(encoded_sample):
                continue

            self.jsonl_dataset.append(
                {
                    "query_token_ids": encoded_sample["query_token_ids"],
                    "query_instruction_length": encoded_sample["query_instruction_length"],
                    "positive_token_ids": encoded_sample["positive_token_ids"],
                    "positive_instruction_length": encoded_sample["positive_instruction_length"],
                    "negative_token_ids": encoded_sample["negative_token_ids"],
                    "negative_instruction_length": encoded_sample["negative_instruction_length"],
                }
            )

    def set_seed(self, seed: int, shuffle: bool = True) -> None:
        # exit if we already computed the data index
        if self.seed is not None and self.seed == seed:
            return

        random.seed(seed)
        if shuffle:
            random.shuffle(self.data)

        self.seed = seed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> EmbeddingDatasetItem:
        if self.memory_map_dataset:
            query_token_ids, positive_token_ids, negative_token_ids = self.get_memory_map_token_ids(index=index)
            query_lense, positive_lense, negative_lenses = self.get_memory_map_instruct_lenses(index=index)
        else:
            query_token_ids, positive_token_ids, negative_token_ids = self.get_jsonl_token_ids(index=index)
            query_lense, positive_lense, negative_lenses = self.get_jsonl_instruct_lenses(index=index)

        if query_lense is not None and positive_lense is not None and negative_lenses is not None:
            assert (
                query_lense < self.sequence_length
                or positive_lense < self.sequence_length
                or any([not negative_lense < self.sequence_length for negative_lense in negative_lenses])
            ), "Your dataset contains an instruction that is longer than the sequence length. \
                This leads to nan embeddings. Increase sequence length."

        if self.use_instruction:
            assert (
                query_lense is not None
            ), "use_instruction was set to true but no query_lense was found in the dataset"
            if not self.query_side_only:
                assert (
                    positive_lense is not None
                ), "use_instruction was set to true but no positive_lense was found in the dataset"
                assert (
                    negative_lenses is not None
                ), "use_instruction was set to true but no negative_lenses were found in the dataset"

        if not self.use_instruction:
            if query_lense is not None:
                query_token_ids = query_token_ids[query_lense:]
            if positive_lense is not None:
                positive_token_ids = positive_token_ids[positive_lense:]
            if negative_lenses is not None:
                negative_token_ids = [negative_token_ids[i][negative_lenses[i] :] for i in range(len(negative_lenses))]
        elif self.query_side_only:
            if positive_lense is not None:
                positive_token_ids = positive_token_ids[positive_lense:]
            if negative_lenses is not None:
                negative_token_ids = [negative_token_ids[i][negative_lenses[i] :] for i in range(len(negative_lenses))]

        loss_weights = []

        def prepare_sample(
            token_ids: list[int], lense: int | None, is_query: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:
            padding_size = self.sequence_length - len(token_ids)
            sample = torch.tensor(token_ids + [self.tokenizer.padding_token_id] * padding_size)
            loss_weight = torch.ones(self.sequence_length, dtype=torch.float)
            if lense is not None and self.use_instruction:
                if not self.query_side_only or (self.query_side_only and is_query):
                    loss_weight[:lense] = 0

            if padding_size > 0:
                loss_weight[-padding_size:] = 0
            return sample, loss_weight

        query_sample, query_loss_weight = prepare_sample(query_token_ids, query_lense, is_query=True)
        loss_weights.append(query_loss_weight)

        positive_sample, positive_loss_weight = prepare_sample(positive_token_ids, positive_lense)
        loss_weights.append(positive_loss_weight)

        negative_samples = []
        assert (
            len(negative_token_ids) >= self.number_of_hard_negatives
        ), "Selected number of hard negatives was smaller than the available negative samples for item"

        for i, negative_completion in enumerate(negative_token_ids[: self.number_of_hard_negatives]):
            negative_sample, negative_loss_weight = prepare_sample(
                negative_completion, negative_lenses[i] if negative_lenses is not None else None
            )
            loss_weights.append(negative_loss_weight)
            negative_samples.append(negative_sample)

        query_sample = query_sample[: self.sequence_length]
        positive_sample = positive_sample[: self.sequence_length]
        negative_samples = [n_sample[: self.sequence_length] for n_sample in negative_samples]

        input_token_ids = torch.stack([query_sample] + [positive_sample] + negative_samples)
        position_ids = torch.arange(0, self.sequence_length).repeat(len(input_token_ids), 1)

        return EmbeddingDatasetItem(
            input_token_ids=input_token_ids,
            position_ids=position_ids,
            loss_weights=torch.stack(loss_weights),
        )

    def collate(self, batch: list[EmbeddingDatasetItem]) -> TextDatasetBatch:
        input_token_ids = torch.stack([sample for batch_item in batch for sample in batch_item.input_token_ids])
        position_ids = torch.stack([sample for batch_item in batch for sample in batch_item.position_ids])
        loss_weights = torch.stack([sample for batch_item in batch for sample in batch_item.loss_weights])

        cumulative_seq_lengths = get_cumulative_seq_lengths(input_token_ids, reset_attention_mask=False)

        return TextDatasetBatch(
            input_token_ids=input_token_ids,
            position_ids=position_ids,
            cumulative_seq_lengths=cumulative_seq_lengths,
            loss_weights=loss_weights,
        )

    def ident(self) -> str:
        md5_hash = hashlib.md5()
        md5_hash.update(str(self.data_path).encode("utf-8"))
        md5_hash.update(json.dumps(self.tokenizer.tokenizer.get_vocab(), sort_keys=True, default=str).encode("utf-8"))

        return f"{md5_hash.hexdigest()}-seq-{self.sequence_length}"

    @staticmethod
    def sync_batch_to_model_parallel(
        topology: Topology | None = None,
        batch: TextDatasetBatch | None = None,
    ) -> TextDatasetBatch:
        if topology is None:
            assert batch is not None
            return batch

        if topology.model_parallel_rank == 0:
            assert batch is not None
            batch.contiguous_()
            assert batch.input_token_ids is not None
            assert batch.cumulative_seq_lengths_padded is not None
            assert batch.position_ids is not None
            long_tensors: list[torch.Tensor | None] = [
                batch.input_token_ids,
                batch.cumulative_seq_lengths_padded.to(torch.long),
                batch.position_ids,
            ]

            float_tensors: list[torch.Tensor | None] = [batch.loss_weights]

        else:
            assert batch is None
            long_tensors = [None, None, None]
            float_tensors = [None]

        broadcast_long_tensors = broadcast_data(tensors=long_tensors, dtype=torch.long, topology=topology)  # type: ignore
        broadcast_float_tensors = broadcast_data(tensors=float_tensors, dtype=torch.float32, topology=topology)  # type: ignore

        cumulative_seq_lengths_padded = broadcast_long_tensors[1].to(torch.int32)

        return TextDatasetBatch(
            input_token_ids=broadcast_long_tensors[0],
            cumulative_seq_lengths_padded=cumulative_seq_lengths_padded,
            position_ids=broadcast_long_tensors[2],
            loss_weights=broadcast_float_tensors[0],
        )

    def get_jsonl_token_ids(self, index: int) -> tuple[list[int], list[int], list[list[int]]]:
        src_index = self.data[index]
        sample = self.jsonl_dataset[src_index]
        return sample["query_token_ids"], sample["positive_token_ids"], sample["negative_token_ids"]

    def get_jsonl_instruct_lenses(self, index: int) -> tuple[int | None, int | None, list[int] | None]:
        src_index = self.data[index]
        sample = self.jsonl_dataset[src_index]
        return (
            sample["query_instruction_length"],
            sample["positive_instruction_length"],
            sample["negative_instruction_length"],
        )

    def get_memory_map_token_ids(self, index: int) -> tuple[list[int], list[int], list[list[int]]]:
        src_index = self.data[index]

        all_tokens = self.mem_map_dataset[src_index].tolist()
        query_length = self.mem_map_dataset.query_length[src_index]
        positive_length = self.mem_map_dataset.positive_length[src_index]
        negative_start_indices = self.mem_map_dataset.negative_start_indices[src_index]

        query = all_tokens[:query_length]
        positive = all_tokens[query_length : query_length + positive_length]
        negatives = []
        offset = query_length + positive_length
        for negative_sample_length in negative_start_indices:
            negatives.append(all_tokens[offset : offset + negative_sample_length])
            offset += negative_sample_length
        return query, positive, negatives

    def get_memory_map_instruct_lenses(self, index: int) -> tuple[int | None, int | None, list[int] | None]:
        src_index = self.data[index]
        query_lense, positive_lense, negative_lense = None, None, None
        if len(self.mem_map_dataset.query_instruction_length):
            query_lense = self.mem_map_dataset.query_instruction_length[src_index]
            positive_lense = self.mem_map_dataset.positive_instruction_length[src_index]
            negative_lense = self.mem_map_dataset.negative_instruction_length[src_index]

        return query_lense, positive_lense, negative_lense

    @staticmethod
    def encode_sample(data_item: dict, tokenizer: Tokenizer) -> dict[str, Any]:
        if isinstance(data_item["query"], (tuple, list)):
            query_token_ids = tokenizer.encode(data_item["query"][0] + " " + data_item["query"][1])
            query_instruction_length = len(tokenizer.encode(data_item["query"][0]))
            positive_token_ids = tokenizer.encode(data_item["positives"][0][0] + " " + data_item["positives"][0][1])
            positive_instruction_length = len(tokenizer.encode(data_item["positives"][0][0]))
            negative_token_ids = [tokenizer.encode(sample[0] + " " + sample[1]) for sample in data_item["negatives"]]
            negative_instruction_length = [len(tokenizer.encode(sample[0])) for sample in data_item["negatives"]]
        else:
            query_token_ids = tokenizer.encode(data_item["query"])
            positive_token_ids = tokenizer.encode(data_item["positives"][0])
            negative_token_ids = [tokenizer.encode(sample) for sample in data_item["negatives"]]
            query_instruction_length = None
            positive_instruction_length = None
            negative_instruction_length = None

        return {
            "query_token_ids": query_token_ids,
            "positive_token_ids": positive_token_ids,
            "negative_token_ids": negative_token_ids,
            "query_instruction_length": query_instruction_length,
            "positive_instruction_length": positive_instruction_length,
            "negative_instruction_length": negative_instruction_length,
        }

    @staticmethod
    def jsonl_to_embedding_mmap(
        jsonl_file: Union[str, Path],
        tokenizer: Tokenizer,
        out_prefix_path: Union[str, Path],
    ) -> None:
        # instantiate a builder to write data
        builder = EmbeddingDatasetMemoryMapDatasetBuilder(prefix_path=Path(out_prefix_path))

        with open(jsonl_file, "r", encoding="UTF-8") as in_file:
            for i, data_item in enumerate(tqdm(in_file)):
                if data_item == "":
                    continue
                data_item = json.loads(data_item)
                assert len(data_item["positives"]) == 1
                assert len(data_item["negatives"]) > 0

                encoded_sample = EmbeddingDataset.encode_sample(data_item, tokenizer)

                builder.add(
                    query_token_ids=np.array(encoded_sample["query_token_ids"]),
                    query_instruction_length=encoded_sample["query_instruction_length"],
                    positive_token_ids=np.array(encoded_sample["positive_token_ids"]),
                    positive_instruction_length=encoded_sample["positive_instruction_length"],
                    negative_token_ids=np.stack([np.array(neg) for neg in encoded_sample["negative_token_ids"]]),
                    negative_instruction_length=encoded_sample["negative_instruction_length"],
                )

        builder.finalize()

        with open(str(out_prefix_path) + ".done", "w") as f:
            f.write("True")


class EmbeddingMemoryMapDataset(MemoryMapDataset):
    def initialize(self) -> None:
        """
        dtype - dtype of tokens
        pointer - start index of each document in the mmap file

        chosen_start_index - index in token array when chosen summary starts
        rejected_start_index - index in token array when rejected summary starts
        """
        # load index
        index_dict = json.loads(self.file_path_meta.read_text())

        self.dtype = np.dtype(index_dict["dtype"])

        self.index_dtype = np.dtype(index_dict["index_dtype"])

        self.dtype_size = self.dtype.itemsize
        self.index_dtype_size = self.index_dtype.itemsize
        self.document_count = index_dict["document_count"]

        self.query_length = index_dict["query_length"]
        self.query_instruction_length = index_dict["query_instruction_length"]
        self.positive_length = index_dict["positive_length"]
        self.positive_instruction_length = index_dict["positive_instruction_length"]
        self.negative_start_indices = index_dict["negative_start_indeces"]
        self.negative_instruction_length = index_dict["negative_instruction_length"]

        # open memory map
        self._bin_buffer_mmap = np.memmap(self.file_path_data, mode="r", order="C", dtype=self.dtype)
        self._bin_buffer = memoryview(self._bin_buffer_mmap)  # type: ignore
        self._bin_buffer_index_mmap = np.memmap(self.file_path_index, mode="r", order="C", dtype=self.index_dtype)
        self._bin_buffer_index = memoryview(self._bin_buffer_index_mmap)  # type: ignore

        self._index = None
        if self.load_index_to_memory:
            self._index = np.array(
                np.frombuffer(
                    self._bin_buffer_index,
                    dtype=self.index_dtype,
                    count=2 * len(self),
                    offset=0,
                ).reshape(len(self), 2)
            )
            self._bin_buffer_index_mmap._mmap.close()  # type: ignore
            del self._bin_buffer_index_mmap


class EmbeddingDatasetMemoryMapDatasetBuilder(MemoryMapDatasetBuilder):
    def __init__(self, prefix_path: Path):
        self.query_length: list[int] = []
        self.positive_length: list[int] = []
        self.negative_start_indeces: list[list[int]] = []
        self.query_instruction_length: list[int] = []
        self.positive_instruction_length: list[int] = []
        self.negative_instruction_length: list[int] = []
        super().__init__(prefix_path)

    @no_type_check
    def add(
        self,
        query_token_ids: np.ndarray,
        positive_token_ids: np.ndarray,
        negative_token_ids: list[np.ndarray],
        query_instruction_length: int | None = None,
        positive_instruction_length: int | None = None,
        negative_instruction_length: int | None = None,
    ):
        negative_token_ids_concatenated = np.concatenate(negative_token_ids, axis=-1)

        all_tokens = np.concatenate([query_token_ids, positive_token_ids, negative_token_ids_concatenated])

        super().add(all_tokens)
        negative_start_indeces = [len(sample) for sample in negative_token_ids]

        self.query_length.append(len(query_token_ids))
        self.positive_length.append(len(positive_token_ids))
        self.negative_start_indeces.append(negative_start_indeces)

        if query_instruction_length is not None:
            self.query_instruction_length.append(query_instruction_length)
        if positive_instruction_length is not None:
            self.positive_instruction_length.append(positive_instruction_length)
        if negative_instruction_length is not None:
            self.negative_instruction_length.append(negative_instruction_length)

    def finalize(self) -> None:
        """
        finalizes the creation of the dataset by closing
        the data file and writing the index
        """
        self.data_file.close()
        self.index_file.close()

        index_dict = {
            "dtype": np.dtype(self.dtype).name,
            "index_dtype": np.dtype(self.index_dtype).name,
            "query_length": self.query_length,
            "positive_length": self.positive_length,
            "query_instruction_length": self.query_instruction_length,
            "positive_instruction_length": self.positive_instruction_length,
            "negative_instruction_length": self.negative_instruction_length,
            "negative_start_indeces": self.negative_start_indeces,
            "document_count": self.document_count,
        }
        json.dump(index_dict, open(self.file_path_meta, "w"))


class EmbeddingBlendedDataset(
    BaseBlendedDataset[
        EmbeddingDatasetItem,
        TextDatasetBatch,
        TextDatasetBatch,
        EmbeddingDataset,
    ]
):
    pass
