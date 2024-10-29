import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm  # type: ignore

from scaling.core import (
    BaseBlendedDataset,
    BaseDataset,
    FileDataset,
    MemoryMapDataset,
    MemoryMapDatasetBuilder,
    Topology,
    broadcast_data,
)
from scaling.transformer.tokenizer import Tokenizer

from .legacy_dataset import get_indexed_dataset_, make_builder
from .text_dataset_batch import TextDatasetBatch, TextDatasetBatchBeforeSync
from .text_dataset_item import TextDatasetItem


class TextDataset(BaseDataset[TextDatasetItem, TextDatasetBatchBeforeSync, TextDatasetBatch]):
    """
    Torch dataset providing tokenized text of always the same sequence length; data is loaded from a memory map.

    every TextDatasetItem in the dataset:
        - contains the field "token_ids"
        - token_ids has a tensor of one dimension and data type long (int64) as data type
        - token_ids are always of sequence_length so that a batching can be conducted

    """

    def __init__(
        self,
        data_prefix: Path,
        sequence_length: int,
        seed: int,
        legacy_dataset: bool = False,
        load_mmap_index_to_memory: bool = False,
        load_data_item_mmap_index_to_memory: bool = False,
        only_full_sequences: bool = False,
        allow_incomplete_sequences_every_n: int = 0,
        use_mmap: bool = True,
        shuffle: bool = True,
        reset_attention_mask: bool = True,
        reset_position_ids: bool = True,
    ):
        """
        data_prefix (`Path`)
            path to a memory map

        sequence_length (`int`)
            expected sequence length of the token_ids in output.
        """
        # remember params
        self.use_mmap = use_mmap
        self.sequence_length = sequence_length
        self.legacy_dataset = legacy_dataset
        self.load_mmap_index_to_memory = load_mmap_index_to_memory
        self.load_data_item_mmap_index_to_memory = load_data_item_mmap_index_to_memory
        self.only_full_sequences = only_full_sequences
        self.allow_incomplete_sequences_every_n = allow_incomplete_sequences_every_n
        if self.load_mmap_index_to_memory or self.load_data_item_mmap_index_to_memory:
            assert not self.legacy_dataset, (
                "cannot set load_mmap_index_to_memory=True or "
                "load_data_item_mmap_index_to_memory=True if using legacy dataset"
            )

        # load data
        if legacy_dataset:
            self.memory_map = get_indexed_dataset_(str(data_prefix), data_impl="mmap", skip_warmup=True)
        elif use_mmap:
            self.memory_map = MemoryMapDataset(  # type: ignore
                prefix_path=data_prefix,
                load_index_to_memory=self.load_mmap_index_to_memory,
            )
        else:
            self.memory_map = FileDataset(  # type: ignore
                prefix_path=data_prefix,
                load_index_to_memory=self.load_mmap_index_to_memory,
            )

        if self.only_full_sequences:
            assert not self.legacy_dataset, "full sequences datasets not supported for legacy datasets."

        # shuffling
        self.seed: int | None = None
        self.data_item_index: MemoryMapDataset | FileDataset | None = None

        self.reset_attention_mask = reset_attention_mask
        self.reset_position_ids = reset_position_ids

        super().__init__(seed=seed, shuffle=shuffle, data_prefix=data_prefix)

    def ident(self) -> str:
        md5_hash = hashlib.md5()
        md5_hash.update(str(self.data_prefix).encode("utf-8"))

        return f"{md5_hash.hexdigest()}-seq-{self.sequence_length}"

    def set_seed(self, seed: int, shuffle: bool = True) -> None:
        if self.legacy_dataset:
            # will be removed anyway
            self.compute_data_index(seed=seed)
        else:
            self._set_seed(seed=seed, shuffle=shuffle)

    def compute_data_index(self, seed: int) -> None:
        # skip if same result is to be expected
        if self.seed is not None and self.seed == seed:
            return
        self.seed = seed

        # Create cache file
        # The cache file is created just on one rank to avoid writing to same file from multiple ranks
        # To ensure consistency and having self.data_index as indexed dataset the data index is not kept in memory
        # but always loaded from the file again.
        cache_file_stem = self.get_data_index_cache_filename_stem(seed)
        cache_file_bin = self.get_data_index_cache_filename_bin(seed)
        cache_file_idx = self.get_data_index_cache_filename_idx(seed)
        cache_file_done = self.get_data_index_cache_filename_done(seed)
        if not Path(cache_file_done).is_file():
            get_rank = lambda: (torch.distributed.get_rank())  # noqa: E731
            # Make sure to only create the data index on one rank and read from the others
            # It is assumed that rank 0 initializes the dataset
            if (not torch.distributed.is_initialized()) or (get_rank() == 0):
                print(
                    f"DecoderDataset creating index for seed {seed} on rank 0 for {Path(self.data_prefix).name}",
                    flush=True,
                )

                # shuffle indexed_dataset
                np_rng = np.random.RandomState(seed=seed)
                doc_indices = np.arange(len(self.memory_map))  # type: ignore[arg-type]
                np_rng.shuffle(doc_indices)

                # collect index

                builder = make_builder(cache_file_bin, impl="mmap")

                data_index_item = list()
                data_index_item_token_count = 0
                for doc_index in tqdm(doc_indices, f"DecoderDataset {self.data_prefix}"):
                    doc_current_pos = 0
                    doc_token_count = self.memory_map.sizes[doc_index]  # type: ignore[union-attr]
                    while doc_current_pos < (
                        doc_token_count - 1
                    ):  # on token for loss target is always added to the seq_len
                        # select end position for current iteration in document
                        doc_current_end_pos = min(
                            doc_token_count,
                            doc_current_pos + 1 + self.sequence_length - data_index_item_token_count,
                        )

                        # append to data item
                        data_index_item_token_count += doc_current_end_pos - doc_current_pos
                        data_index_item.append(
                            (
                                int(doc_index),
                                int(doc_current_pos),
                                int(doc_current_end_pos),
                            )
                        )
                        if data_index_item_token_count == self.sequence_length + 1:
                            item = torch.IntTensor(data_index_item).flatten()
                            builder.add_item(item)
                            data_index_item = list()
                            data_index_item_token_count = 0

                        doc_current_pos = (
                            doc_current_end_pos - 1
                        )  # the end position includes the +1 token for the loss target. this token can be part of
                        # the next item

                builder.finalize(cache_file_idx)
                with open(cache_file_done, "w") as f:
                    f.write("True")

        # Load cache files
        # Wait on all ranks until files are available
        cache_files_found = False
        attempt_count = 0
        while not cache_files_found:
            cache_files_found = (
                Path(cache_file_bin).is_file() and Path(cache_file_idx).is_file() and Path(cache_file_done).is_file()
            )
            attempt_count += 1
            if cache_files_found:
                break
            if attempt_count % 12 == 0:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(
                    f"DecoderDataset waiting on index for seed {seed} on rank {rank} for "
                    f"{Path(self.data_prefix).name}; elapsed {attempt_count * 5 / 60} minutes",
                    flush=True,
                )
            time.sleep(5)

        self.data_item_index = get_indexed_dataset_(cache_file_stem, "mmap", True)  # type: ignore[assignment]

    def get_data_index_cache_filename_stem(self, seed: int) -> str:
        cache_file = str(self.data_prefix) + f"_index_cache_decoder_dataset_seed_{seed}_seq_len_{self.sequence_length}"

        if self.only_full_sequences:
            cache_file += (
                f"_only_full_sequences_allow_incomplete_sequences_every_n_{self.allow_incomplete_sequences_every_n}"
            )

        return cache_file

    def _set_seed(self, seed: int, shuffle: bool = True) -> None:
        """
        Computes and index on the tokenized dataset such that each item in the index
        - results in a batch item of sequence length
        - no padding token is used
        - longer texts are split into different data items
        - shorter texts are concatenated and separated with an end of text token
        """
        # exit if we already computed the data index
        if self.seed is not None and self.seed == seed:
            return

        self.seed = seed

        # Create cache file
        # The cache file is created just on one rank to avoid writing to same file from multiple ranks
        # To ensure consistency and having self.data_index as indexed dataset the data index is not kept in memory
        # but always loaded from the file again.
        cache_file_stem = self.get_data_index_cache_filename_stem(seed)
        cache_file_bin = self.get_data_index_cache_filename_bin(seed)
        cache_file_idx = self.get_data_index_cache_filename_idx(seed)
        cache_file_meta = self.get_data_index_cache_filename_meta(seed)
        if not Path(cache_file_meta).is_file():
            get_rank = lambda: (torch.distributed.get_rank())  # noqa: E731
            # Make sure to only create the data index on one rank and read from the others
            # It is assumed that rank 0 initializes the dataset
            if (not torch.distributed.is_initialized()) or (torch.distributed.is_initialized() and get_rank() == 0):
                print(
                    f"TextDataset creating index for seed {seed} on rank 0 for {Path(self.data_prefix).name}",
                    flush=True,
                )

                if Path(cache_file_bin).is_file():
                    Path(cache_file_bin).unlink()

                if Path(cache_file_idx).is_file():
                    Path(cache_file_idx).unlink()

                # shuffle indexed_dataset
                np_rng = np.random.RandomState(seed=seed)
                doc_indices = np.arange(len(self.memory_map))  # type: ignore
                if shuffle:
                    np_rng.shuffle(doc_indices)

                # collect index

                builder = MemoryMapDatasetBuilder(Path(cache_file_stem), dtype=np.int64, index_dtype=np.int64)  # type: ignore[arg-type]

                data_index_item = list()
                data_index_item_token_count = 0
                full_sequences_count: int = 0
                half_sequences_count: int = 0
                in_half_step: bool = False
                for doc_index in tqdm(doc_indices, f"DecoderDataset {self.data_prefix}"):
                    doc_current_pos = 0
                    doc_token_count = self.memory_map.sizes(doc_index)  # type: ignore
                    while doc_current_pos < (
                        doc_token_count - 1
                    ):  # on token for loss target is always added to the seq_len
                        # select end position for current iteration in document
                        doc_current_end_pos = min(
                            doc_token_count,
                            doc_current_pos + 1 + self.sequence_length - data_index_item_token_count,
                        )

                        if self.only_full_sequences:
                            # Break if the Document gets to short to fill a full sequences.
                            if in_half_step:
                                pass

                            elif (int(doc_current_end_pos) - int(doc_current_pos)) < (self.sequence_length + 1):
                                if (
                                    self.allow_incomplete_sequences_every_n != 0
                                    and (
                                        (full_sequences_count / self.allow_incomplete_sequences_every_n)
                                        - half_sequences_count
                                    )
                                    >= 1
                                ):
                                    in_half_step = True
                                else:
                                    break

                            else:
                                full_sequences_count += 1

                        # append to data item
                        data_index_item_token_count += doc_current_end_pos - doc_current_pos

                        data_index_item.append(
                            (
                                int(doc_index),
                                int(doc_current_pos),
                                int(doc_current_end_pos),
                            )
                        )

                        if data_index_item_token_count == self.sequence_length + 1:
                            builder.add(np.array(data_index_item, dtype=np.int64).flatten())
                            data_index_item = list()
                            data_index_item_token_count = 0

                            if in_half_step:
                                half_sequences_count += 1

                            in_half_step = False

                        doc_current_pos = (
                            doc_current_end_pos - 1
                        )  # the end position includes the +1 token for the loss target. this token can be part of
                        # the next item

                builder.finalize()

        # Load cache files
        # Wait on all ranks until files are available
        cache_files_found = False
        attempt_count = 0
        while not cache_files_found:
            cache_files_found = (
                Path(cache_file_bin).is_file() and Path(cache_file_idx).is_file() and Path(cache_file_meta).is_file()
            )
            attempt_count += 1
            if cache_files_found:
                break
            if attempt_count % 12 == 0:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(
                    f"TextDataset waiting on index for seed {seed} on rank {rank} for {Path(self.data_prefix).name}; "
                    f"elapsed {attempt_count * 5 / 60} minutes",
                    flush=True,
                )
            time.sleep(5)

        if self.use_mmap:
            self.data_item_index = MemoryMapDataset(
                prefix_path=Path(cache_file_stem),
                load_index_to_memory=self.load_data_item_mmap_index_to_memory,
            )
        else:
            self.data_item_index = FileDataset(
                prefix_path=Path(cache_file_stem),
                load_index_to_memory=self.load_data_item_mmap_index_to_memory,
            )

    def __len__(self) -> int:
        return len(self.data_item_index)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> TextDatasetItem:
        assert self.data_item_index is not None, "data item index not set"

        # retrieve token ids from memory map
        source_document_index = self.data_item_index[index]

        source_document_index_split = np.split(source_document_index, source_document_index.shape[0] // 3)  # type: ignore[assignment]
        source_document_index_split_list = [i.tolist() for i in source_document_index_split]  # type: ignore[assignment]
        token_ids: list[int] = list()
        for document_index, start, end in source_document_index_split_list:
            t_ids = self.memory_map[document_index][start:end]  # type: ignore
            token_ids.extend(t_ids)
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

        return TextDatasetItem(token_ids=token_ids_tensor)

    def collate(self, batch: list[TextDatasetItem]) -> TextDatasetBatchBeforeSync:
        """
        Used to collate lists of samples into batches
        The default implementation returns a BaseDataBatch NamedTuple with the same attributes as the first element
        of the batch
        """
        # assume that all batch items have the same data type
        # get list of fields with their type
        token_ids = torch.stack(
            [batch_item.token_ids for batch_item in batch]
        )  # don't move to cuda, otherwise background data loader processes will not work

        return TextDatasetBatchBeforeSync(
            token_ids=token_ids,
            reset_attention_mask=self.reset_attention_mask,
            reset_position_ids=self.reset_position_ids,
        )

    @staticmethod
    def sync_batch_to_model_parallel(
        topology: Topology | None,
        batch: TextDatasetBatchBeforeSync | None,
    ) -> TextDatasetBatch:
        if topology is None:
            assert batch is not None
            input_token_ids = batch.token_ids[:, :-1]
            target_token_ids = batch.token_ids[:, 1:]

            return TextDatasetBatch(
                input_token_ids=input_token_ids,
                target_token_ids=target_token_ids,
                reset_attention_mask=batch.reset_attention_mask,
                reset_position_ids=batch.reset_position_ids,
            )

        if topology.model_parallel_rank == 0:
            assert batch is not None
            tensors: list[torch.Tensor | None] = [
                batch.token_ids,
                torch.tensor([batch.reset_attention_mask], dtype=torch.int64),
                torch.tensor([batch.reset_position_ids], dtype=torch.int64),
            ]

        else:
            assert batch is None
            tensors = [None, None, None]

        broadcast_tensors = broadcast_data(tensors=tensors, dtype=torch.long, topology=topology)  # type: ignore

        input_token_ids = broadcast_tensors[0][:, :-1]
        target_token_ids = broadcast_tensors[0][:, 1:]
        reset_attention_mask = bool(broadcast_tensors[1].item())
        reset_position_ids = bool(broadcast_tensors[2].item())

        return TextDatasetBatch(
            input_token_ids=input_token_ids,
            target_token_ids=target_token_ids,
            reset_attention_mask=reset_attention_mask,
            reset_position_ids=reset_position_ids,
        )

    @staticmethod
    def jsonl_to_memory_map(data_file_jsonl: Path, prefix_path_memory_map: Path) -> None:
        """
        converts a jsonl file to a memory map
        """

        data_file_jsonl = Path(data_file_jsonl)
        prefix_path_memory_map = Path(prefix_path_memory_map)

        tokenizer = Tokenizer.default()
        builder = MemoryMapDatasetBuilder(prefix_path=prefix_path_memory_map)

        with open(data_file_jsonl, "r", encoding="UTF-8") as f:
            for line in f:
                data_item = json.loads(line)
                tokenized_text = tokenizer.encode(data_item["text"])
                tokenized_text = tokenized_text + [tokenizer.eos_token_id]
                builder.add(np_array=np.array(tokenized_text))
        builder.finalize()


class TextBlendedDataset(
    BaseBlendedDataset[
        TextDatasetItem,
        TextDatasetBatchBeforeSync,
        TextDatasetBatch,
        TextDataset,
    ]
):
    pass
