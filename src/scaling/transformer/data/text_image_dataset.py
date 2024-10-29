import hashlib
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm  # type: ignore

from scaling.core import (
    BaseBlendedDataset,
    BaseDataset,
    Topology,
    broadcast_data,
)
from scaling.core.data.file_dataset import FileDataset
from scaling.core.data.memory_map import MemoryMapDataset, MemoryMapDatasetBuilder
from scaling.core.data.pb_memory_map import PbMemoryMap
from scaling.core.data.proto import text_image_dataset_pb2
from scaling.transformer.data.dataset_item import TextImageDatasetItem
from scaling.transformer.data.text_dataset_batch import TextDatasetBatch
from scaling.transformer.data.utils import get_cumulative_seq_lengths
from scaling.transformer.model.image_encoder import clip_transform

IMAGE_ENCODER_TOKEN_COUNTS = 144  # TODO so far this is constant
IMAGE_TRANSFORM_FN = clip_transform((384, 384))  # TODO so far this is constant


class TextImageDataset(BaseDataset[TextImageDatasetItem, TextDatasetBatch, TextDatasetBatch]):
    """
    Initialize the TextImageDataset.

    :param data_prefix:         The prefix path for the dataset files.
    :type data_prefix:          Path
    :param sequence_length:     The length of the sequences.
    :type sequence_length:      int
    :param softprompt_n_tokens: The number of tokens for the soft prompt.
    :type softprompt_n_tokens:  int
    :param data_index_prefix:   The prefix path for the data index files, defaults to an empty Path.
    :type data_index_prefix:    Path, optional
    :param padding_token_id:    The ID of the padding token, defaults to 0.
    :type padding_token_id:     int, optional
    :param shuffle:             Whether to shuffle the dataset, defaults to True.
    :type shuffle:              bool, optional
    :param seed:                The seed for random number generation, defaults to 42.
    :type seed:                 int, optional
    :param use_mmap:            Whether to use memory mapping for the dataset, defaults to True.
    :type use_mmap:             bool, optional
    :param load_data_item_mmap_index_to_memory:
                                Whether to load the data item memory map index into memory, defaults to False.
    :type load_data_item_mmap_index_to_memory:
                                bool, optional
    """

    def __init__(
        self,
        data_prefix: Path,
        sequence_length: int,
        softprompt_n_tokens: int,
        data_index_prefix: Path | None = None,
        padding_token_id: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        use_mmap: bool = True,
        load_data_item_mmap_index_to_memory: bool = False,
    ):
        # remember params
        self.data_path_parent = Path(data_prefix).parent
        self.sequence_length = sequence_length
        self.softprompt_n_tokens = softprompt_n_tokens
        self.padding_token_id = padding_token_id

        # shuffling
        self.seed: Optional[int] = None
        self.data_item_index: Optional[MemoryMapDataset | FileDataset] = None
        self.load_data_item_mmap_index_to_memory = load_data_item_mmap_index_to_memory
        self.use_mmap = use_mmap

        self.memory_map = PbMemoryMap(
            prefix_path=data_prefix,
            pb_datatype=text_image_dataset_pb2.TextImageExample,  # type: ignore
        )

        super().__init__(
            seed=seed,
            shuffle=shuffle,
            data_prefix=data_prefix,
            data_index_prefix=data_index_prefix,
        )

    def set_seed(self, seed: int, shuffle: bool = True) -> None:
        """
        Set the seed for the dataset.

        :param seed:        The seed for random number generation.
        :type seed:         int
        :param shuffle:     Whether to shuffle the dataset, defaults to True.
        :type shuffle:      bool, optional
        """

        if self.seed is not None and self.seed == seed:
            return None

        self.seed = seed

        cache_file_stem = self.get_data_index_cache_filename_stem(seed)
        cache_file_bin = self.get_data_index_cache_filename_bin(seed)
        cache_file_idx = self.get_data_index_cache_filename_idx(seed)
        cache_file_meta = self.get_data_index_cache_filename_meta(seed)

        if not Path(cache_file_meta).is_file():
            self._build_index_cache(
                shuffle=shuffle,
                cache_file_stem=cache_file_stem,
                cache_file_bin=cache_file_bin,
                cache_file_idx=cache_file_idx,
            )

        self._wait_for_cache_files(
            cache_file_bin=cache_file_bin,
            cache_file_idx=cache_file_idx,
            cache_file_meta=cache_file_meta,
        )

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

    def _build_index_cache(
        self,
        shuffle: bool,
        cache_file_stem: str,
        cache_file_bin: str,
        cache_file_idx: str,
    ) -> None:
        if not (
            not torch.distributed.is_initialized()
            or torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
        ):
            # only rank 0 should build the index
            return None

        print(
            f"TextImageDataset creating index for seed {self.seed} on " "rank 0 for {Path(self.data_prefix).name}",
            flush=True,
        )

        if Path(cache_file_bin).is_file():
            Path(cache_file_bin).unlink()

        if Path(cache_file_idx).is_file():
            Path(cache_file_idx).unlink()

        doc_indices = np.arange(len(self.memory_map))
        builder = MemoryMapDatasetBuilder(
            Path(cache_file_stem),
            dtype=np.dtype(np.int64),
            index_dtype=np.dtype(np.int64),
        )

        if shuffle:
            np_rng = np.random.RandomState(seed=self.seed)
            np_rng.shuffle(doc_indices)

        for doc_index in tqdm(doc_indices):
            builder.add(np.array([doc_index]))

        builder.finalize()

    def _wait_for_cache_files(
        self,
        cache_file_bin: str,
        cache_file_idx: str,
        cache_file_meta: str,
    ) -> bool:
        attempt_count = 0
        cache_files_found = False
        while not cache_files_found:
            file_bin_found = Path(cache_file_bin).is_file()
            file_idx_found = Path(cache_file_idx).is_file()
            file_meta_found = Path(cache_file_meta).is_file()
            cache_files_found = all([file_bin_found, file_idx_found, file_meta_found])
            if cache_files_found:
                break

            attempt_count += 1
            if attempt_count % 12 == 0:
                rank = 0
                if torch.distributed.is_initialized():
                    rank = torch.distributed.get_rank()

                print(
                    f"TextImageDataset waiting on index for seed {self.seed} on "
                    f"rank {rank} for {Path(self.data_prefix).name}; "
                    f"elapsed {attempt_count * 5 / 60} minutes",
                    flush=True,
                )

            time.sleep(5)

        return True

    def ident(self) -> str:
        md5_hash = hashlib.md5()
        md5_hash.update(str(self.data_prefix).encode("utf-8"))
        return f"{md5_hash.hexdigest()}-seq-{self.sequence_length}"

    def __len__(self) -> int:
        return len(self.memory_map)

    def __getitem__(self, index: int) -> TextImageDatasetItem:
        if self.data_item_index is None:
            raise ValueError("Data item index not set. Call 'set_seed' " "before accessing data.")

        data_indices = self.data_item_index[index]
        data = self.memory_map[data_indices[0]]

        input_token_list: list[int] = data["input_token_list"]
        target_token_list: list[int] = data["target_token_list"]
        loss_mask_list: list[int] = data["loss_mask_list"]
        prompt_image_data: list[bytes] = data["prompt_image_data"]

        image_loc = [(loc["start_index"], loc["end_index"]) for loc in data["prompt_image_locations"]]
        prompt_image_locations: list[tuple[int, int]] = image_loc

        if self.softprompt_n_tokens > 0:
            input_token_list = [0] * self.softprompt_n_tokens + input_token_list
            target_token_list = [0] * self.softprompt_n_tokens + target_token_list
            loss_mask_list = [0] * self.softprompt_n_tokens + loss_mask_list

            # shift image locations
            if prompt_image_locations is not None:
                for i, loc in enumerate(prompt_image_locations):
                    start_loc, end_loc = loc
                    start_loc += self.softprompt_n_tokens
                    end_loc += self.softprompt_n_tokens
                    prompt_image_locations[i] = (start_loc, end_loc)

        padding_size = self.sequence_length - len(input_token_list)
        input_token_list = input_token_list + [self.padding_token_id] * padding_size
        target_token_list = target_token_list + [self.padding_token_id] * padding_size
        loss_mask_list = loss_mask_list + [0] * padding_size

        input_token_list = input_token_list[: self.sequence_length]
        target_token_list = target_token_list[: self.sequence_length]
        loss_mask_list = loss_mask_list[: self.sequence_length]

        input_token_ids = torch.tensor(input_token_list, dtype=torch.long)
        target_token_ids = torch.tensor(target_token_list, dtype=torch.long)
        loss_weights = torch.tensor(loss_mask_list, dtype=torch.float32)

        prompt_images_transformed = []
        for image_data in prompt_image_data:
            image = Image.open(BytesIO(image_data))
            prompt_images_transformed.append(IMAGE_TRANSFORM_FN(image))

        cumulative_seq_lengths = torch.tensor([0, self.sequence_length], dtype=torch.int32)
        position_ids = torch.arange(0, self.sequence_length)

        return TextImageDatasetItem(
            input_token_ids=input_token_ids,
            target_token_ids=target_token_ids,
            cumulative_seq_lengths=cumulative_seq_lengths,
            position_ids=position_ids,
            loss_weights=loss_weights,
            input_images=prompt_images_transformed,
            input_image_locations=prompt_image_locations,
        )

    def collate(self, batch: List[TextImageDatasetItem]) -> TextDatasetBatch:
        """
        Collate a list of TextImageDatasetItem instances into a TextDatasetBatch.

        :param batch:   A list of TextImageDatasetItem instances.
        :type batch:    List[TextImageDatasetItem]
        :return:        A TextDatasetBatch instance.
        :rtype:         TextDatasetBatch
        """

        # assume that all batch items have the same data type
        # get list of fields with their type
        input_token_ids = torch.stack(
            [batch_item.input_token_ids for batch_item in batch]
        )  # don't move to cuda, otherwise background data loader processes will not work
        target_token_ids = torch.stack(
            [batch_item.target_token_ids for batch_item in batch]
        )  # don't move to cuda, otherwise background data loader processes will not work
        position_ids = torch.stack(
            [batch_item.position_ids for batch_item in batch]
        )  # don't move to cuda, otherwise background data loader processes will not work
        loss_weights = torch.stack(
            [batch_item.loss_weights for batch_item in batch]
        )  # don't move to cuda, otherwise background data loader processes will not work

        cumulative_seq_lengths = get_cumulative_seq_lengths(input_token_ids, reset_attention_mask=False)

        if any(batch_item.input_images is not None and len(batch_item.input_images) > 0 for batch_item in batch):
            input_images_list = list()
            input_image_locations_list = list()
            for batch_item_index, batch_item in enumerate(batch):
                if batch_item.input_images is None:
                    batch_item.input_images = list()
                if batch_item.input_image_locations is None:
                    batch_item.input_image_locations = list()
                assert len(batch_item.input_images) == len(batch_item.input_image_locations)
                for img, (start, end) in zip(batch_item.input_images, batch_item.input_image_locations):
                    input_images_list.append(img)
                    input_image_locations_list.append([batch_item_index, start, end])

            input_images = torch.stack(input_images_list, dim=0)
            input_image_locations = torch.tensor(input_image_locations_list, dtype=torch.long)
        else:
            input_images = None
            input_image_locations = None

        return TextDatasetBatch(
            input_token_ids=input_token_ids,
            target_token_ids=target_token_ids,
            cumulative_seq_lengths=cumulative_seq_lengths,
            position_ids=position_ids,
            loss_weights=loss_weights,
            input_images=input_images,
            input_image_locations=input_image_locations,
        )

    @staticmethod
    def sync_batch_to_model_parallel(
        topology: Optional[Topology],
        batch: Optional[TextDatasetBatch],
    ) -> TextDatasetBatch:
        if topology is None:
            assert batch is not None
            return batch

        if topology.model_parallel_rank == 0:
            assert batch is not None
            batch.contiguous_()
            assert batch.input_token_ids is not None
            assert batch.target_token_ids is not None
            assert batch.cumulative_seq_lengths_padded is not None
            assert batch.position_ids is not None
            long_tensors: List[Optional[torch.Tensor]] = [
                batch.input_token_ids,
                batch.target_token_ids,
                batch.cumulative_seq_lengths_padded.to(torch.long),
                batch.position_ids,
            ]

            if batch.input_image_locations is None:
                long_tensors.append(torch.tensor([-1], dtype=torch.long))
            else:
                long_tensors.append(batch.input_image_locations)

            float_tensors: List[Optional[torch.Tensor]] = [batch.loss_weights]
            if batch.input_images is None:
                float_tensors.append(torch.tensor([-1.0], dtype=torch.float))
            else:
                float_tensors.append(batch.input_images)

        else:
            assert batch is None
            long_tensors = [None, None, None, None, None]
            float_tensors = [None, None]

        broadcast_long_tensors = broadcast_data(tensors=long_tensors, dtype=torch.long, topology=topology)  # type: ignore
        broadcast_float_tensors = broadcast_data(tensors=float_tensors, dtype=torch.float32, topology=topology)  # type: ignore

        input_image_locations_ = broadcast_long_tensors[4]
        input_images_ = broadcast_float_tensors[1]
        if (input_image_locations_ == -1).all():
            input_image_locations = None
            input_images = None
        else:
            input_image_locations = input_image_locations_
            input_images = input_images_

        cumulative_seq_lengths_padded = broadcast_long_tensors[2].to(torch.int32)

        return TextDatasetBatch(
            input_token_ids=broadcast_long_tensors[0],
            target_token_ids=broadcast_long_tensors[1],
            cumulative_seq_lengths_padded=cumulative_seq_lengths_padded,
            position_ids=broadcast_long_tensors[3],
            loss_weights=broadcast_float_tensors[0],
            input_images=input_images,
            input_image_locations=input_image_locations,
        )


class TextImageBlendedDataset(
    BaseBlendedDataset[
        TextImageDatasetItem,
        TextDatasetBatch,
        TextDatasetBatch,
        TextImageDataset,
    ]
):
    pass
