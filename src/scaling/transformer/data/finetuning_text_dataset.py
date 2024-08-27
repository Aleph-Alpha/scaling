import hashlib
import json
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image  # type: ignore

from scaling.core import (
    BaseBlendedDataset,
    BaseDataset,
    BaseDatasetItem,
    MemoryMapDataset,
    MemoryMapDatasetBuilder,
    Topology,
    broadcast_data,
)
from scaling.transformer.data.utils import get_cumulative_seq_lengths
from scaling.transformer.model.image_encoder import clip_transform
from scaling.transformer.tokenizer import Tokenizer

from .text_dataset_batch import TextDatasetBatch

IMAGE_ENCODER_TOKEN_COUNTS = 144  # TODO so far this is constant
IMAGE_TRANSFORM_FN = clip_transform((384, 384))  # TODO so far this is constant


class FinetuningTextDatasetItem(BaseDatasetItem):
    input_token_ids: torch.Tensor
    target_token_ids: torch.Tensor
    cumulative_seq_lengths: torch.Tensor
    position_ids: torch.Tensor
    loss_weights: torch.Tensor
    input_images: list[torch.Tensor] | None
    input_image_locations: list[tuple[int, int]] | None

    def __init__(
        self,
        input_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        cumulative_seq_lengths: torch.Tensor,
        position_ids: torch.Tensor,
        loss_weights: torch.Tensor,
        input_images: list[torch.Tensor] | None = None,
        input_image_locations: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.input_token_ids = input_token_ids
        self.target_token_ids = target_token_ids
        self.cumulative_seq_lengths = cumulative_seq_lengths
        self.position_ids = position_ids
        self.loss_weights = loss_weights
        self.input_images = input_images
        self.input_image_locations = input_image_locations


class FinetuningTextDataset(BaseDataset[FinetuningTextDatasetItem, TextDatasetBatch, TextDatasetBatch]):
    """
    Torch dataset providing tokenized text of always the same sequence length; data is loaded from a memory map.

    every FinetuningTextDatasetItem in the dataset:

    """

    def __init__(
        self,
        data_prefix: Path,
        sequence_length: int,
        seed: int,
        softprompt_n_tokens: int,
        tokenizer: Tokenizer,
        tokenizer_no_prefix_space: Tokenizer,
        memory_map_dataset: bool = False,
        shuffle: bool = True,
    ):
        """
        data_prefix (`Path`)
            path to a memory map

        sequence_length (`int`)
            expected sequence length of the token_ids in output.
        """
        # remember params
        self.data_prefix = Path(data_prefix)
        self.data_prefix_parent = Path(data_prefix).parent
        self.sequence_length = sequence_length
        self.softprompt_n_tokens = softprompt_n_tokens
        self.tokenizer = tokenizer
        self.tokenizer_no_prefix_space = tokenizer_no_prefix_space
        self.memory_map_dataset = memory_map_dataset

        self.dataset: Union[None, MemoryMapDataset]
        if self.memory_map_dataset:
            self.dataset = MemoryMapDataset(prefix_path=self.data_prefix)
            self.data: list[int] = list(range(len(self.dataset)))
        else:
            self.data: list[dict[str, Any]] = json.load(open(self.data_prefix, "r", encoding="UTF-8"))  # type: ignore[no-redef]
            self.dataset = None

        # shuffling
        self.seed: int | None = None
        self.data_item_index: list | None = None

        super().__init__(seed=seed, shuffle=shuffle)

    def ident(self) -> str:
        md5_hash = hashlib.md5()
        md5_hash.update(str(self.data_prefix).encode("utf-8"))

        # Memory map dataset does not use tokenizers
        if not self.memory_map_dataset:
            md5_hash.update(
                json.dumps(self.tokenizer.tokenizer.get_vocab(), sort_keys=True, default=str).encode("utf-8")
            )
            md5_hash.update(
                json.dumps(
                    self.tokenizer_no_prefix_space.tokenizer.get_vocab(),
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            )

        return f"{md5_hash.hexdigest()}-seq-{self.sequence_length}"

    def set_seed(self, seed: int, shuffle: bool = True) -> None:
        """
        Computes and index on the tokenized dataset such that each item in  the index

        - results in a batch item of sequence length
        - no padding token is used
        - longer texts are split into different data items
        - shorter texts are concatenated and separated with an end of text token
        """
        # exit if we already computed the data index
        if self.seed is not None and self.seed == seed:
            return

        random.seed(seed)
        if shuffle:
            random.shuffle(self.data)

        self.seed = seed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> FinetuningTextDatasetItem:
        eos_token_id: int = self.tokenizer.eos_token_id
        prompt_token_ids: list[int]
        completion_token_ids: list[int]
        if self.memory_map_dataset:
            prompt_token_ids, completion_token_ids = self.get_memory_map_token_ids(index=index)
            prompt_images_transformed = None
            prompt_image_locations = None
        else:
            (
                prompt_token_ids,
                completion_token_ids,
                prompt_images_transformed,
                prompt_image_locations,
            ) = self.get_json_token_ids(index=index)

        if self.softprompt_n_tokens > 0:
            prompt_token_ids = [0] * self.softprompt_n_tokens + prompt_token_ids

            # shift image locations
            if prompt_image_locations is not None:
                prompt_image_locations_: list[tuple[int, int]] = list()
                for start, end in prompt_image_locations:
                    prompt_image_locations_.append(
                        (
                            start + self.softprompt_n_tokens,
                            end + self.softprompt_n_tokens,
                        )
                    )
                prompt_image_locations = prompt_image_locations_

        padding_size = self.sequence_length - len(prompt_token_ids) - len(completion_token_ids) + 1

        token_ids = prompt_token_ids + completion_token_ids + [eos_token_id] * padding_size
        if len(token_ids) > self.sequence_length + 1:
            token_ids = token_ids[: (self.sequence_length + 1)]

        # retrieve token ids from memory map
        input_token_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_token_ids = torch.tensor(token_ids[1:], dtype=torch.long)
        cumulative_seq_lengths = torch.tensor([0, self.sequence_length], dtype=torch.int32)
        position_ids = torch.arange(0, self.sequence_length)

        loss_weights = torch.ones(input_token_ids.shape, dtype=torch.float)
        loss_weights[: len(prompt_token_ids) - 1] = 0

        # Mask out all the padding token. But still keep the first one because we are padding with the EOL token id,
        # and we still want to predict a single EOL token after the sequence.
        if (padding_size - 1) > 0:
            loss_weights[-(padding_size - 1) :] = 0

        return FinetuningTextDatasetItem(
            input_token_ids=input_token_ids,
            target_token_ids=target_token_ids,
            cumulative_seq_lengths=cumulative_seq_lengths,
            position_ids=position_ids,
            loss_weights=loss_weights,
            input_images=prompt_images_transformed,
            input_image_locations=prompt_image_locations,
        )

    def get_memory_map_token_ids(self, index: int) -> tuple[list[int], list[int]]:
        src_index = self.data[index]
        assert self.dataset is not None
        src_token_ids = self.dataset[src_index].tolist()
        len_prompt_token_ids = src_token_ids[0]
        prompt_token_ids = src_token_ids[1 : len_prompt_token_ids + 1]
        completion_token_ids = src_token_ids[len_prompt_token_ids + 1 :]

        return prompt_token_ids, completion_token_ids

    def get_json_token_ids(
        self, index: int
    ) -> tuple[
        list[int],
        list[int],
        list[torch.Tensor] | None,
        list[tuple[int, int]] | None,
    ]:
        data_item = self.data[index]

        prompt_images_transformed: list[torch.Tensor] | None = None
        prompt_image_locations: list[tuple[int, int]] | None = None
        # tokenize prompt
        if isinstance(data_item["prompt"], list):  # type: ignore[index]
            prompt_token_ids: list[int] = list()
            prompt_images_transformed = list()
            prompt_image_locations = list()
            try:
                for i, p in enumerate(data_item["prompt"]):  # type: ignore[index]
                    assert isinstance(p, str)
                    # if p.endswith(".jpg"):
                    #     assert (self.data_prefix_parent / p).is_file(), f"not found {p}"
                    if p.endswith(".jpg") and (self.data_prefix_parent / p).is_file():
                        image = Image.open(str((self.data_prefix_parent / p)))
                        prompt_images_transformed.append(IMAGE_TRANSFORM_FN(image))
                        prompt_image_locations.append(
                            (
                                len(prompt_token_ids),
                                len(prompt_token_ids) + IMAGE_ENCODER_TOKEN_COUNTS,
                            )
                        )
                        prompt_token_ids.extend([self.tokenizer.eos_token_id] * IMAGE_ENCODER_TOKEN_COUNTS)
                    else:
                        if i == 0:
                            prompt_token_ids.extend(self.tokenizer.encode(p))
                        else:
                            prompt_token_ids.extend(self.tokenizer_no_prefix_space.encode(p))
            except Exception as e:
                if "File name too long" in str(e):
                    if i == 0:
                        prompt_token_ids.extend(self.tokenizer.encode(p))
                    else:
                        prompt_token_ids.extend(self.tokenizer_no_prefix_space.encode(p))
                else:
                    raise e
        else:
            assert isinstance(data_item["prompt"], str)  # type: ignore[index]
            prompt_token_ids = self.tokenizer.encode(data_item["prompt"])  # type: ignore[index]

        # tokenize completion
        completion_token_ids = self.tokenizer.encode(data_item["completion"])  # type: ignore[index]

        return (
            prompt_token_ids,
            completion_token_ids,
            prompt_images_transformed,
            prompt_image_locations,
        )

    def collate(self, batch: list[FinetuningTextDatasetItem]) -> TextDatasetBatch:
        """
        Used to collate lists of samples into batches
        The default implementation returns a BaseDataBatch NamedTuple with the same attributes as the first element
        of the batch
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
        topology: Topology | None,
        batch: TextDatasetBatch | None,
    ) -> TextDatasetBatch:
        if topology is None:
            assert batch is not None
            return batch

        if topology.model_parallel_rank == 0:
            assert batch is not None
            batch.contiguous_()
            assert batch is not None
            assert batch.input_token_ids is not None
            assert batch.target_token_ids is not None
            assert batch.cumulative_seq_lengths_padded is not None
            assert batch.position_ids is not None
            long_tensors: list[torch.Tensor | None] = [
                batch.input_token_ids,
                batch.target_token_ids,
                batch.cumulative_seq_lengths_padded.to(torch.long),
                batch.position_ids,
            ]

            if batch.input_image_locations is None:
                long_tensors.append(torch.tensor([-1], dtype=torch.long))
            else:
                long_tensors.append(batch.input_image_locations)

            float_tensors: list[torch.Tensor | None] = [batch.loss_weights]
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

    @staticmethod
    def convert_jsonl(
        jsonl_file: Union[str, Path],
        tokenizer: Tokenizer,
        tokenizer_no_prefix_space: Tokenizer,
        out_prefix_path: Union[str, Path],
    ) -> None:
        # instantiate a builder to write data
        builder = MemoryMapDatasetBuilder(prefix_path=Path(out_prefix_path))

        with open(jsonl_file, "r", encoding="UTF-8") as in_file:
            for line in in_file:
                line = line.strip()
                if line == "":
                    continue

                line_data = json.loads(line)
                prompt_token_ids = tokenizer.encode(line_data["prompt"])
                completion_token_ids = tokenizer_no_prefix_space.encode(line_data["completion"])

                data_item = [len(prompt_token_ids)] + prompt_token_ids + completion_token_ids
                builder.add(np_array=np.array(data_item))

        builder.finalize()


class FinetuningTextBlendedDataset(
    BaseBlendedDataset[
        FinetuningTextDatasetItem,
        TextDatasetBatch,
        TextDatasetBatch,
        FinetuningTextDataset,
    ]
):
    pass
