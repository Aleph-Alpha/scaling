import hashlib
import json
import random
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image  # type: ignore

from scaling.core import (
    BaseBlendedDataset,
    BaseDataset,
    Topology,
    broadcast_data,
)
from scaling.transformer.data.utils import get_cumulative_seq_lengths
from scaling.transformer.model.image_encoder import clip_transform
from scaling.transformer.tokenizer import Tokenizer

from .dataset_item import TextImageDatasetItem
from .text_dataset_batch import TextDatasetBatch

IMAGE_ENCODER_TOKEN_COUNTS = 144  # TODO so far this is constant
IMAGE_TRANSFORM_FN = clip_transform((384, 384))  # TODO so far this is constant


class FinetuningChatDataset(BaseDataset[TextImageDatasetItem, TextDatasetBatch, TextDatasetBatch]):
    """
    Torch dataset providing tokenized text of always the same sequence length; data is loaded from a memory map.

    every FinetuningChatDataset in the dataset:

    """

    def __init__(
        self,
        data_path: Path,
        sequence_length: int,
        seed: int,
        softprompt_n_tokens: int,
        tokenizer: Tokenizer,
        tokenizer_no_prefix_space: Tokenizer,
        shuffle: bool = True,
    ):
        """
        data_prefix (`Path`)
            path to a memory map

        sequence_length (`int`)
            expected sequence length of the token_ids in output.
        """
        # remember params
        self.data_path = Path(data_path)
        self.data_path_parent = Path(data_path).parent
        self.sequence_length = sequence_length
        self.softprompt_n_tokens = softprompt_n_tokens
        self.tokenizer = tokenizer
        self.tokenizer_no_prefix_space = tokenizer_no_prefix_space

        with open(self.data_path, "r", encoding="UTF-8") as r:
            self.data_jsonl: list[list[dict[str, Any]]] = [json.loads(s) for s in str(r.read()).split("\n") if s != ""]  # type: ignore[no-redef]

        self.data: list[dict[str, Any]] = []  # type: ignore[no-redef]

        # shuffling
        self.seed: Optional[int] = None
        self.data_item_index: Optional[list] = None

        self.load_data()

        super().__init__(seed=seed, shuffle=shuffle)

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

    def load_data(self) -> None:
        for d_json in self.data_jsonl:
            token_list: list[int] = []
            loss_mask_list: list[int] = []
            prompt_images_path: Optional[list[Path]] = None
            prompt_image_locations: Optional[list[tuple[int, int]]] = None

            first_text: bool = True

            for element in d_json:
                content: str = element["content"]
                type: str = element["type"]
                has_loss: bool = element.get("has_loss", False)

                if type == "text":
                    if first_text:
                        token: list[int] = self.tokenizer.encode(content)
                    else:
                        token = self.tokenizer_no_prefix_space.encode(content)

                    token_list.extend(token)
                    loss_mask_list.extend([int(has_loss)] * len(token))

                    first_text = False

                elif type == "image":
                    if prompt_images_path is None:
                        prompt_images_path = list()

                    if prompt_image_locations is None:
                        prompt_image_locations = list()

                    prompt_images_path.append((self.data_path / content))
                    prompt_image_locations.append(
                        (
                            len(token_list),
                            len(token_list) + IMAGE_ENCODER_TOKEN_COUNTS,
                        )
                    )
                    token_list.extend([self.tokenizer.padding_token_id] * IMAGE_ENCODER_TOKEN_COUNTS)

                else:
                    raise NotImplementedError(f"Content type {type} is not supported")

            input_token_list: list[int] = token_list[:-1]
            target_token_list: list[int] = token_list[1:]
            loss_mask_list = loss_mask_list[1:]

            self.data.append(
                {
                    "input_token_list": input_token_list,
                    "target_token_list": target_token_list,
                    "loss_mask_list": loss_mask_list,
                    "prompt_images_path": prompt_images_path,
                    "prompt_image_locations": prompt_image_locations,
                }
            )

    def ident(self) -> str:
        md5_hash = hashlib.md5()
        md5_hash.update(str(self.data_path).encode("utf-8"))

        md5_hash.update(json.dumps(self.tokenizer.tokenizer.get_vocab(), sort_keys=True, default=str).encode("utf-8"))
        md5_hash.update(
            json.dumps(
                self.tokenizer_no_prefix_space.tokenizer.get_vocab(),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        )

        return f"{md5_hash.hexdigest()}-seq-{self.sequence_length}"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> TextImageDatasetItem:
        padding_token_id: int = self.tokenizer.padding_token_id
        data = self.data[index]

        input_token_list: list[int] = data["input_token_list"]
        target_token_list: list[int] = data["target_token_list"]
        loss_mask_list: list[int] = data["loss_mask_list"]
        prompt_images_path: Optional[list[Path]] = data["prompt_images_path"]
        prompt_image_locations: Optional[list[tuple[int, int]]] = data["prompt_image_locations"]
        prompt_images_transformed: Optional[list[torch.Tensor]] = None

        if self.softprompt_n_tokens > 0:
            input_token_list = [0] * self.softprompt_n_tokens + input_token_list
            target_token_list = [0] * self.softprompt_n_tokens + target_token_list
            loss_mask_list = [0] * self.softprompt_n_tokens + loss_mask_list

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

        padding_size = self.sequence_length - len(input_token_list)

        input_token_list = input_token_list + [padding_token_id] * padding_size
        target_token_list = target_token_list + [padding_token_id] * padding_size
        loss_mask_list = loss_mask_list + [0] * padding_size

        input_token_list = input_token_list[: self.sequence_length]
        target_token_list = target_token_list[: self.sequence_length]
        loss_mask_list = loss_mask_list[: self.sequence_length]

        input_token_ids = torch.tensor(input_token_list, dtype=torch.long)
        target_token_ids = torch.tensor(target_token_list, dtype=torch.long)
        loss_weights = torch.tensor(loss_mask_list, dtype=torch.float32)

        if prompt_images_path is not None:
            prompt_images_transformed = list()

            for images_path in prompt_images_path:
                image = Image.open(str((self.data_path_parent / images_path)))
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

    def collate(self, batch: list[TextImageDatasetItem]) -> TextDatasetBatch:
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
            long_tensors: list[Optional[torch.Tensor]] = [
                batch.input_token_ids,
                batch.target_token_ids,
                batch.cumulative_seq_lengths_padded.to(torch.long),
                batch.position_ids,
            ]

            if batch.input_image_locations is None:
                long_tensors.append(torch.tensor([-1], dtype=torch.long))
            else:
                long_tensors.append(batch.input_image_locations)

            float_tensors: list[Optional[torch.Tensor]] = [batch.loss_weights]
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


class FinetuningChatBlendedDataset(
    BaseBlendedDataset[
        TextImageDatasetItem,
        TextDatasetBatch,
        TextDatasetBatch,
        FinetuningChatDataset,
    ]
):
    pass
