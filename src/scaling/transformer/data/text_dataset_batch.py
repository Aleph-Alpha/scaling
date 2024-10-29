from typing import Any

import torch

from scaling.core import BaseDatasetBatch
from scaling.transformer.data.utils import (
    add_cumulative_seq_lengths_padding,
    get_cumulative_seq_lengths,
    get_position_ids,
    remove_cumulative_seq_lengths_padding,
)

from .inference_settings import InferenceSettings


class TextDatasetBatchBeforeSync(BaseDatasetBatch):
    token_ids: torch.Tensor

    def __init__(self, token_ids: torch.Tensor, reset_attention_mask: bool = True, reset_position_ids: bool = True):
        self.token_ids = token_ids
        self.reset_attention_mask = reset_attention_mask
        self.reset_position_ids = reset_position_ids

    def only_inputs(self) -> "TextDatasetBatchBeforeSync":
        return self

    def only_targets(self) -> "TextDatasetBatchBeforeSync":
        return self


class TextDatasetBatch(BaseDatasetBatch):
    input_token_ids: torch.Tensor | None  # also used in inference
    input_images: torch.Tensor | None  # also used in inference
    input_image_locations: torch.Tensor | None  # just used for finetuning
    target_token_ids: torch.Tensor | None
    position_ids: torch.Tensor | None  # also used in inference
    cumulative_seq_lengths: torch.Tensor | None
    cumulative_seq_lengths_padded: torch.Tensor | None  # only used for pipe communication
    loss_weights: (
        torch.Tensor | None
    )  # also used in inference for embedding weighting, it serves as an attention mask on the input excluding padding
    inference_settings: InferenceSettings | None  # only used in inference
    embeddings: torch.Tensor | None  # only used in inference

    @staticmethod
    def field_names() -> list[str]:
        return [
            "input_token_ids",
            "input_images",
            "input_image_locations",
            "target_token_ids",
            "position_ids",
            "cumulative_seq_lengths",
            "cumulative_seq_lengths_padded",
            "loss_weights",
            "inference_settings",
            "embeddings",
        ]

    def as_tuple(self) -> tuple[Any, ...]:
        field_names = self.field_names()
        attr_list = list()
        field_names_communicated = list()
        for field_name in field_names:
            attr = getattr(self, field_name)
            if attr is not None:
                attr_list.append(attr)
                field_names_communicated.append(field_name)

        return tuple(attr_list + [field_names_communicated])

    @classmethod
    def from_tuple(cls, d: tuple[Any, ...]) -> "TextDatasetBatch":
        """
        convert a tuple with tensors as values for pipe communication to the layer input class
        you might need to merge split tensors again
        """
        field_names_communicated = d[-1]
        attr_list = d[:-1]
        assert len(attr_list) == len(field_names_communicated)
        attr_dict = {field_name: attr for field_name, attr in zip(field_names_communicated, attr_list)}
        return cls(**attr_dict)

    def __init__(
        self,
        input_token_ids: torch.Tensor | None = None,
        input_images: torch.Tensor | None = None,
        input_image_locations: torch.Tensor | None = None,
        target_token_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cumulative_seq_lengths: torch.Tensor | None = None,
        cumulative_seq_lengths_padded: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
        inference_settings: InferenceSettings | None = None,
        embeddings: torch.Tensor | None = None,
        reset_attention_mask: bool = True,
        reset_position_ids: bool = True,
    ) -> None:
        self.input_token_ids = input_token_ids
        self.input_images = input_images
        self.input_image_locations = input_image_locations
        self.target_token_ids = target_token_ids
        self.position_ids = position_ids
        self.cumulative_seq_lengths = cumulative_seq_lengths
        self.cumulative_seq_lengths_padded = cumulative_seq_lengths_padded
        self.loss_weights = loss_weights
        self.inference_settings = inference_settings
        self.embeddings = embeddings

        if self.input_token_ids is not None:
            if self.cumulative_seq_lengths is None and self.cumulative_seq_lengths_padded is None:
                self.cumulative_seq_lengths = get_cumulative_seq_lengths(
                    self.input_token_ids, reset_attention_mask=reset_attention_mask
                )
            elif self.cumulative_seq_lengths_padded is not None and self.cumulative_seq_lengths is None:
                # after syncing to model parallel we only have the padded tensor
                self.cumulative_seq_lengths = remove_cumulative_seq_lengths_padding(self.cumulative_seq_lengths_padded)
            if self.cumulative_seq_lengths_padded is None and self.cumulative_seq_lengths is not None:
                micro_batch_size, sequence_length = self.input_token_ids.size()
                self.cumulative_seq_lengths_padded = add_cumulative_seq_lengths_padding(
                    self.cumulative_seq_lengths, micro_batch_size * (sequence_length + 1)
                )
            if self.position_ids is None:
                self.position_ids = get_position_ids(
                    input_token_ids=self.input_token_ids, reset_position_ids=reset_position_ids
                )
            if self.loss_weights is None:
                self.loss_weights = torch.ones_like(self.input_token_ids, dtype=torch.float).contiguous()

    def only_inputs(self) -> "TextDatasetBatch":
        """
        function removing all properties from batch that are not inputs (i.e. targets)
        this may be useful to reduce memory load
        """
        return TextDatasetBatch(
            input_token_ids=self.input_token_ids,
            input_images=self.input_images,
            input_image_locations=self.input_image_locations,
            position_ids=self.position_ids,
            cumulative_seq_lengths=self.cumulative_seq_lengths,
            loss_weights=self.loss_weights,
        )

    def only_targets(self) -> "TextDatasetBatch":
        """
        function removing all properties from batch that are not targets (i.e. inputs)
        this may be useful to reduce memory load
        """
        return TextDatasetBatch(target_token_ids=self.target_token_ids, loss_weights=self.loss_weights)
