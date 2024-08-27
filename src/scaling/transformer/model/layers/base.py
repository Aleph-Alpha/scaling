from typing import Any

import torch

from scaling.core import (
    BaseLayer,
    BaseLayerIO,
)
from scaling.transformer.data.inference_settings import InferenceSettings


class TransformerLayerIO(BaseLayerIO):
    activations: torch.Tensor
    position_ids: torch.Tensor
    cumulative_seq_lengths: torch.Tensor | None
    cumulative_seq_lengths_padded: torch.Tensor  # only used for pipe communication
    loss_weights: torch.Tensor | None
    inference_settings: InferenceSettings | None
    embeddings: torch.Tensor | None
    embeddings_head: torch.Tensor | None
    attention_scores_manipulation: torch.Tensor | None

    @staticmethod
    def field_names() -> list[str]:
        return [
            "activations",
            "position_ids",
            "cumulative_seq_lengths",
            "cumulative_seq_lengths_padded",
            "loss_weights",
            "inference_settings",
            "embeddings",
            "embeddings_head",
            "attention_scores_manipulation",
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
    def from_tuple(cls, d: tuple[Any, ...]) -> "TransformerLayerIO":
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
        activations: torch.Tensor,
        position_ids: torch.Tensor,
        cumulative_seq_lengths_padded: torch.Tensor,
        cumulative_seq_lengths: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
        inference_settings: InferenceSettings | None = None,
        embeddings: torch.Tensor | None = None,
        embeddings_head: torch.Tensor | None = None,
        attention_scores_manipulation: torch.Tensor | None = None,
    ) -> None:
        self.activations = activations
        self.position_ids = position_ids
        self.cumulative_seq_lengths = cumulative_seq_lengths
        self.cumulative_seq_lengths_padded = cumulative_seq_lengths_padded
        self.loss_weights = loss_weights
        self.inference_settings = inference_settings
        self.embeddings = embeddings
        self.embeddings_head = embeddings_head
        self.attention_scores_manipulation = attention_scores_manipulation


class TransformerLayerBaseIO(BaseLayer[TransformerLayerIO, TransformerLayerIO, TransformerLayerIO]):
    @staticmethod
    def input_to_tuple(
        input: TransformerLayerIO,
    ) -> tuple[Any, ...]:
        """
        convert layer input to a tuple with tensors as values for pipe communication and activation checkpointing
        this may include a split to model parallel
        tuple_to_input will be called on the tuple, here you might need to merge split tensors again
        we are using a tuple because torch requires tuples for activation checkpointing
        """
        return input.as_tuple()

    @staticmethod
    def tuple_to_input(d: tuple[Any, ...]) -> TransformerLayerIO:
        """
        convert a tuple with tensors as values for pipe communication to the layer input class
        you might need to merge split tensors again
        """
        return TransformerLayerIO.from_tuple(d)

    @staticmethod
    def output_to_tuple(
        output: TransformerLayerIO,
    ) -> tuple[Any, ...]:
        """
        convert layer output to a tuple with tensors as values for pipe communication and activation checkpointing
        this may include a split to model parallel
        tuple_to_input will be called on the tuple, here you might need to merge split tensors again
        we are using a tuple because torch requires tuples for activation checkpointing
        """
        return output.as_tuple()

    @staticmethod
    def tuple_to_last_stage_activation(d: tuple[Any, ...]) -> TransformerLayerIO:
        """
        convert a tuple with tensors as values for pipe communication to the last layer's output class
        you might need to merge split tensors again
        """
        return TransformerLayerIO.from_tuple(d)
