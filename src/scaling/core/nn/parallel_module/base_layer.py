from abc import abstractmethod
from typing import Any, Generic, TypeVar

import torch

from ...data import BaseDatasetBatch, BaseLayerIO

BaseLossInputGeneric = TypeVar("BaseLossInputGeneric")
BaseLossOutputGeneric = TypeVar("BaseLossOutputGeneric", torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]])
BaseDatasetBatchGeneric = TypeVar("BaseDatasetBatchGeneric", bound=BaseDatasetBatch)
BaseLayerInputGeneric = TypeVar("BaseLayerInputGeneric")
BaseLayerOutputGeneric = TypeVar("BaseLayerOutputGeneric", bound=BaseLayerIO)
BaseLayerLastLayerOutputGeneric = TypeVar("BaseLayerLastLayerOutputGeneric", bound=BaseLayerIO)


class BaseLayer(
    torch.nn.Module,
    Generic[BaseLayerInputGeneric, BaseLayerOutputGeneric, BaseLayerLastLayerOutputGeneric],
):
    @abstractmethod
    def forward(self, x: BaseLayerInputGeneric) -> BaseLayerOutputGeneric:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def input_to_tuple(
        input: BaseLayerInputGeneric,
    ) -> tuple[Any, ...]:
        """
        convert layer input to a tuple with tensors as values for pipe communication and activation checkpointing
        this may include a split to model parallel
        tuple_to_input will be called on the tuple, here you might need to merge split tensors again
        we are using a tuple because torch requires tuples for activation checkpointing
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tuple_to_input(d: tuple[Any, ...]) -> BaseLayerInputGeneric:
        """
        convert a tuple with tensors as values for pipe communication to the layer input class
        you might need to merge split tensors again
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def output_to_tuple(
        output: BaseLayerOutputGeneric,
    ) -> tuple[Any, ...]:
        """
        convert layer output to a tuple with tensors as values for pipe communication and activation checkpointing
        this may include a split to model parallel
        tuple_to_input will be called on the tuple, here you might need to merge split tensors again
        we are using a tuple because torch requires tuples for activation checkpointing
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tuple_to_last_stage_activation(d: tuple[Any, ...]) -> BaseLayerLastLayerOutputGeneric:
        """
        convert a tuple with tensors as values for pipe communication to the last layer's output class
        you might need to merge split tensors again
        """
        raise NotImplementedError

    def _forward_tuple_input(self, *args: tuple) -> torch.Tensor:
        x = self.tuple_to_input(tuple(args))
        return self(x)
