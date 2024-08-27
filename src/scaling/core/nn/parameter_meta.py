from typing import Any, Optional, TypedDict

import torch


class ParameterMetaState(TypedDict):
    local_shape: tuple[int, ...]
    is_model_parallel: bool
    model_parallel_dimension: Optional[int]
    layer_index: Optional[int]
    parameter_name: Optional[str]
    is_tied: bool
    tied_layer_indices: set[int]
    tied_grad_on_model_parallel: bool


class CoreParameterMeta:
    def __init__(
        self,
        local_shape: tuple[int, ...],
        is_model_parallel: bool,
        model_parallel_dimension: Optional[int] = None,
        layer_index: Optional[int] = None,
        parameter_name: Optional[str] = None,
        layer_class_name: Optional[str] = None,
        is_tied: bool = False,
        tied_layer_indices: Optional[set[int]] = None,
        tied_grad_on_model_parallel: bool = False,
    ):
        self.local_shape = local_shape
        self.is_model_parallel = is_model_parallel
        self.model_parallel_dimension = model_parallel_dimension
        self.layer_index = layer_index
        self.parameter_name = parameter_name
        self.layer_class_name = layer_class_name
        self.is_tied = is_tied
        self.tied_layer_indices: set[int] = set() if tied_layer_indices is None else tied_layer_indices
        if layer_index is not None:
            self.set_layer_index(layer_index=layer_index)

        self.tied_grad_on_model_parallel = tied_grad_on_model_parallel

    def __repr__(self) -> str:
        return (
            f"CoreParameterMeta [{self.parameter_name}] layer_index [{self.layer_index}] "
            f"layer_class_name [{self.layer_class_name}] is_model_parallel [{self.is_model_parallel}]"
        )

    @property
    def is_model_parallel_duplicate(self) -> bool:
        return not self.is_model_parallel

    @property
    def key(self) -> str:
        """
        unique identifier within a constant model architecture independent of layout
        """
        return self.key_for_layer(self.layer_index)  # type: ignore[arg-type]

    def key_for_layer(self, layer_index: int) -> str:
        return (
            f"layer_index_{layer_index}_parameter_name_{self.parameter_name}_"
            f"is_model_parallel_{self.is_model_parallel}_"
            f"model_parallel_dimension_{self.model_parallel_dimension}"
        )

    def possible_keys(self) -> list[str]:
        if self.is_tied:
            return [self.key_for_layer(i) for i in self.tied_layer_indices]
        else:
            return [self.key]

    def state_dict(self) -> ParameterMetaState:
        return {
            "local_shape": self.local_shape,
            "is_model_parallel": self.is_model_parallel,
            "model_parallel_dimension": self.model_parallel_dimension,
            "layer_index": self.layer_index,
            "parameter_name": self.parameter_name,
            "is_tied": self.is_tied,
            "tied_layer_indices": self.tied_layer_indices,
            "tied_grad_on_model_parallel": self.tied_grad_on_model_parallel,
        }

    @classmethod
    def from_state_dict(cls, state_dict: ParameterMetaState) -> "CoreParameterMeta":
        return cls(**state_dict)

    def set_layer_index(self, layer_index: int) -> None:
        self.layer_index = layer_index

        if self.is_tied:
            self.tied_layer_indices.add(layer_index)

    def set_parameter_name(self, parameter_name: str) -> None:
        self.parameter_name = parameter_name

    def set_layer_class_name(self, layer_class_name: str) -> None:
        self.layer_class_name = layer_class_name

    def set_is_tied(self, is_tied: bool) -> None:
        self.is_tied = is_tied

    def set(
        self,
        layer_index: int,
        parameter_name: str,
        layer_class_name: str,
        is_tied: bool,
    ) -> None:
        self.set_is_tied(is_tied)
        self.set_layer_index(layer_index)
        self.set_parameter_name(parameter_name)
        self.set_layer_class_name(layer_class_name)

    @staticmethod
    def register_on_parameter(
        parameter: torch.Tensor,
        is_model_parallel: bool,
        model_parallel_dimension: Optional[int] = None,
        layer_index: Optional[int] = None,
        parameter_name: Optional[str] = None,
        layer_class_name: Optional[str] = None,
        is_tied: bool = False,
        tied_grad_on_model_parallel: bool = False,
    ) -> "CoreParameterMeta":
        assert not hasattr(parameter, "core_parameter_meta"), "core_parameter_meta already registered"

        local_shape = tuple(parameter.shape)

        meta = CoreParameterMeta(
            local_shape=local_shape,
            is_model_parallel=is_model_parallel,
            model_parallel_dimension=model_parallel_dimension,
            layer_index=layer_index,
            parameter_name=parameter_name,
            layer_class_name=layer_class_name,
            is_tied=is_tied,
            tied_grad_on_model_parallel=tied_grad_on_model_parallel,
        )

        parameter.core_parameter_meta = meta  # type: ignore

        return meta

    def __eq__(self, o: Any) -> bool:
        if not isinstance(o, CoreParameterMeta):
            return False

        return self.key == o.key
