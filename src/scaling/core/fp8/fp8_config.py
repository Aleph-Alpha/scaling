from enum import Enum

import torch
from pydantic import Field, model_validator

from scaling.core.config import BaseConfig


class FP8DType(str, Enum):
    """
    More info about the different FP8 formats can be found here https://asawicki.info/articles/fp8_tables.php
    """

    E4M3 = "e4m3"
    E5M2 = "e5m2"

    @property
    def torch_dtype(self) -> torch.dtype:
        match self:
            case FP8DType.E4M3:
                return torch.float8_e4m3fn
            case FP8DType.E5M2:
                return torch.float8_e5m2
            case _:
                raise NotImplementedError()


class FP8MatmulConfig(BaseConfig):
    left_dtype: FP8DType = Field(FP8DType.E4M3, description="Data type for the left matrix in a matrix-matrix product")
    right_dtype: FP8DType = Field(
        FP8DType.E4M3, description="Data type for the right matrix in a matrix-matrix product"
    )

    @property
    def torch_dtypes(self) -> tuple[torch.dtype, torch.dtype]:
        return self.left_dtype.torch_dtype, self.right_dtype.torch_dtype

    @model_validator(mode="after")
    def ensure_valid_fp8_dtype_combination(self) -> "FP8MatmulConfig":
        if self.left_dtype == self.right_dtype == FP8DType.E5M2:
            raise ValueError("Received invalid combination of fp8 dtypes. Cannot multiply two e5m2 tensors.")
        return self


class FP8LinearConfig(BaseConfig):
    dtypes_forward: FP8MatmulConfig | None = Field(
        FP8MatmulConfig(left_dtype=FP8DType.E4M3, right_dtype=FP8DType.E4M3),
        description="Data types for the forward matmul (x, w).",
    )

    dtypes_grad_input: FP8MatmulConfig | None = Field(
        FP8MatmulConfig(left_dtype=FP8DType.E5M2, right_dtype=FP8DType.E4M3),
        description="Data types for the grad input matmul (grad_y, w).",
    )

    dtypes_grad_weight: FP8MatmulConfig | None = Field(
        FP8MatmulConfig(left_dtype=FP8DType.E4M3, right_dtype=FP8DType.E5M2),
        description="Data types for the grad weight matmul (x, grad_y).",
    )

    load_in_fp8: bool = Field(False, description="Load weights in FP8. This flag is only used internally.")

    @property
    def torch_dtypes_forward(self) -> tuple[torch.dtype, torch.dtype] | None:
        if self.dtypes_forward is None:
            return None
        else:
            return self.dtypes_forward.torch_dtypes

    @property
    def torch_dtypes_grad_input(self) -> tuple[torch.dtype, torch.dtype] | None:
        if self.dtypes_grad_input is None:
            return None
        else:
            return self.dtypes_grad_input.torch_dtypes

    @property
    def torch_dtypes_grad_weight(self) -> tuple[torch.dtype, torch.dtype] | None:
        if self.dtypes_grad_weight is None:
            return None
        else:
            return self.dtypes_grad_weight.torch_dtypes
