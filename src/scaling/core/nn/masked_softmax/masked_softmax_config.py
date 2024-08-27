from enum import Enum

from pydantic import Field

from scaling.core.config import BaseConfig


class MaskedSoftmaxKernel(Enum):
    TORCH = "torch"
    FLASH_ATTENTION = "flash_attention"


class MaskedSoftmaxConfig(BaseConfig):
    kernel: MaskedSoftmaxKernel = Field(
        MaskedSoftmaxKernel.TORCH,
        description="select an optimization kernel, "
        "if anything other than torch is selected "
        "the optional gpu_optimization dependencies need to be installed",
    )

    softmax_in_fp32: bool = Field(
        False,
        description="Cast tensor to fp32 before softmax for higher precision; "
        "this cannot be applied for fused kernels",
    )

    scale: float = Field(
        1.0,
        description="Scale with which scores are multiplied (not divided!) before softmax is applied. "
        "If scale is applied setting also softmax_in_fp32 is likely helpful.",
    )

    deterministic_flash_attn_bwd: bool = Field(
        False,
        description="Enables flash attention deterministic in the backward pass. It's slower and uses more memory.",
    )
