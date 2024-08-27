from typing import Optional

import torch

from .masked_softmax_config import MaskedSoftmaxConfig, MaskedSoftmaxKernel


class MaskedSoftmaxTorch(torch.nn.Module):
    def __init__(self, config: MaskedSoftmaxConfig) -> None:
        super().__init__()
        self.config = config
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.config.softmax_in_fp32 and x.dtype != torch.float32:
            input_dtype = x.dtype
            x = x.float()
        else:
            input_dtype = None

        if self.config.scale != 1.0:
            x = x * self.config.scale

        x.masked_fill_(mask.to(x.device), -10000.0)
        probs = self.softmax(x)

        if self.config.softmax_in_fp32 and input_dtype is not None:
            probs = probs.to(input_dtype)

        return probs


class MaskedSoftmax(torch.nn.Module):
    def __init__(self, config: MaskedSoftmaxConfig) -> None:
        super().__init__()
        self.config = config
        self.kernel: Optional[torch.nn.Module]
        if self.config.kernel == MaskedSoftmaxKernel.TORCH:
            self.kernel = MaskedSoftmaxTorch(config=config)
        elif self.config.kernel == MaskedSoftmaxKernel.FLASH_ATTENTION:
            # Set kernel to None because we will use the Flash Attention impl in this case
            self.kernel = None
        else:
            raise NotImplementedError(f"kernel {self.config.kernel} not implemented")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert self.kernel is not None
        return self.kernel(x=x, mask=mask)
