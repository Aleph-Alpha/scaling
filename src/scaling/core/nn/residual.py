from typing import Any

import torch

from .scale import scale_bwd, scale_fwd
from .umup import UMuParametrization


class _NormedResidualOp(torch.nn.Module):
    def __init__(
        self,
        umup_residual_mults: list[float] | None = None,
        umup_residual_layer_index: int | None = None,
        umup_pre_norm: bool | None = None,
    ) -> None:
        """This is the base class for pre/post norm residual operations compatible with u-mup.

        residual mults: list of residual multipliers in the model, i.e. residual_mults[k] is the
        multiplier for the k-th residual block

        residual_layer_index: position of this residual op, i.e. should be k for the k-th residual block.

        pre_norm: determines whether the scales will be calculated for pre or post norm architecture.
        """
        super().__init__()
        self.umup_residual_mults = umup_residual_mults
        self.umup_residual_layer_index = umup_residual_layer_index
        self.umup_pre_norm = umup_pre_norm

        # umup parameters
        self._use_umup = False
        self.residual_scale: float
        self.skip_scale: float

    def umup_setup(self, depth: int, **kwargs: Any) -> None:
        assert self.umup_residual_mults is not None
        assert self.umup_residual_layer_index is not None
        assert self.umup_pre_norm is not None

        (
            self.residual_scale,
            self.skip_scale,
        ) = UMuParametrization.get_umup_residual_scales(
            self.umup_residual_mults, self.umup_residual_layer_index, self.umup_pre_norm, depth
        )
        self._use_umup = True


class NormedResidualSplit(_NormedResidualOp):
    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_umup:
            hidden_state_residual = scale_bwd(hidden_state, self.residual_scale)
            hidden_state_skip = scale_bwd(hidden_state, self.skip_scale)
            return hidden_state_residual, hidden_state_skip
        else:
            return hidden_state, hidden_state


class NormedResidualAdd(_NormedResidualOp):
    def forward(self, hidden_state_residual: torch.Tensor, hidden_state_skip: torch.Tensor) -> torch.Tensor:
        if self._use_umup:
            hidden_state_residual = scale_fwd(hidden_state_residual, self.residual_scale)
            hidden_state_skip = scale_fwd(hidden_state_skip, self.skip_scale)

        return hidden_state_residual + hidden_state_skip
