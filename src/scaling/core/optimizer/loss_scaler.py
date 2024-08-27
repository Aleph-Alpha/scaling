# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from Microsoft Corporation
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple, Optional

import torch

from .base import BaseOptimizerState
from .loss_scaler_config import LossScalerConfig
from .parameter_group import OptimizerParamGroup


class LossScalerState(BaseOptimizerState):
    current_scale: float
    current_hysteresis: float
    no_overflow_steps: int


def has_inf_or_nan(x: torch.Tensor) -> bool:
    try:
        # if x is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as x
        # (which is true for some recent version of pytorch).
        cpu_sum = float(x.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # cpu_sum = float(x.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if cpu_sum == float("inf") or cpu_sum == -float("inf") or cpu_sum != cpu_sum:
            return True
        return False


class LossScalerOutput(NamedTuple):
    overflow: Optional[bool]
    no_overflow_steps: Optional[int]
    current_loss_scale: Optional[float]


class LossScaler:
    def __init__(self, config: LossScalerConfig, parameter_groups: list[OptimizerParamGroup]) -> None:
        self.config = config

        self._current_scale = self.config.initial_scale
        self._current_hysteresis = self.config.hysteresis
        self._no_overflow_steps = 0

        # record parameters for overflow checking
        parameters = list()
        for parameter_group in parameter_groups:
            parameters.extend(parameter_group.parameters_for_overflow_check)
        self.parameters = parameters

    def some_overflow_in_local_param_grads(self) -> bool:
        for parameter in self.parameters:
            if parameter.grad is not None and has_inf_or_nan(parameter.grad.data):
                return True
        return False

    def some_overflow_in_global_param_grads(self) -> bool:
        local_overflow = self.some_overflow_in_local_param_grads()
        overflow_tensor = torch.cuda.ByteTensor([local_overflow])  # type: ignore[attr-defined]

        torch.distributed.all_reduce(
            overflow_tensor,
            op=torch.distributed.ReduceOp.MAX,
            group=torch.distributed.group.WORLD,  # TODO full data parallel copy?
        )

        overflow = overflow_tensor[0].item()
        return bool(overflow)

    def step(self) -> LossScalerOutput:
        if not self.config.enable:
            return LossScalerOutput(overflow=None, no_overflow_steps=None, current_loss_scale=None)

        # check overflow
        overflow = self.some_overflow_in_global_param_grads()

        # apply loss scaling dependent on overflow
        if overflow:
            if self.config.hysteresis == 1 or self._current_hysteresis == 1:
                self._current_scale = max(
                    self._current_scale / self.config.factor,
                    self.config.min_scale,
                )
            else:
                self._current_hysteresis -= 1
            self._no_overflow_steps = 0
        else:
            if self.config.consecutive_hysteresis:
                self._current_hysteresis = self.config.hysteresis
            if self._no_overflow_steps > 0 and (self._no_overflow_steps) % self.config.window == 0:
                if not self.config.consecutive_hysteresis:
                    self._current_hysteresis = self.config.hysteresis
                self._current_scale *= self.config.factor
            self._no_overflow_steps += 1

        return LossScalerOutput(
            overflow=overflow,
            no_overflow_steps=self._no_overflow_steps,
            current_loss_scale=self._current_scale,
        )

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if not self.config.enable:
            return loss

        loss = loss * self._current_scale
        return loss

    def state_dict(self) -> LossScalerState:
        return {
            "current_scale": self._current_scale,
            "current_hysteresis": self._current_hysteresis,
            "no_overflow_steps": self._no_overflow_steps,
        }

    def load_state_dict(self, state_dict: LossScalerState) -> None:
        self._current_scale = state_dict["current_scale"]
        self._current_hysteresis = state_dict["current_hysteresis"]
        self._no_overflow_steps = state_dict["no_overflow_steps"]
