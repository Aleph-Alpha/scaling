# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from John Olafenwa
# Copyright (c) 2018 John Olafenwa
# SPDX-License-Identifier: MIT

from pydantic import Field

from scaling.core.config import BaseConfig


class LossScalerConfig(BaseConfig):
    """
    Loss scaling is designed to combat the problem of underflowing gradients encountered at long
    times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in overflowing gradients.

    The optimizer then skips the update step for this particular iteration/minibatch,
    and the loss scaler adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients detected,
    the loss scaler increases the loss scale once more.
    In this way the  loss scaler attempts to "ride the edge" of
    always using the highest loss scale possible without incurring overflow.
    """

    enable: bool = Field(
        False,
        description="",
    )

    initial_scale: float = Field(
        2.0**32,
        description="Initial loss scale",
    )

    window: int = Field(
        1000,
        description="",
    )

    hysteresis: float = Field(
        2,
        description="",
    )

    consecutive_hysteresis: bool = Field(
        False,
        description="",
    )

    min_scale: float = Field(
        1.0,
        description="",
    )

    factor: float = Field(
        2.0,
        description="",
    )
