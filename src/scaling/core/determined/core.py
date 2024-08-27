from typing import Any

import determined as det


def init(*args: tuple, **kwargs: dict[str, Any]) -> Any:
    """
    Wrapper around det.core.init, with specific arguments
    we want to control set for any project using
    aleph-alpha-scaling.

    The manual mode (det.core._tensorboard_mode.TensorboardMode.MANUAL) allows the user
    full control of when tensorboard files are flushed. The default (AUTO) writes way too
    many files, which makes the filesystem brittle.
    """
    assert "tensorboard_mode" not in kwargs, "We don't allow any overwrite for tensorboard_mode"
    kwargs = dict(kwargs)
    kwargs["tensorboard_mode"] = det.core._tensorboard_mode.TensorboardMode.MANUAL  # type: ignore[assignment]
    return det.core.init(*args, **kwargs)  # type: ignore [arg-type]
