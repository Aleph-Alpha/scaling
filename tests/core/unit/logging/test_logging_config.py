import os
from contextlib import nullcontext as does_not_raise
from typing import ContextManager

import pytest
from pydantic import ValidationError

from scaling.core.logging import LoggerConfig


@pytest.mark.cpu
def test_logging_config_wandb():
    os.environ["WANDB_API_KEY"] = "foo"
    config = LoggerConfig()
    config2 = LoggerConfig(wandb_api_key="bar")
    os.environ["WANDB_API_KEY"] = "baz"
    config3 = LoggerConfig(use_wandb=True)
    assert config.wandb_api_key == "foo"
    assert config2.wandb_api_key == "bar"
    assert config3.wandb_api_key == "baz"
    assert config3.use_wandb


@pytest.mark.cpu
def test_logging_config_wandb_no_api_key():
    os.environ["WANDB_API_KEY"] = ""
    with pytest.raises(ValidationError):
        LoggerConfig(use_wandb=True)

    del os.environ["WANDB_API_KEY"]
    with pytest.raises(ValidationError):
        LoggerConfig(use_wandb=True)

    assert LoggerConfig(use_wandb=True, wandb_api_key="foo")


@pytest.mark.cpu
@pytest.mark.parametrize(
    "use_wandb, api_key, is_env_variable_set, expectation",
    [
        pytest.param(
            True,
            "",
            False,
            pytest.raises(ValidationError, match="If 'use_wandb' is set to True a wandb api key needs to be provided."),
            id="use_wandb is true, but api key is empty string",
        ),
        pytest.param(
            True,
            None,
            False,
            pytest.raises(ValidationError, match="If 'use_wandb' is set to True a wandb api key needs to be provided."),
            id="use_wandb is true, but api key is None",
        ),
        pytest.param(
            True,
            "some_key",
            False,
            does_not_raise(),
            id="use_wandb is true, api key is provided",
        ),
        pytest.param(
            True,
            None,
            True,
            does_not_raise(),
            id="use_wandb is true, api key is not provided in config, but set as env variable",
        ),
        pytest.param(False, "", False, does_not_raise(), id="use_wandb is false and api key is empty string"),
        pytest.param(False, None, False, does_not_raise(), id="use_wandb is false and api key is None"),
        pytest.param(False, "some_key", False, does_not_raise(), id="use_wandb is false and api key is provided"),
    ],
)
def test_logger_config_validation_for_wandb_and_api_key(
    use_wandb: bool, api_key: str | None, is_env_variable_set: bool, expectation: ContextManager
) -> None:
    if is_env_variable_set:
        os.environ["WANDB_API_KEY"] = "some_key"

    with expectation:
        LoggerConfig(use_wandb=use_wandb, wandb_api_key=api_key)
