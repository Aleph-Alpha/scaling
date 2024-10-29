import warnings
from enum import Enum
from pathlib import Path
from typing import Optional

import pytest
import torch
from pydantic import Field, ValidationError

from scaling.core import BaseConfig


@pytest.mark.short
def test_should_not_protect_model_namespace():
    warnings.filterwarnings("error")

    class TestConfig(BaseConfig):
        model_parallel_size: int = Field(
            0,
            description="",
        )

    _ = TestConfig()


@pytest.mark.short
def test_should_have_non_mutable_config():
    class TestConfig(BaseConfig):
        test_int: int = Field(
            0,
            description="",
        )

    config = TestConfig()
    with pytest.raises(ValidationError):
        config.test_int = 42


@pytest.mark.short
def test_should_error_with_unknown_field():
    class TestConfig(BaseConfig):
        test_int: int = Field(
            0,
            description="",
        )

    with pytest.raises(ValidationError):
        TestConfig(test_unknown=2)


@pytest.mark.short
def test_should_accept_optional_none_for_int():
    class TestConfig(BaseConfig):
        test_int: Optional[int] = Field(
            0,
            description="",
        )

    TestConfig(test_int=None)


@pytest.mark.short
def test_should_not_accept_none_for_int():
    class TestConfig(BaseConfig):
        test_int: int = Field(
            0,
            description="",
        )

    with pytest.raises(ValidationError):
        TestConfig(test_int=None)


@pytest.mark.short
def test_should_not_accept_str_for_int():
    class TestConfig(BaseConfig):
        test_int: int = Field(
            0,
            description="",
        )

    with pytest.raises(ValidationError):
        TestConfig(test_int="abc")


@pytest.mark.short
def test_should_json_serialize():
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    class Precision(Enum):
        FP16 = "float16"
        BFLOAT16 = "bfloat16"
        FP32 = "float32"

        @property
        def dtype(self) -> torch.dtype:
            if self == Precision.FP16:
                return torch.float16
            elif self == Precision.BFLOAT16:
                return torch.bfloat16
            elif self == Precision.FP32:
                return torch.float32
            else:
                raise NotImplementedError

    class TestConfig(BaseConfig):
        test_int: int = Field(
            0,
            description="",
        )

        test_path: Optional[Path] = Field(
            None,
            description="",
        )

        test_enum: Optional[Color] = Field(None, description="")

        test_precision: Optional[Precision] = Field(Precision.FP16, description="")

    config = TestConfig(test_int=42, test_path="/mnt/checkpoints", test_enum=Color.BLUE)

    config_serialized = config.as_str()
    config_deserialized = TestConfig.from_str(config_serialized)

    assert isinstance(config_deserialized.test_precision.dtype, torch.dtype)

    assert config_deserialized == config


@pytest.mark.short
def test_save_config_template(tmp_path: Path):
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    class Precision(Enum):
        FP16 = "float16"
        BFLOAT16 = "bfloat16"
        FP32 = "float32"

    class TestConfigNested(BaseConfig):
        test_int: int = Field(
            0,
            description="an int parameter",
        )

        test_path: Optional[Path] = Field(
            None,
            description="a path parameter",
        )

        test_enum: Optional[Color] = Field(None, description="")

    class TestConfig(BaseConfig):
        test_int_default: int = Field(
            0,
            description="an int parameter",
        )

        test_int: int = Field(
            description="an int parameter without default",
        )

        test_path: Optional[Path] = Field(
            None,
            description="a path parameter",
        )

        test_str: Optional[str] = Field(
            "default_str",
            description="a str parameter",
        )

        test_enum: Optional[Color] = Field(None, description="")

        test_precision: Precision = Field(Precision.FP16, description="")

        config: TestConfigNested = Field(TestConfigNested(), description="a nested config")

    out_file = tmp_path / "test_config.yaml"
    TestConfig.save_template(out_file)


@pytest.mark.short
def test_save_example_config_template(tmp_path: Path):
    from tests.core.minimal import MinimalConfig

    out_file = tmp_path / "test_config.yaml"
    MinimalConfig.save_template(out_file)
