import os
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from dateutil.parser import parse  # type: ignore
from pydantic import Field, model_validator

from scaling.core.config import BaseConfig


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TensorboardRanksConfig(BaseConfig):
    pipe_parallel_rank: int | None = None
    data_parallel_rank: int | None = None
    model_parallel_rank: int | None = None


class LoggerConfig(BaseConfig):
    log_level: LogLevel = Field(LogLevel.INFO, description="")

    log_dir: Path | None = Field(None, description="")

    metrics_ranks: list[int] | None = Field(
        None,
        description="define the global ranks of process to write metrics. "
        "If the list is omitted or None only rank 0 will write metrics.",
    )

    use_wandb: bool = Field(False, description="")
    wandb_ranks: list[int] | None = Field(
        None,
        description="define the global ranks of process to write to wandb. "
        "If the list is omitted or None only rank 0 will write to wandb.",
    )

    wandb_host: str = Field("https://api.wandb.ai", description="url of the wandb host")
    wandb_team: str = Field("aleph-alpha", description="Team name for Weights and Biases.")
    wandb_project: str = Field("aleph-alpha-scaling", description="wandb project name")
    wandb_group: str = Field("debug", description="wandb project name")
    wandb_api_key: str | None = Field(
        default=None,
        description="set wandb api key in order not to perform a wandb login first",
    )

    use_tensorboard: bool = Field(False, description="")

    tensorboard_ranks: list[int] | TensorboardRanksConfig | None = Field(
        None,
        description="define the global ranks of process to write to tensorboard. "
        "If a list, its entries specify the tensorboard ranks. "
        "If a TensorboardRanksConfig, topology.get_global_rank_group() will be used "
        "to determine the actual ranks. If None only rank 0 will write to tensorboard.",
    )

    determined_metrics_ranks: list[int] | None = Field(
        None,
        description="define the global ranks of process to write metrics to determined. "
        "If the list is omitted or None only rank 0 will write to determined.",
    )

    @model_validator(mode="before")
    def check_tensorboard_requires_log_dir(cls, values: dict[str, Any]) -> dict[str, Any]:
        use_tensorboard = values.get("use_tensorboard", cls.model_fields["use_tensorboard"].default)
        log_dir = values.get("log_dir", cls.model_fields["log_dir"].default)
        if use_tensorboard and log_dir is None:
            warnings.warn("Setting 'use_tensorboard' to False because no 'log_dir' is defined.")
            values["use_tensorboard"] = False
        return values

    @model_validator(mode="before")
    def add_dates_to_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        log_dir = values.get("log_dir")
        if log_dir is not None:
            log_dir = Path(log_dir)
            if not is_date(log_dir.name):
                log_dir = log_dir / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                values["log_dir"] = log_dir

        wandb_group = values.get("wandb_group")
        if wandb_group is not None:
            if not is_date(wandb_group.split("-")[-1]):
                wandb_group += "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                values["wandb_group"] = wandb_group

        return values

    @model_validator(mode="before")
    def check_if_api_key_is_provided_when_using_wandb(cls, values: dict[str, Any]) -> dict[str, Any]:
        wandb_api_key = values.get("wandb_api_key", cls.model_fields["wandb_api_key"].default)
        use_wandb = values.get("use_wandb", cls.model_fields["use_wandb"].default)
        if not wandb_api_key:
            wandb_api_key = get_wandb_api_from_env()
        if use_wandb and not wandb_api_key:
            raise ValueError("If 'use_wandb' is set to True a wandb api key needs to be provided.")
        values["wandb_api_key"] = wandb_api_key
        return values

    def is_rank_in_wandb_ranks(self, rank: int | None) -> bool:
        return check_if_in_rank(rank, ranks=self.wandb_ranks)

    def is_rank_in_metrics_ranks(self, rank: int | None) -> bool:
        return check_if_in_rank(rank, ranks=self.metrics_ranks)

    def is_rank_in_determined_metrics_ranks(self, rank: int | None) -> bool:
        return check_if_in_rank(rank, ranks=self.determined_metrics_ranks)


def check_if_in_rank(target_rank: int | None, ranks: list[int] | None) -> bool:
    if target_rank is None:
        return False
    if ranks is not None:
        return target_rank in ranks
    return target_rank == 0


def get_wandb_api_from_env() -> str | None:
    """Get Weights and Biases API key from WANDB_API_KEY env variable if it exists. Otherwise, return None"""
    return os.getenv("WANDB_API_KEY")


def is_date(string: str, fuzzy: bool = False) -> bool:
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
