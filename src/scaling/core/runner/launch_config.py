import base64
import json
import os
from argparse import REMAINDER, ArgumentParser
from pathlib import Path
from typing import Any, Optional

from pydantic import Field

from ..config import BaseConfig


class LaunchConfig(BaseConfig):
    master_port: int = Field(
        description="Port used by PyTorch distributed for communication during training.",
    )

    master_addr: str = Field(
        description="IP address of of master node.",
    )

    world_size: int = Field(
        description="Total world size of job",
    )

    global_rank: int = Field(
        description="Global rank of the current process",
    )

    local_slot: int = Field(
        description="GPU id of the current process",
    )

    payload: Optional[dict[Any, Any]] = Field(
        None,
        description="GPU id of the current process",
    )

    @classmethod
    def from_launcher_args(cls) -> "LaunchConfig":
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        world_size = os.environ["WORLD_SIZE"]
        global_rank = os.environ["RANK"]
        local_slot = os.environ["LOCAL_SLOT"]

        parser = ArgumentParser(description="process launch")

        # Optional arguments for the launch helper
        parser.add_argument(
            "--payload",
            type=str,
            default=None,
            help="base64 encoded payload",
        )
        parser.add_argument("remaining_args", nargs=REMAINDER)

        args = parser.parse_args()

        if args.payload is None:
            payload = None
        else:
            payload = decode_base64(args.payload)

        return cls(
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            global_rank=global_rank,
            local_slot=local_slot,
            payload=payload,
        )

    def overwrite_config_dict_with_launcher_args(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        config_dict["topology"]["world_size"] = self.world_size
        config_dict["topology"]["global_rank"] = self.global_rank
        config_dict["topology"]["local_slot"] = self.local_slot
        if (
            config_dict.get("profiler", dict()).get("profiler_output") is None
            and config_dict.get("logger", dict()).get("log_dir") is not None
        ):
            config_dict["profiler"]["profiler_output"] = Path(config_dict["logger"]["log_dir"]) / "profile.json"
        return config_dict


def decode_base64(s: str) -> dict[Any, Any]:
    decoded = base64.urlsafe_b64decode(s)
    result = json.loads(decoded)
    return result
