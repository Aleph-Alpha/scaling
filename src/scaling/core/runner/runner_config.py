from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import Field

from ..config import BaseConfig


class RunnerType(Enum):
    PDSH = "pdsh"
    PDSH_DOCKER = "pdsh_docker"


class RunnerDockerConfig(BaseConfig):
    docker_container: Optional[str] = Field(None, description="Name of the docker container to be started")

    docker_sudo: bool = Field(False, description="Run docker command with sudo")

    docker_mounts: Optional[List[Tuple[str, str]]] = Field(
        None,
        description="List of directories to be mounted in the docker "
        "from first string arg to second str arg in the inner list",
    )


class RunnerConfig(BaseConfig, populate_by_name=True):
    runner_type: RunnerType = Field(
        RunnerType.PDSH,
        description="Type of the runner to be invoked.",
    )

    hostsfile: Optional[Path] = Field(
        None,
        description="Hostsfile path (in MPI style) that defines the resource pool available to the job "
        "(e.g., worker-0 slots=4)",
        alias="hostfile",  # Aliases is necessary for supporting legacy configs
    )

    hosts: Optional[List[str]] = Field(
        None,
        description="List of hosts alternative to hostsfile (e.g., worker-0 slots=4)",
    )

    master_port: int = Field(
        29500,
        description="(optional) Port used by PyTorch distributed for communication during training.",
    )

    master_addr: Optional[str] = Field(
        None,
        description="optional) IP address of node 0, will be inferred via 'hostname -I' if not specified.",
    )

    script: Optional[Path] = Field(
        None,
        description="User script to launch",
    )

    default_gpu_count: int = Field(
        8,
        description="Number of GPUs per node, if used if not defined in hosts' slots",
    )

    docker_config: RunnerDockerConfig = Field(
        RunnerDockerConfig(),
        description="docker configuration in case using a docker runner type",
    )

    use_determined: bool = Field(
        False,
        description="Flag indicating if determined is to be used for metric and checkpoint tracking.",
    )
