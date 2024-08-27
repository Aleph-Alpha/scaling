"""
The runner starts one launcher process on each node.
Its task is to make sure all relevant nodes are part of training.
On each individual node the launcher takes care of starting a process for each local rank (gpu).
"""

from .launch_config import LaunchConfig
from .runner import RunnerConfig, RunnerType, runner_main
from .runner_config import RunnerDockerConfig
