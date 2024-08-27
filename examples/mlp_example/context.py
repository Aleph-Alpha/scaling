from scaling.core import BaseContext
from scaling.core.topology import Topology

from examples.mlp_example.config import MLPConfig


class MLPContext(BaseContext):
    config: MLPConfig

    def __init__(
        self,
        config: MLPConfig,
        topology: Topology,
    ):
        super().__init__(config=config, topology=topology)
