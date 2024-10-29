from examples.mlp_example.config import MLPConfig
from scaling.core import BaseContext
from scaling.core.topology import Topology


class MLPContext(BaseContext):
    config: MLPConfig

    def __init__(
        self,
        config: MLPConfig,
        topology: Topology,
    ):
        super().__init__(config=config, topology=topology)
