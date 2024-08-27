from scaling.core import BaseContext, Topology

from .config import MinimalConfig


class MinimalContext(BaseContext):
    config: MinimalConfig

    def __init__(self, config: MinimalConfig, topology: Topology):
        super().__init__(config=config, topology=topology)
