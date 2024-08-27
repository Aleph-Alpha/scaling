import torch

from scaling.core import BaseDatasetItem


class TextDatasetItem(BaseDatasetItem):
    token_ids: torch.Tensor

    def __init__(self, token_ids: torch.Tensor):
        super().__init__()
        self.token_ids = token_ids
