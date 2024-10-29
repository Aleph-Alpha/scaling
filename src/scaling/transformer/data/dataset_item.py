import torch

from scaling.core.data import BaseDatasetItem


class TextImageDatasetItem(BaseDatasetItem):
    input_token_ids: torch.Tensor
    target_token_ids: torch.Tensor
    cumulative_seq_lengths: torch.Tensor
    position_ids: torch.Tensor
    loss_weights: torch.Tensor
    input_images: list[torch.Tensor] | None
    input_image_locations: list[tuple[int, int]] | None

    def __init__(
        self,
        input_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        cumulative_seq_lengths: torch.Tensor,
        position_ids: torch.Tensor,
        loss_weights: torch.Tensor,
        input_images: list[torch.Tensor] | None = None,
        input_image_locations: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.input_token_ids = input_token_ids
        self.target_token_ids = target_token_ids
        self.cumulative_seq_lengths = cumulative_seq_lengths
        self.position_ids = position_ids
        self.loss_weights = loss_weights
        self.input_images = input_images
        self.input_image_locations = input_image_locations
