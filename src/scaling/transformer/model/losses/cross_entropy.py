from typing import Dict, Tuple

import torch

from scaling.core.nn.scale import scale_bwd, scale_fwd
from scaling.core.nn.umup import UMuParametrization
from scaling.transformer.context.config import CrossEntropyLossFunctionConfig, UMuPConfig
from scaling.transformer.data.text_dataset_batch import TextDatasetBatch
from scaling.transformer.model.layers.base import TransformerLayerIO


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, loss_config: CrossEntropyLossFunctionConfig, umup_config: UMuPConfig):
        super().__init__()
        self._use_umup = umup_config.enable
        self.umup_loss_mult = umup_config.loss_mult

    def _standard_forward(
        self,
        output_activations_flatten: torch.Tensor,
        target_token_ids_flatten: torch.Tensor,
        loss_weights_flatten: torch.Tensor,
    ) -> torch.Tensor:
        losses = torch.nn.functional.cross_entropy(
            output_activations_flatten,
            target=target_token_ids_flatten,
            reduction="none",
        )

        loss = torch.sum(losses * loss_weights_flatten) / loss_weights_flatten.sum()

        return loss

    def _umup_forward(
        self,
        output_activations_flatten: torch.Tensor,
        target_token_ids_flatten: torch.Tensor,
        loss_weights_flatten: torch.Tensor,
    ) -> torch.Tensor:
        vocab_size = output_activations_flatten.shape[-1]

        fwd_scale, bwd_scale = UMuParametrization.get_umup_cross_entropy_scales(self.umup_loss_mult, vocab_size)

        output_activations_flatten = scale_fwd(output_activations_flatten, fwd_scale)
        output_activations_flatten = scale_bwd(output_activations_flatten, bwd_scale)

        losses = torch.nn.functional.cross_entropy(
            output_activations_flatten,
            target=target_token_ids_flatten,
            reduction="none",
        )

        loss = torch.sum(losses * loss_weights_flatten)
        loss = scale_fwd(loss, 1 / loss_weights_flatten.sum())

        return loss

    def forward(
        self, output: TransformerLayerIO, batch: TextDatasetBatch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates cross entropy loss for every probability distribution
        in the output activations and the target token ids.
        Returns the average loss for this batch as well as the accuracy.
        """
        assert (
            batch.target_token_ids is not None
        ), "target_token_ids not set in batch; you may want to revisit the implementation of Batch.only_targets()"
        assert (
            batch.loss_weights is not None
        ), "loss_weights not set in batch; you may want to revisit the implementation of Batch.only_targets()"

        loss_weights_flatten = batch.loss_weights.float().flatten()
        output_activations_flatten = output.activations.float().reshape(-1, output.activations.shape[-1])
        target_token_ids_flatten = batch.target_token_ids.flatten()

        if self._use_umup:
            loss = self._umup_forward(output_activations_flatten, target_token_ids_flatten, loss_weights_flatten)
        else:
            loss = self._standard_forward(output_activations_flatten, target_token_ids_flatten, loss_weights_flatten)

        loss_mask_flatten: torch.Tensor = loss_weights_flatten > 0
        loss_mask_flatten = loss_mask_flatten.float()

        accuracy = (output_activations_flatten.argmax(-1) == target_token_ids_flatten).float()
        accuracy = torch.sum(accuracy * loss_mask_flatten) / loss_mask_flatten.sum()

        return loss, {"accuracy": accuracy}
