# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from Snap Inc.
# Copyright (c) 2021 Snap Inc.
# SPDX-License-Identifier: MIT

import collections
from typing import Any, Callable, TypeVar

import torch

from scaling.core import (
    BaseLayer,
    Topology,
    VocabParallelEmbedding,
)
from scaling.transformer.data.text_dataset_batch import TextDatasetBatch

from ...context.config import (
    TransformerArchitectureConfig,
)
from ..image_encoder import ImageEncoder
from .base import TransformerLayerIO

TextDatasetBatchGeneric = TypeVar("TextDatasetBatchGeneric", bound=TextDatasetBatch)


class BaseEmbeddingInput(
    BaseLayer[TextDatasetBatchGeneric, TransformerLayerIO, TransformerLayerIO],
):
    def __init__(
        self,
        architecture_config: TransformerArchitectureConfig,
        topology: Topology | None = None,
        init_method: Callable[[torch.Tensor], torch.Tensor] = torch.nn.init.xavier_normal_,
    ):
        super().__init__()
        self.architecture_config = architecture_config
        self.topology = topology

        self.embedding = VocabParallelEmbedding(
            num_embeddings=self.architecture_config.vocab_size,
            embedding_dim=architecture_config.hidden_size,
            topology=self.topology,
            dtype=architecture_config.precision.dtype,
            init_method=init_method,
            finetunable_token_ids=architecture_config.finetunable_token_ids,
        )
        self.dropout = torch.nn.Dropout(p=architecture_config.dropout_embedding)

        # Image encoder
        if architecture_config.image_encoder:
            self.image_encoder = ImageEncoder(
                out_features=architecture_config.hidden_size,
                dropout_p=architecture_config.dropout_image_encoder,
                layernorm_config=architecture_config.layernorm,
                image_encoder="ClipRN50x16",
                dtype=architecture_config.precision.dtype,
                device=torch.device("cuda") if topology is None else topology.device,
            )

        # Softprompts
        self.softprompt_name: str | None = None
        if architecture_config.softprompt_config is not None:
            self.softprompt_name = architecture_config.softprompt_config.name
            setattr(
                self,
                f"softprompt_{self.softprompt_name}",
                torch.nn.Parameter(
                    torch.zeros(
                        (
                            architecture_config.softprompt_config.n_tokens,
                            architecture_config.hidden_size,
                        ),
                        dtype=architecture_config.precision.dtype,
                        device=torch.device("cuda") if topology is None else topology.device,
                    )
                ),
            )
            init_method(getattr(self, f"softprompt_{self.softprompt_name}"))

        self.cache: dict[int, torch.Tensor | None] = dict()

    def forward(self, x: TextDatasetBatchGeneric) -> TransformerLayerIO:
        # get cache parameters the cache is used for conceptual suppression
        if x.inference_settings is None:
            use_cache = False
            reset_cache = False
            cache_index = 0
        else:
            use_cache = x.inference_settings.use_cache
            reset_cache = x.inference_settings.reset_cache
            cache_index = x.inference_settings.cache_index

        # reset the cache straight away to not keep memory in use for the following batches without suppression
        if reset_cache:
            self.cache[cache_index] = None

        assert x.input_token_ids is not None

        activations = self.embedding(x.input_token_ids)

        if self.topology is not None:
            with self.topology.model_parallel_constant_rng():
                activations = self.dropout(activations)
        else:
            activations = self.dropout(activations)

        # magma training
        if x.input_images is not None:
            if self.topology is not None:
                with self.topology.model_parallel_constant_rng():
                    img_embeddings = self.image_encoder(x.input_images)
            else:
                img_embeddings = self.image_encoder(x.input_images)

            if x.input_image_locations is not None:
                # finetuning
                assert len(x.input_image_locations) == img_embeddings.shape[0]

                for image_embedding, (batch_index, start_pos, end_pos) in zip(img_embeddings, x.input_image_locations):
                    activations[batch_index, start_pos:end_pos] = image_embedding
            elif self.training:
                # train case, just cat at the beginning
                # cat is used instead of a replace to allow for gradients
                assert x.inference_settings is None
                activations = torch.cat(
                    [img_embeddings, activations[:, img_embeddings.shape[1] :, :]],
                    dim=1,
                ).contiguous()
            else:
                # inference case
                # There can be multiple images per batch item
                # Therefore cat is not used and embeddings are just replaced
                # The image locations have been filled with the eos as dummy token
                assert x.inference_settings is not None
                assert x.inference_settings.input_image_locations is not None
                assert len(x.inference_settings.input_image_locations) == img_embeddings.shape[0]

                for image_embedding, (batch_index, start_pos, end_pos) in zip(
                    img_embeddings, x.inference_settings.input_image_locations
                ):
                    activations[batch_index, start_pos:end_pos] = image_embedding

        # softprompt training
        if self.softprompt_name is not None:
            softprompt = getattr(self, f"softprompt_{self.softprompt_name}")
            # we expect placeholders in the embedding
            # cat is used to allow for gradient flow
            activations = torch.cat(
                [
                    softprompt.unsqueeze(0).repeat(activations.shape[0], 1, 1),
                    activations[:, softprompt.shape[0] :, :],
                ],
                dim=1,
            ).contiguous()

        if x.inference_settings is not None and 0 in x.inference_settings.embedding_layers:
            embeddings = x.embeddings
            assert embeddings is not None
            embeddings[0, :, :, :] = activations
        else:
            embeddings = x.embeddings

        # attention manipulation
        loss_weights = x.loss_weights
        attention_scores_manipulation: torch.Tensor | None = None
        if x.inference_settings is not None and x.inference_settings.inference_control_parameters is not None:
            assert (
                len(x.inference_settings.inference_control_parameters) == activations.shape[0]
            ), "number of inference_control_parameters does not match batch size"

            # initialize the attention control matrix
            attention_scores_manipulation = torch.zeros(
                activations.size(0),
                1,
                activations.size(1),
                activations.size(1),
                device=activations.device,
                dtype=activations.dtype,
            )

            # set the identity default score manipulation for the non log-additive case
            for batch_index, inference_control_parameters in enumerate(
                x.inference_settings.inference_control_parameters
            ):
                if not inference_control_parameters.control_log_additive:
                    attention_scores_manipulation[batch_index, :, :, :] = 1.0

            # initialize similarity matrix if needed
            sim_matrix: torch.Tensor | None = None
            if any(
                [p.contextual_control_threshold is not None for p in x.inference_settings.inference_control_parameters]
            ):
                activations_for_similarity = activations
                if use_cache:
                    if not reset_cache:
                        cached_activations = self.cache[cache_index]
                        assert cached_activations is not None
                        activations_for_similarity = torch.cat([cached_activations, activations_for_similarity], dim=1)

                    self.cache[cache_index] = activations_for_similarity

                sim_matrix = self.get_embedding_similarity_matrix(embeddings=activations_for_similarity)

            # add control token_indices and factors if conceptual control is defined
            for batch_index, inference_control_parameters in enumerate(
                x.inference_settings.inference_control_parameters
            ):
                # skip batch item if not applicable
                if inference_control_parameters is None:
                    continue

                if inference_control_parameters.controls is None or all(
                    [c.token_index == -1 for c in inference_control_parameters.controls]
                ):
                    continue

                # initialize aggregated collector
                control_token_index_to_factor = collections.defaultdict(lambda: 0.0)

                # add to collector
                for control in inference_control_parameters.controls:
                    # skip for token index -1
                    if control.token_index < 0:
                        continue

                    # add to collector
                    control_token_index_to_factor[control.token_index] = control.factor

                    # get more indices to suppress for conceptual suppression
                    if inference_control_parameters.contextual_control_threshold is not None:
                        assert sim_matrix is not None
                        similarity_scores = sim_matrix[batch_index][control.token_index]
                        assert (
                            similarity_scores.ndim == 1
                        ), f"Expected similarity_scores.ndim to be 1 but got: {similarity_scores.ndim}"

                        additional_indices = (
                            (similarity_scores >= inference_control_parameters.contextual_control_threshold)
                            .nonzero()
                            .view(-1)
                            .tolist()
                        )
                        for additional_index in additional_indices:
                            # remove the baseline index, which will always have a cosine similarity of 1
                            if additional_index == control.token_index:
                                continue

                            # derive factor for control
                            additional_factor = self.get_control_factor_from_cosine_similarity(
                                control_factor=control.factor,
                                cosine_similarity=similarity_scores[additional_index].item(),
                            )

                            # we aggregate with max in case of multiple input suppression token indices
                            control_token_index_to_factor[additional_index] = min(
                                additional_factor,
                                control_token_index_to_factor[additional_index],
                            )

                # add to manipulation matrix and loss weights for pooling for embeddings
                for (
                    control_token_index,
                    control_factor,
                ) in control_token_index_to_factor.items():
                    if reset_cache:  # only set weights on first forward pass
                        assert loss_weights is not None
                        loss_weights[batch_index, control_token_index] = (
                            loss_weights[batch_index, control_token_index] * control_factor
                        )
                    if inference_control_parameters.control_log_additive:
                        attention_scores_manipulation[batch_index, :, :, control_token_index] = (
                            -10000 if control_factor == 0.0 else torch.log(torch.tensor(control_factor))
                        )
                    else:
                        attention_scores_manipulation[batch_index, :, :, control_token_index] = control_factor

        return TransformerLayerIO(
            activations=activations,
            position_ids=x.position_ids,  # type: ignore[arg-type]
            cumulative_seq_lengths=x.cumulative_seq_lengths,
            cumulative_seq_lengths_padded=x.cumulative_seq_lengths_padded,  # type: ignore[arg-type]
            attention_scores_manipulation=attention_scores_manipulation,
            loss_weights=loss_weights,
            inference_settings=x.inference_settings,
            embeddings=x.embeddings,
        )

    def get_control_factor_from_cosine_similarity(self, control_factor: float, cosine_similarity: float) -> float:
        ## the formula we use for calculating the control factor for a conceptually similar token
        ## given a suppression factor and the cossim of the similar token w.r.t the input token
        ## if control_factor == x and cosine_similarity == 1.0, then it returns x
        ## if control_factor == x and cosine_similarity == 0.0, then it returns 1.0
        ## if control_factor == x and cosine_similarity == 0.5, then it returns (1+x)/2
        if 0 <= cosine_similarity <= 1.0:
            x = (1 - control_factor) * (1 - cosine_similarity) + control_factor
        else:
            x = 1.0
        return x

    def get_embedding_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Returns a similarity matrix of shape (batch_size, seq, seq) containing the cosine
        similarities of each token w.r.t every other token.
        """
        assert embeddings.ndim == 3, f"Expected embeddings_batch to have 3 dimensions but got {embeddings.ndim}"
        batch_size, _ = embeddings.shape[0], embeddings.shape[1]
        cossim_matrices = torch.zeros(
            embeddings.shape[0],  # batch
            embeddings.shape[1],  # seq
            embeddings.shape[1],  # seq
        )

        with torch.no_grad():
            for batch_idx in range(batch_size):
                source_embeddings = embeddings[batch_idx].float()
                sim_matrix = self.get_similarity_matrix(a=source_embeddings, b=source_embeddings)
                cossim_matrices[batch_idx] = sim_matrix
        return cossim_matrices.clip(-1, 1)

    def get_similarity_matrix(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        finds the cosine similarity matrix between each item of a w.r.t each item of b
        a and b are expected to be 2-dimensional (seq, hidden_dim)
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    @staticmethod
    def input_to_tuple(
        input: TextDatasetBatchGeneric,
    ) -> tuple[Any, ...]:
        """
        convert layer input to a tuple with tensors as values for pipe communication and activation checkpointing
        this may include a split to model parallel
        tuple_to_input will be called on the tuple, here you might need to merge split tensors again
        we are using a tuple because torch requires tuples for activation checkpointing
        """
        return input.as_tuple()

    @staticmethod
    def tuple_to_input(d: tuple[Any, ...]) -> TextDatasetBatchGeneric:
        return TextDatasetBatch.from_tuple(d)  # type: ignore[return-value]

    @staticmethod
    def output_to_tuple(
        output: TransformerLayerIO,
    ) -> tuple[Any, ...]:
        """
        convert layer output to a tuple with tensors as values for pipe communication and activation checkpointing
        this may include a split to model parallel
        tuple_to_input will be called on the tuple, here you might need to merge split tensors again
        we are using a tuple because torch requires tuples for activation checkpointing
        """
        return output.as_tuple()

    @staticmethod
    def tuple_to_last_stage_activation(d: tuple[Any, ...]) -> TransformerLayerIO:
        """
        convert a tuple with tensors as values for pipe communication to the last layer's output class
        you might need to merge split tensors again
        """
        return TransformerLayerIO.from_tuple(d)


class EmbeddingInput(
    BaseEmbeddingInput[TextDatasetBatch],
):
    pass
