from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import yaml

from scaling.core import (
    BaseLayerIO,
    LayerSpec,
    PipePartitionMethod,
)
from scaling.core.nn.parallel_module.inference_module import InferenceModule, RecorderSetting
from scaling.transformer.data.inference_settings import InferenceSettings
from scaling.transformer.model.layers.base import TransformerLayerIO

from ..context.config import TransformerArchitectureConfig
from ..data import TextDatasetBatch
from ..model.model import get_transformer_layer_specs
from ..tokenizer import Tokenizer
from .sample import sample_argmax


class CompletionOutput:
    def __init__(self, completion_text: str | None, completion_tokens: list[int], completion_logits: torch.Tensor):
        self.completion_text = completion_text
        self.completion_tokens = completion_tokens
        self.completion_logits = completion_logits


class TransformerInferenceModule(InferenceModule):
    def __init__(
        self,
        layer_specs: list[LayerSpec],
        devices: Sequence[int] = (0,),
        pipe_partition_method: PipePartitionMethod = PipePartitionMethod.UNIFORM,
        pipe_partition_overwrite: list[int] | None = None,
        tokenizer: Tokenizer | None = None,
    ):
        super().__init__(
            layer_specs=layer_specs,
            devices=devices,
            pipe_partition_method=pipe_partition_method,
            pipe_partition_overwrite=pipe_partition_overwrite,
        )
        self.tokenizer = tokenizer

    @staticmethod
    def _parse_config_file(config_file: Path) -> TransformerArchitectureConfig:
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        architecture_dict = config_dict["transformer_architecture"]
        return TransformerArchitectureConfig.from_dict(architecture_dict)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path,
        devices: Sequence[int] = (0,),
        pipe_partition_method: PipePartitionMethod = PipePartitionMethod.UNIFORM,
        pipe_partition_overwrite: list[int] | None = None,
        config_file: Path | None = None,
        vocab_file: Path | None = None,
    ) -> "TransformerInferenceModule":
        """Load a model from a checkpoint directory as inference module. By default assumes weights,
        model config and vocab file to be present in checkpoint directory.
        """
        if config_file is None:
            config_file = checkpoint_dir / "config.yml"
        if vocab_file is None:
            vocab_file = checkpoint_dir / "vocab.json"

        assert config_file.is_file(), "Config file not found"
        assert vocab_file.is_file(), "Vocab file not found"

        architecture_config = TransformerInferenceModule._parse_config_file(config_file)
        tokenizer = Tokenizer.from_file(str(vocab_file))
        layer_specs = get_transformer_layer_specs(architecture_config=architecture_config)
        model = cls(
            layer_specs=layer_specs,
            devices=devices,
            pipe_partition_method=pipe_partition_method,
            pipe_partition_overwrite=pipe_partition_overwrite,
            tokenizer=tokenizer,
        )
        model.load_checkpoint(checkpoint_dir)
        return model

    def forward(self, x: BaseLayerIO) -> TransformerLayerIO:  # parent class override for typing purposes
        out = super().forward(x)
        assert isinstance(out, TransformerLayerIO)
        return out

    def _pre_process_input(
        self,
        input_text: str | None = None,
        input_tokens: list[int] | None = None,
        process_for_cached_inference: bool = True,
    ) -> TextDatasetBatch:
        assert (input_text is None) ^ (input_tokens is None), "Either input_text or input_tokens needs to be provided"
        if input_text is not None:
            assert self.tokenizer is not None
            input_tokens = self.tokenizer.encode(input_text)

        input_token_ids = torch.tensor(input_tokens).unsqueeze(dim=0)
        inference_settings = InferenceSettings(
            use_cache=process_for_cached_inference, reset_cache=True, cache_index=0, embedding_layers=[-1]
        )

        return TextDatasetBatch(input_token_ids=input_token_ids, inference_settings=inference_settings)

    def _post_process_output(self, output: TransformerLayerIO) -> torch.Tensor:
        return output.activations.squeeze()

    def logits(self, input_text: str | None = None, input_tokens: list[int] | None = None) -> torch.Tensor:
        """Takes input text or tokens and returns the final model activations, i.e. typically logits as tensor."""
        input_batch = self._pre_process_input(input_text=input_text, input_tokens=input_tokens)
        output = self.forward(input_batch)
        return self._post_process_output(output).squeeze()

    def logits_with_hidden_state_recorder(
        self,
        input_text: str | None = None,
        input_tokens: list[int] | None = None,
        recorder_settings_per_layer: dict[int, RecorderSetting] | None = None,
    ) -> tuple[torch.Tensor, dict[int, dict[str, Any]]]:
        """Logits method that additionally records hidden states of the model.
        Recorder can be configured per layer.
        """
        input_batch = self._pre_process_input(input_text=input_text, input_tokens=input_tokens)
        output, recorder_result = super().forward_with_hidden_state_recorder(
            input_batch, recorder_settings_per_layer=recorder_settings_per_layer
        )
        assert isinstance(output, TransformerLayerIO)
        return self._post_process_output(output), recorder_result

    def _generate_prequel(
        self,
        use_cache: bool,
        stop_tokens: Sequence[int] | None = None,
        input_text: str | None = None,
        input_tokens: list[int] | None = None,
    ) -> tuple[TextDatasetBatch, Sequence[int], int, list, list, InferenceSettings]:
        if stop_tokens is None:
            assert self.tokenizer is not None, "If no tokenizer is provided, a stop token needs to be set manually"
            stop_tokens = [self.tokenizer.eos_token_id]
        current_input = self._pre_process_input(
            input_text=input_text, input_tokens=input_tokens, process_for_cached_inference=use_cache
        )
        assert current_input.input_token_ids is not None
        input_length = current_input.input_token_ids.shape[-1]
        completion_tokens: list[int] = []
        completion_logits: list[torch.Tensor] = []
        inference_settings = InferenceSettings(
            use_cache=use_cache, reset_cache=not use_cache, cache_index=0, embedding_layers=[-1]
        )
        return current_input, stop_tokens, input_length, completion_tokens, completion_logits, inference_settings

    def _generate_uncached(
        self,
        max_tokens: int,
        input_text: str | None = None,
        input_tokens: list[int] | None = None,
        sample_fn: Callable[[torch.Tensor], torch.Tensor] = sample_argmax,
        stop_tokens: Sequence[int] | None = None,
    ) -> CompletionOutput:
        current_input, stop_tokens, input_length, completion_tokens, _, inference_settings = self._generate_prequel(
            use_cache=False,
            stop_tokens=stop_tokens,
            input_text=input_text,
            input_tokens=input_tokens,
        )

        for _ in range(max_tokens):
            output = self.forward(current_input)
            next_token = sample_fn(output.activations)
            completion_tokens.append(int(next_token.item()))
            assert current_input.input_token_ids is not None
            next_token = next_token.to(current_input.input_token_ids.device)
            new_input_token_ids = torch.cat([current_input.input_token_ids, next_token.unsqueeze(dim=0)], dim=-1)
            current_input = TextDatasetBatch(input_token_ids=new_input_token_ids, inference_settings=inference_settings)
            if int(next_token.item()) in stop_tokens:
                break

        completion_text: str | None = None
        completion_logits = self._post_process_output(output)[input_length - 1 :]

        if self.tokenizer is not None:
            completion_text = self.tokenizer.decode(completion_tokens)

        return CompletionOutput(
            completion_text=completion_text, completion_tokens=completion_tokens, completion_logits=completion_logits
        )

    def _generate_cached(
        self,
        max_tokens: int,
        input_text: str | None = None,
        input_tokens: list[int] | None = None,
        sample_fn: Callable[[torch.Tensor], torch.Tensor] = sample_argmax,
        stop_tokens: Sequence[int] | None = None,
    ) -> CompletionOutput:
        (current_input, stop_tokens, input_length, completion_tokens, completion_logits, inference_settings) = (
            self._generate_prequel(
                use_cache=True,
                stop_tokens=stop_tokens,
                input_text=input_text,
                input_tokens=input_tokens,
            )
        )

        for k in range(max_tokens):
            output = self.forward(current_input)
            next_token = sample_fn(output.activations)
            completion_tokens.append(int(next_token.item()))
            completion_logits.append(output.activations[:, -1, :])
            next_position_ids = torch.tensor([input_length + k]).unsqueeze(dim=0)
            next_token_ids = next_token.unsqueeze(dim=0)
            current_input = TextDatasetBatch(
                input_token_ids=next_token_ids,
                inference_settings=inference_settings,
                position_ids=next_position_ids,
            )
            if int(next_token.item()) in stop_tokens:
                break

        completion_text: str | None = None
        if self.tokenizer is not None:
            completion_text = self.tokenizer.decode(completion_tokens)

        return CompletionOutput(
            completion_text=completion_text,
            completion_tokens=completion_tokens,
            completion_logits=torch.cat(completion_logits),
        )

    def generate(
        self,
        max_tokens: int,
        input_text: str | None = None,
        input_tokens: list[int] | None = None,
        sample_fn: Callable[[torch.Tensor], torch.Tensor] = sample_argmax,
        stop_tokens: Sequence[int] | None = None,
        use_cache: bool = True,
    ) -> CompletionOutput:
        """Takes input text or tokens and returns a completion object that contains completion text,
        tokens and logits."""
        if use_cache:
            return self._generate_cached(
                max_tokens=max_tokens,
                input_text=input_text,
                input_tokens=input_tokens,
                sample_fn=sample_fn,
                stop_tokens=stop_tokens,
            )
        else:
            return self._generate_uncached(
                max_tokens=max_tokens,
                input_text=input_text,
                input_tokens=input_tokens,
                sample_fn=sample_fn,
                stop_tokens=stop_tokens,
            )
