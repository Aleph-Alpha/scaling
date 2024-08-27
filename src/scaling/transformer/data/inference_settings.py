from typing import NamedTuple, Union

from pydantic import Field

from scaling.core import BaseConfig


class Control(BaseConfig):
    token_index: int = Field(description="token index to be controlled")

    factor: float = Field(description="control factor")


class InferenceSuppressionParameters(NamedTuple):
    contextual_control_threshold: float | None
    control_log_additive: bool
    controls: list[Control] | None


class InferenceSettings:
    use_cache: bool
    reset_cache: bool
    cache_index: int
    embedding_layers: list[int]
    input_image_locations: list[tuple[int, int, int]] | None

    inference_control_parameters: list[InferenceSuppressionParameters] | None

    control_log_additive_batch: Union[bool, list[bool]]

    def __init__(
        self,
        use_cache: bool,
        reset_cache: bool,
        cache_index: int,
        embedding_layers: list[int],
        input_image_locations: list[tuple[int, int, int]] | None = None,
        inference_control_parameters: list[InferenceSuppressionParameters] | None = None,
    ) -> None:
        self.use_cache = use_cache
        self.reset_cache = reset_cache
        self.cache_index = cache_index
        self.embedding_layers = embedding_layers
        self.input_image_locations = input_image_locations
        self.inference_control_parameters = inference_control_parameters

        if self.inference_control_parameters is None:
            self.control_log_additive_batch = True
        elif all([sp.control_log_additive for sp in self.inference_control_parameters]):
            self.control_log_additive_batch = True
        elif all([not sp.control_log_additive for sp in self.inference_control_parameters]):
            self.control_log_additive_batch = False
        else:
            self.control_log_additive_batch = [sp.control_log_additive for sp in self.inference_control_parameters]
