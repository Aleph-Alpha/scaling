from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Sequence

import torch

from ...data import BaseLayerIO
from ...topology import PipePartitionMethod
from .layer_spec import LayerSpec, TiedLayerSpec
from .partitioned_module import PipePartitionedModule


@dataclass
class RecorderSetting:
    include_modules: Optional[Sequence[str]] = ("",)
    exclude_modules: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        assert (
            self.include_modules is None or self.exclude_modules is None
        ), "Cannot specify both include_modules and exclude_modules"


class HiddenStateRecorder:
    def __init__(self, module: PipePartitionedModule, recorder_settings_per_layer: dict[int, RecorderSetting]):
        self._module = module
        self.recorder_settings_per_layer: dict[int, RecorderSetting] = recorder_settings_per_layer
        self.current_record: dict[int, dict[str, Any]] = {k: dict() for k in recorder_settings_per_layer.keys()}
        self._hooks: list[Any] = list()

    def __enter__(self) -> None:
        self.start_recording()

    def __exit__(self, *_args: tuple, **_kwargs: dict) -> None:
        self.stop_recording()

    def record_output(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        layer_index: int,
        name: str,
    ) -> None:
        self.current_record[layer_index][name] = output

    def start_recording(self) -> None:
        for layer_index, layer_settings in self.recorder_settings_per_layer.items():
            layer = self._module._layers[layer_index]
            if layer_settings.include_modules is not None:
                for name, submodule in layer.named_modules():
                    if name in layer_settings.include_modules:
                        self._hooks.append(
                            submodule.register_forward_hook(
                                partial(self.record_output, layer_index=layer_index, name=name)
                            )
                        )
            elif layer_settings.exclude_modules is not None:
                for name, submodule in layer.named_modules():
                    if name not in layer_settings.exclude_modules:
                        self._hooks.append(
                            submodule.register_forward_hook(
                                partial(self.record_output, layer_index=layer_index, name=name)
                            )
                        )

    def stop_recording(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def delete_records(self) -> None:
        del self.current_record
        self.current_record = {k: {} for k in self.recorder_settings_per_layer.keys()}


class InferenceModule(PipePartitionedModule):
    def __init__(
        self,
        layer_specs: list[LayerSpec],
        devices: Sequence[int] = (0,),
        pipe_partition_method: PipePartitionMethod = PipePartitionMethod.UNIFORM,
        pipe_partition_overwrite: list[int] | None = None,
    ):
        self.devices: list[torch.device]
        layer_specs_converted = []
        for layer_spec in layer_specs:
            if isinstance(layer_spec, TiedLayerSpec):  # we convert tied layer specs to normal layer specs for inference
                layer_specs_converted.append(LayerSpec(module_class=layer_spec.module_class, **layer_spec.kwargs))
            else:
                layer_specs_converted.append(layer_spec)
        super().__init__(
            layer_specs=layer_specs_converted,
            devices=devices,
            pipe_partition_method=pipe_partition_method,
            pipe_partition_overwrite=pipe_partition_overwrite,
        )
        self.eval()

    @torch.no_grad()
    def forward(self, x: BaseLayerIO) -> BaseLayerIO:
        for device, pipe_parallel_coordinate in zip(self.devices, self._pipe_partition_coordinates):
            for local_index, layer in enumerate(
                self._layers[pipe_parallel_coordinate.start : pipe_parallel_coordinate.end]
            ):
                if local_index == 0:
                    x.to_(device)
                x = layer(x)
        return x

    @torch.no_grad()
    def forward_with_hidden_state_recorder(
        self, x: BaseLayerIO, recorder_settings_per_layer: Optional[dict[int, RecorderSetting]] = None
    ) -> tuple[BaseLayerIO, dict[int, dict[str, Any]]]:
        if recorder_settings_per_layer is None:
            recorder_settings_per_layer = dict()

        hidden_state_recorder = HiddenStateRecorder(self, recorder_settings_per_layer=recorder_settings_per_layer)
        with hidden_state_recorder:
            x = self.forward(x)

        return x, hidden_state_recorder.current_record
