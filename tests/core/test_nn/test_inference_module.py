from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from scaling.core import BaseLayerIO, LayerSpec, TiedLayerSpec, Topology
from scaling.core.nn.parallel_module.inference_module import InferenceModule, RecorderSetting
from tests.core.minimal.context import MinimalConfig
from tests.core.minimal.data import MinimalBatch
from tests.core.minimal.model.model import (
    MinimalEmbeddingInput,
    MinimalEmbeddingTied,
    MinimalLayerNorm,
    MinimalLinearColumnParallel,
    MinimalLinearRowParallel,
    MinimalParallelModule,
)

LAYER_SPECS = [
    LayerSpec(module_class=MinimalEmbeddingInput),
    LayerSpec(
        module_class=MinimalLinearColumnParallel,
    ),
    LayerSpec(
        module_class=MinimalLinearRowParallel,
    ),
    LayerSpec(
        module_class=MinimalLayerNorm,
    ),
]

TIED_LAYER_SPECS = [
    TiedLayerSpec(
        module_class=MinimalEmbeddingInput,
        key="embedding_tying",
        tied_weight_attributes=["embedding.weight"],
    ),
    LayerSpec(
        module_class=MinimalLinearColumnParallel,
    ),
    LayerSpec(
        module_class=MinimalLinearRowParallel,
    ),
    TiedLayerSpec(
        module_class=MinimalEmbeddingTied,
        key="embedding_tying",
        tied_weight_attributes=["embedding.weight"],
    ),
]


@pytest.mark.parametrize("devices", [(0,), (0, 1)])
@pytest.mark.parametrize("weight_tying", [True, False])
def test_inference_module_init_and_forward_pass(devices: tuple[int], weight_tying: bool):
    if weight_tying:
        inference_module = InferenceModule(layer_specs=TIED_LAYER_SPECS, devices=devices)
    else:
        inference_module = InferenceModule(layer_specs=LAYER_SPECS, devices=devices)
    assert len(inference_module._layers) == 4

    if devices == (0,):
        for layer in inference_module._layers:
            for p in layer.parameters():
                assert p.device == torch.device("cuda", 0)
    elif devices == (0, 1):
        for k, layer in enumerate(inference_module._layers):
            for p in layer.parameters():
                assert p.device == torch.device("cuda", int(k // 2))

    x = MinimalBatch(inputs=torch.tensor([0, 1], dtype=torch.long), targets=torch.tensor([0, 1], dtype=torch.long))

    out = inference_module(x)

    if devices == (0,):
        assert out.activations.device == torch.device("cuda", 0)
    else:
        assert out.activations.device == torch.device("cuda", 1)


@pytest.mark.parametrize("devices", [(0,), (0, 1)])
@pytest.mark.parametrize("requested_layers", [[], [0, 1, 2, 3], [1, 3], [-1]])
@pytest.mark.parametrize("include_modules,exclude_modules", [(None, None), ([""], None), (None, ["", "norm"])])
def test_inference_module_hidden_states_extraction(
    devices, requested_layers: list[int], include_modules: list[str] | None, exclude_modules: list[str] | None
):
    inference_module = InferenceModule(layer_specs=LAYER_SPECS, devices=devices)

    x = MinimalBatch(inputs=torch.tensor([0, 1], dtype=torch.long), targets=torch.tensor([0, 1], dtype=torch.long))

    recorder_settings_per_layer = {
        k: RecorderSetting(include_modules=include_modules, exclude_modules=exclude_modules) for k in requested_layers
    }

    _, hidden_states = inference_module.forward_with_hidden_state_recorder(
        x, recorder_settings_per_layer=recorder_settings_per_layer
    )

    assert set(hidden_states.keys()) == set(requested_layers)

    for layer_index, out_dict in hidden_states.items():
        if layer_index == 0:
            if include_modules is None and exclude_modules is None:
                assert len(out_dict) == 0
            elif include_modules is not None:
                assert set(out_dict.keys()) == set(include_modules)
            else:
                assert set(out_dict.keys()) == {"embedding"}
        if layer_index == 1:
            if include_modules is None and exclude_modules is None:
                assert len(out_dict) == 0
            elif include_modules is not None:
                assert set(out_dict.keys()) == set(include_modules)
            else:
                assert set(out_dict.keys()) == {"linear"}
        if layer_index == 2:
            if include_modules is None and exclude_modules is None:
                assert len(out_dict) == 0
            elif include_modules is not None:
                assert set(out_dict.keys()) == set(include_modules)
            else:
                assert set(out_dict.keys()) == {"linear"}
        if layer_index == 3:
            if include_modules is None and exclude_modules is None:
                assert len(out_dict) == 0
            elif include_modules is not None:
                assert set(out_dict.keys()) == set(include_modules)
            else:
                assert len(out_dict) == 0

        for k, v in out_dict.items():
            if k != "":
                assert isinstance(v, torch.Tensor)
            else:
                assert isinstance(v, BaseLayerIO)


@pytest.mark.parametrize("devices", [(0,), (0, 1)])
def test_inference_module_hidden_states_extraction_different_settings_on_different_layers(devices: tuple[int]):
    inference_module = InferenceModule(layer_specs=LAYER_SPECS, devices=devices)

    x = MinimalBatch(inputs=torch.tensor([0, 1], dtype=torch.long), targets=torch.tensor([0, 1], dtype=torch.long))

    recorder_settings_per_layer = {
        1: RecorderSetting(include_modules=[""]),
        2: RecorderSetting(include_modules=["linear"]),
    }

    _, hidden_states = inference_module.forward_with_hidden_state_recorder(
        x, recorder_settings_per_layer=recorder_settings_per_layer
    )

    assert set(hidden_states[1].keys()) == {""}
    assert set(hidden_states[2].keys()) == {"linear"}


@pytest.mark.parametrize("weight_tying", [True, False])
@pytest.mark.parametrize("devices", [(0,), (0, 1)])
def test_inference_module_consistency_with_parallel_module(tmp_path: Path, weight_tying: bool, devices: tuple[int]):
    # launch_config = LaunchConfig.from_launcher_args()
    config_dict: Dict[str, Any] = dict()
    config_dict["topology"] = dict()
    config_dict["topology"]["model_parallel_size"] = 1
    config_dict["topology"]["pipe_parallel_size"] = 1
    config_dict["topology"]["world_size"] = 1
    config_dict["topology"]["global_rank"] = 0
    config_dict["topology"]["local_slot"] = 0
    config_dict["topology"]["micro_batch_size"] = 2
    config_dict["topology"]["gradient_accumulation_steps"] = 1

    # initialize
    config: MinimalConfig = MinimalConfig.from_dict(config_dict)
    topology = Topology(config=config.topology)
    topology.initialize_device()

    if weight_tying:
        layer_specs = TIED_LAYER_SPECS
    else:
        layer_specs = LAYER_SPECS
    # initialize model and save a checkpoint
    parallel_module = MinimalParallelModule(layer_specs=layer_specs, topology=topology)
    (tmp_path / "checkpoint").mkdir(parents=True, exist_ok=True)
    parallel_module.save_checkpoint(tmp_path / "checkpoint", separate_file_for_parameters=None)

    x = MinimalBatch(
        inputs=torch.tensor([0, 1], dtype=torch.long).cuda(), targets=torch.tensor([0, 1], dtype=torch.long).cuda()
    )

    out_parallel = parallel_module(x).activations.cpu()

    inference_module = InferenceModule(layer_specs=layer_specs, devices=devices)

    inference_module.load_checkpoint(tmp_path / "checkpoint")

    out_inference = inference_module(x).activations.cpu()

    assert torch.allclose(out_parallel, out_inference)
