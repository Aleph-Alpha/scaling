import pytest
import yaml

from examples.mlp_example.config import MLPConfig
from scaling.transformer.context.config import TransformerConfig


@pytest.mark.cpu
@pytest.mark.parametrize("relative_config_path", ["examples/transformer_example/config.yml"])
def test_transformer_config_loads(path_to_root, relative_config_path):
    with open(path_to_root / relative_config_path, "r") as f:
        config = yaml.safe_load(f)

    result = TransformerConfig.from_dict(config)
    assert result.training_groups


@pytest.mark.cpu
@pytest.mark.parametrize("relative_config_path", ["examples/mlp_example/config.yml"])
def test_mlp_config_loads(path_to_root, relative_config_path):
    with open(path_to_root / relative_config_path, "r") as f:
        config = yaml.safe_load(f)

    result = MLPConfig.from_dict(config)
    assert result.architecture
