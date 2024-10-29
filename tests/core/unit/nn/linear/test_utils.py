from unittest.mock import MagicMock, patch

import pytest
import torch

from scaling.core.nn.linear import utils


def mock_topology(device: torch.device) -> MagicMock:
    topology = MagicMock()
    topology.device = device
    return topology


@pytest.mark.unit
@pytest.mark.parametrize(
    "topology, device, expected",
    [
        (None, None, torch.device("cuda:0")),
        (mock_topology(torch.device("cuda:1")), None, torch.device("cuda:1")),
        (None, torch.device("cuda:2"), torch.device("cuda:2")),
    ],
)
def test_get_device(topology: MagicMock, device: torch.device, expected: torch.device):
    with patch("torch.cuda.current_device", return_value=0):
        assert (
            utils.get_device(
                topology=topology,
                device=device,
            )
            == expected
        )


@pytest.mark.unit
def test_get_device_raises_assertion_error():
    topology = mock_topology(torch.device("cuda:1"))
    device = torch.device("cuda:0")

    with pytest.raises(AssertionError):
        utils.get_device(
            topology=topology,
            device=device,
        )
