from contextlib import nullcontext as does_not_raise
from unittest import mock

import pytest

from scaling.transformer.utils.get_tflops import HardwareType


@pytest.mark.parametrize(
    "expected,name",
    [
        (HardwareType.A100, "NVIDIA A100-SXM4-80GB"),
        (HardwareType.A100, "NVIDIA A100-SXM4-40GB"),
        (HardwareType.H100, "NVIDIA H100 PCIe"),
        (HardwareType.RTX3090, "NVIDIA GeForce RTX 3090"),
        (HardwareType.RTX4090, "NVIDIA GeForce RTX 4090"),
        (HardwareType.DEFAULT, "NVIDIA Some Other weird Graphics Card"),
    ],
)
def test_get_via_torch_all(expected: HardwareType, name: str) -> None:
    with mock.patch("torch.cuda.get_device_name", lambda: name):
        assert expected == HardwareType.get_via_torch()


def test_get_via_torch():
    """
    Assert that we know the GPU the tests run on
    """
    with does_not_raise():
        HardwareType.get_via_torch()


def test_max_tflops():
    """
    Test that the max_tflops property works for all defined hardware types
    """
    for hardware_type in HardwareType:
        assert isinstance(hardware_type.max_tflops, float)
