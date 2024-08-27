from collections import OrderedDict

import torch
from einops import rearrange
from PIL.Image import Image
from torchvision.transforms import (  # type: ignore
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC  # type: ignore


def convert_image_to_rgb(image: Image) -> Image:
    return image.convert("RGB")


def clip_transform(n_px: tuple[int, int]) -> Compose:
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()

        # all conv layers have stride 1. an  avgpool is performed after the second convolution when stride > 1
        self.conv1 = torch.nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)

        self.conv2 = torch.nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.avgpool = torch.nn.AvgPool2d(stride) if stride > 1 else torch.nn.Identity()

        self.conv3 = torch.nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes * self.expansion)

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("-1", torch.nn.AvgPool2d(stride)),
                        (
                            "0",
                            torch.nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", torch.nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ClipModifiedResNet(torch.nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs antialiasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers: list[int], num_init_channels: int = 64):
        super().__init__()

        # the 3-layer stem
        self.conv1 = torch.nn.Conv2d(3, num_init_channels // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_init_channels // 2)
        self.conv2 = torch.nn.Conv2d(
            num_init_channels // 2,
            num_init_channels // 2,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = torch.nn.BatchNorm2d(num_init_channels // 2)
        self.conv3 = torch.nn.Conv2d(
            num_init_channels // 2,
            num_init_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn3 = torch.nn.BatchNorm2d(num_init_channels)
        self.avgpool = torch.nn.AvgPool2d(2)
        self.relu = torch.nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = num_init_channels  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(num_init_channels, layers[0])
        self.layer2 = self._make_layer(num_init_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(num_init_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(num_init_channels * 8, layers[3], stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> torch.nn.Sequential:
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def stem(x: torch.Tensor) -> torch.Tensor:
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3),
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = rearrange(x, pattern="b d h w -> b (h w) d")

        return x
