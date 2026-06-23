"""VGG-16 (Simonyan & Zisserman, ICLR 2015) — student implementation.

You may NOT use ``torchvision.models.vgg16`` or ``timm.create_model``.
You MAY read those reference implementations and re-type the architecture.

Skeleton scaffolding (forward pass, head wiring) is provided. Fill in the
``# TODO`` blocks.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.heads import MultiTaskHead


# Standard VGG-16 layer configuration (numbers = output channels, "M" = maxpool).
VGG16_CFG = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]


def make_vgg_layers(cfg: list, batch_norm: bool = True) -> nn.Sequential:
    """Build the convolutional feature extractor from ``cfg``.

    Hard-coded per the VGG paper (Simonyan & Zisserman, ICLR 2015, Table 1,
    configuration D). Every conv uses 3x3 kernels with stride 1 and padding 1
    (so the spatial size is preserved across a conv), and each "M" stage halves
    the spatial size with a 2x2 stride-2 max pool. ReLU follows every conv;
    the -BN variant inserts BatchNorm2d between conv and ReLU.
    """
    layers: list[nn.Module] = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """VGG-16-BN with a multi-task classification head."""

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = make_vgg_layers(VGG16_CFG, batch_norm=True)

        # After 5 maxpool stages, 224x224 -> 7x7. Channels = 512.
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Classifier MLP, hard-coded per the VGG paper (FC-4096 -> FC-4096).
        # The original VGG ends with a third FC-1000 softmax layer; here we stop
        # at the 4096-dim feature vector and hand it to the multi-task head.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head = MultiTaskHead(in_features=4096, dropout=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return self.head(x)
