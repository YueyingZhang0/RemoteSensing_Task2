from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SCEProbeCNN(nn.Module):
    """4-block CNN + GAP + MLP -> scalar regression probe."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        c = base_channels
        self.b1 = ConvBlock(in_channels, c)
        self.b2 = ConvBlock(c, c * 2)
        self.b3 = ConvBlock(c * 2, c * 4)
        self.b4 = ConvBlock(c * 4, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(c * 4, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.pool(self.b3(x))
        x = self.pool(self.b4(x))
        x = self.gap(x).flatten(1)
        return {"sci_pred": self.mlp(x)}
