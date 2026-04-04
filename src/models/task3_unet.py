from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetBaseline(nn.Module):
    """Compact U-Net for 128x128 -> logits (B,1,H,W)."""

    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bot = DoubleConv(c4, c5)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bot(self.pool(e4))

        d = self.up4(b)
        d = self.dec4(torch.cat([d, e4], dim=1))
        d = self.up3(d)
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = self.up2(d)
        d = self.dec2(torch.cat([d, e2], dim=1))
        d = self.up1(d)
        d = self.dec1(torch.cat([d, e1], dim=1))
        return self.out_conv(d)


class UNetWithSCE(nn.Module):
    """
    U-Net + SCE branch on bottleneck + residual soft modulation on shallowest skip (e2, 64x64).
    """

    def __init__(self, in_channels: int = 3, base: int = 32, mod_eps: float = 0.15):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.mod_eps = mod_eps

        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bot = DoubleConv(c4, c5)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

        self.sce_gap = nn.AdaptiveAvgPool2d(1)
        self.sce_emb = nn.Sequential(
            nn.Linear(c5, 128),
            nn.ReLU(inplace=True),
        )
        self.sce_head = nn.Linear(128, 1)
        self.mod_fc = nn.Linear(128, c2)

    def forward(self, x: torch.Tensor, seg_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bot(self.pool(e4))

        if seg_only:
            e2_mod = e2
            sci_pred = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        else:
            gap = self.sce_gap(b).flatten(1)
            emb = self.sce_emb(gap)
            sci_pred = self.sce_head(emb)
            gam = torch.tanh(self.mod_fc(emb)).view(-1, e2.shape[1], 1, 1)
            e2_mod = e2 * (1.0 + self.mod_eps * gam)

        d = self.up4(b)
        d = self.dec4(torch.cat([d, e4], dim=1))
        d = self.up3(d)
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = self.up2(d)
        d = self.dec2(torch.cat([d, e2_mod], dim=1))
        d = self.up1(d)
        d = self.dec1(torch.cat([d, e1], dim=1))
        logits = self.out_conv(d)
        return logits, sci_pred
