"""Build Task-3 segmentation backbones (same I/O as UNetBaseline: logits Bx1xHxW)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.task3_unet import UNetBaseline


class _BilinearResizeIO(nn.Module):
    """Run a core network at ``core_size``; expose I/O at ``io_size`` (e.g. 128 -> 224 -> 128)."""

    def __init__(self, core: nn.Module, *, io_size: int, core_size: int) -> None:
        super().__init__()
        self.core = core
        self.io_size = int(io_size)
        self.core_size = int(core_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.core_size or x.shape[-2] != self.core_size:
            x = F.interpolate(
                x, size=(self.core_size, self.core_size), mode="bilinear", align_corners=False
            )
        y = self.core(x)
        if y.shape[-1] != self.io_size or y.shape[-2] != self.io_size:
            y = F.interpolate(
                y, size=(self.io_size, self.io_size), mode="bilinear", align_corners=False
            )
        return y


def build_task3_seg_model(
    arch: str,
    *,
    in_channels: int = 3,
    image_size: int = 128,
) -> nn.Module:
    """Return a model with forward(x) -> logits (B, 1, H, W).

    Architectures:
    - ``unet`` / ``baseline``: compact in-repo U-Net (``UNetBaseline``, base=32).
    - ``attention_unet``: MONAI Attention U-Net (2D). Requires ``monai``.
    - ``swin_unet``: U-Net decoder with Swin-Tiny encoder (``tu-swin_*`` via SMP/timm). SMP asserts
      height/width 224 for this encoder, so inputs are **bilinearly resized to 224 inside** and logits
      resized back to ``image_size`` (default 128). Fallback: ``mit_b0`` encoder if ``tu-swin`` is
      unavailable in the installed SMP build.

    Args:
        arch: One of ``unet``, ``baseline``, ``attention_unet``, ``swin_unet``.
        in_channels: RGB = 3.
        image_size: Patch spatial size (passed to timm via SMP ``encoder_params`` when supported).
    """
    a = (arch or "unet").strip().lower()
    if a in ("unet", "baseline"):
        return UNetBaseline(in_channels=in_channels, base=32)
    if a == "attention_unet":
        try:
            from monai.networks.nets import AttentionUnet
        except ImportError as e:
            raise ImportError(
                "attention_unet requires monai. Install: pip install monai"
            ) from e
        return AttentionUnet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
        )
    if a == "swin_unet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "swin_unet requires segmentation_models_pytorch. "
                "Install: pip install segmentation_models_pytorch"
            ) from e
        kw: dict = dict(
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
        try:
            core = smp.Unet(
                encoder_name="tu-swin_tiny_patch4_window7_224",
                **kw,
            )
            return _BilinearResizeIO(core, io_size=image_size, core_size=224)
        except KeyError:
            core = smp.Unet(encoder_name="mit_b0", **kw)
            return core
    raise ValueError(
        f"Unknown arch={arch!r}. Expected unet|baseline|attention_unet|swin_unet."
    )
