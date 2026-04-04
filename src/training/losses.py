from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_smooth_l1 = nn.SmoothL1Loss()


def regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _smooth_l1(pred, target)
