from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_scatter_gt_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    title: str,
    target_col: str,
) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, s=12)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    plt.xlabel(f"GT {target_col}")
    plt.ylabel(f"Pred {target_col}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_train_val_curves(
    train_vals: List[float],
    val_vals: List[float],
    save_path: str | Path,
    title: str = "Training curves",
    ylabel: str = "SmoothL1 loss",
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(train_vals, label="train")
    plt.plot(val_vals, label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_scatter_feature(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y_true, alpha=0.5, label="GT")
    plt.scatter(x, y_pred, alpha=0.5, label="Pred")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
