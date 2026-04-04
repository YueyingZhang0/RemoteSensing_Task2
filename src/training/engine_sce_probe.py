from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.training.losses import regression_loss
from src.training.metrics import regression_metrics
from src.utils.io import save_json
from src.utils.plotting import plot_scatter_gt_pred, plot_train_val_curves


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = regression_loss(out["sci_pred"], targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return {"loss": total_loss / max(n, 1)}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_pred: List[float] = []
    all_gt: List[float] = []
    all_names: List[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        out = model(images)
        loss = regression_loss(out["sci_pred"], targets)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_pred.extend(out["sci_pred"].cpu().numpy().ravel().tolist())
        all_gt.extend(targets.cpu().numpy().ravel().tolist())
        all_names.extend(batch["patch_name"])

    y_true = np.array(all_gt)
    y_pred = np.array(all_pred)
    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(n, 1)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["patch_names"] = all_names
    return metrics


def fit_sce_probe(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int,
    patience: int,
    output_dir: str | Path,
    target_col: str = "sci_res2_norm",
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience_left = patience
    best_state: Optional[dict] = None

    for epoch in range(epochs):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = validate_one_epoch(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()

        train_losses.append(tr["loss"])
        val_losses.append(va["loss"])

        if va["loss"] < best_val_loss - 1e-8:
            best_val_loss = va["loss"]
            best_epoch = epoch
            patience_left = patience
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_col": target_col,
            "epoch": best_epoch,
            "val_loss": best_val_loss,
        },
        out / "best_sce_branch.pt",
    )

    va_final = validate_one_epoch(model, val_loader, device)
    tr_final = validate_one_epoch(model, train_loader, device)

    val_scalar = {k: v for k, v in va_final.items() if k not in ("y_true", "y_pred", "patch_names")}
    train_scalar = {k: v for k, v in tr_final.items() if k not in ("y_true", "y_pred", "patch_names")}

    metrics = {
        "config": {"target_col": target_col, "epochs_ran": len(train_losses), "best_epoch": best_epoch},
        "train": train_scalar,
        "val": val_scalar,
    }
    save_json(metrics, out / "metrics.json")

    history = {"train_loss": train_losses, "val_loss": val_losses}
    save_json(history, out / "history.json")

    plot_train_val_curves(train_losses, val_losses, out / "train_curve.png")
    plot_scatter_gt_pred(va_final["y_true"], va_final["y_pred"], out / "best_scatter.png", "SCE probe: val", target_col)
    plot_scatter_gt_pred(va_final["y_true"], va_final["y_pred"], out / "pred_vs_gt.png", "SCE probe: pred vs GT", target_col)

    return {
        "metrics": metrics,
        "val_result": va_final,
        "best_epoch": best_epoch,
    }
