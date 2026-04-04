from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from src.utils.io import save_json


def dice_loss(logits: torch.Tensor, mask: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (prob * mask).sum(dims)
    psum = prob.sum(dims)
    msum = mask.sum(dims)
    dice = (2 * inter + smooth) / (psum + msum + smooth)
    return 1.0 - dice.mean()


def seg_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return dice_loss(logits, mask) + 0.5 * F.binary_cross_entropy_with_logits(logits, mask)


def _per_sample_dice_recall(
    logits: torch.Tensor,
    mask: torch.Tensor,
    thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    m = mask.detach().cpu().numpy() > 0.5
    pred = prob > thresh
    b = prob.shape[0]
    dices: List[float] = []
    recalls: List[float] = []
    smooth = 1e-5
    for i in range(b):
        p = pred[i, 0].astype(np.float32)
        g = m[i, 0].astype(np.float32)
        inter = (p * g).sum()
        dices.append(float((2 * inter + smooth) / (p.sum() + g.sum() + smooth)))
        tp = ((p > 0) & (g > 0)).sum()
        fn = ((p == 0) & (g > 0)).sum()
        recalls.append(float(tp / (tp + fn + smooth)))
    return np.array(dices), np.array(recalls)


def collate_seg_sce(batch: List[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "sce": torch.stack([b["sce"] for b in batch]),
        "patch_name": [b["patch_name"] for b in batch],
        "is_hard": [bool(b.get("is_hard", False)) for b in batch],
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ours: bool,
) -> Dict[str, Any]:
    model.eval()
    dice_all: List[float] = []
    rec_all: List[float] = []
    dice_hard: List[float] = []
    rec_hard: List[float] = []
    sci_p: List[float] = []
    sci_g: List[float] = []

    for batch in loader:
        img = batch["image"].to(device)
        m = batch["mask"].to(device)
        sce = batch["sce"].to(device)
        hard = batch["is_hard"]

        if ours:
            logits, sci_pred = model(img, seg_only=False)
        else:
            logits = model(img)

        d, r = _per_sample_dice_recall(logits, m)
        for i in range(len(d)):
            dice_all.append(d[i])
            rec_all.append(r[i])
            if hard[i]:
                dice_hard.append(d[i])
                rec_hard.append(r[i])

        if ours:
            sci_p.extend(sci_pred.detach().cpu().numpy().ravel().tolist())
            sci_g.extend(sce.detach().cpu().numpy().ravel().tolist())

    out: Dict[str, Any] = {
        "val_dice_mean": float(np.mean(dice_all)) if dice_all else float("nan"),
        "val_recall_mean": float(np.mean(rec_all)) if rec_all else float("nan"),
    }
    if dice_hard:
        out["hard_dice_mean"] = float(np.mean(dice_hard))
        out["hard_recall_mean"] = float(np.mean(rec_hard))
    else:
        out["hard_dice_mean"] = float("nan")
        out["hard_recall_mean"] = float("nan")

    if ours and len(sci_p) > 2:
        x = np.asarray(sci_g, dtype=np.float64)
        y = np.asarray(sci_p, dtype=np.float64)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            out["val_sci_corr"] = None
        else:
            out["val_sci_corr"] = float(pearsonr(x, y)[0])
    else:
        out["val_sci_corr"] = None
    return out


def _plot_task3_curves(rows: List[dict], path: Path, ours: bool) -> None:
    ep = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(ep, [r["train_seg_loss"] for r in rows], label="train_seg_loss")
    axes[0].plot(ep, [r["val_dice_mean"] for r in rows], label="val_dice_mean")
    axes[0].set_ylabel("loss / dice")
    axes[0].legend(loc="best")
    axes[0].set_title("Task 3 training")
    axes[1].plot(ep, [r["val_recall_mean"] for r in rows], label="val_recall_mean", color="C2")
    if ours:
        sc = [r.get("val_sci_corr") for r in rows]
        sc_f = [float(x) if x is not None and not np.isnan(x) else np.nan for x in sc]
        axes[1].plot(ep, sc_f, label="val_sci_corr (Pearson)", color="C3")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("recall / corr")
    axes[1].legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_task3(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    ours: bool,
    out_dir: Path,
    warmup_epochs: int,
    joint_epochs: int,
    lr: float,
    weight_decay: float,
    lambda_sce: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, Any]] = []
    best_dice = -1.0
    best_state: Optional[dict] = None
    total_epochs = warmup_epochs + joint_epochs

    for epoch in range(total_epochs):
        phase_joint = epoch >= warmup_epochs
        model.train()
        ep_seg = 0.0
        n_batches = 0

        for batch in train_loader:
            img = batch["image"].to(device)
            m = batch["mask"].to(device)
            sce = batch["sce"].to(device)
            opt.zero_grad(set_to_none=True)

            if ours:
                logits, sci_pred = model(img, seg_only=not phase_joint)
                loss_s = seg_loss(logits, m)
                if phase_joint:
                    loss = loss_s + lambda_sce * F.smooth_l1_loss(sci_pred, sce)
                else:
                    loss = loss_s
            else:
                logits = model(img)
                loss_s = seg_loss(logits, m)
                loss = loss_s

            loss.backward()
            opt.step()
            ep_seg += float(loss_s.detach())
            n_batches += 1

        train_seg_mean = ep_seg / max(n_batches, 1)
        val_metrics = evaluate_loader(model, val_loader, device, ours=ours)
        row: Dict[str, Any] = {
            "epoch": epoch,
            "phase": "joint" if phase_joint else "warmup",
            "train_seg_loss": train_seg_mean,
            "val_dice_mean": val_metrics["val_dice_mean"],
            "val_recall_mean": val_metrics["val_recall_mean"],
            "hard_dice_mean": val_metrics["hard_dice_mean"],
            "hard_recall_mean": val_metrics["hard_recall_mean"],
            "val_sci_corr": val_metrics.get("val_sci_corr"),
        }
        history.append(row)

        vd = val_metrics["val_dice_mean"]
        if not np.isnan(vd) and vd > best_dice:
            best_dice = vd
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"model_state_dict": model.state_dict(), "best_val_dice": best_dice}, out_dir / "best_model.pt")
    save_json(history, out_dir / "history.json")
    _plot_task3_curves(history, out_dir / "train_curves.png", ours=ours)

    final_val = evaluate_loader(model, val_loader, device, ours=ours)
    save_json(final_val, out_dir / "val_metrics.json")

    hard_json = {
        "hard_dice_mean": final_val["hard_dice_mean"],
        "hard_recall_mean": final_val["hard_recall_mean"],
        "n_hard_patches_note": "subset of val where is_hard=True (top sci_res2_norm)",
    }
    if ours and final_val.get("val_sci_corr") is not None:
        hard_json["val_sci_corr_full_val"] = final_val["val_sci_corr"]
    save_json(hard_json, out_dir / "hard_patch_metrics.json")
