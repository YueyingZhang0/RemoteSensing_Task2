from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from src.models.sce_probe_cnn import SCEProbeCNN
from src.utils.io import save_json


def load_frozen_sce_probe(
    checkpoint: Path,
    device: torch.device,
    *,
    in_channels: int = 3,
    base_channels: int = 32,
    mlp_hidden: int = 64,
    dropout: float = 0.0,
) -> nn.Module:
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = SCEProbeCNN(
        in_channels=in_channels,
        base_channels=base_channels,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


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


def _dice_loss_per_sample(logits: torch.Tensor, mask: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * mask).sum(dim=(2, 3))
    psum = prob.sum(dim=(2, 3))
    msum = mask.sum(dim=(2, 3))
    dice = (2 * inter + smooth) / (psum + msum + smooth)
    return 1.0 - dice.squeeze(1)


def _bce_per_sample(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, mask, reduction="none").mean(dim=(1, 2, 3))


def seg_loss_per_sample(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _dice_loss_per_sample(logits, mask) + 0.5 * _bce_per_sample(logits, mask)


def oracle_sample_weights(
    scores: torch.Tensor,
    tau: float,
    alpha: float,
    gamma: float,
    min_weight: float,
    max_weight: float,
    normalize_in_batch: bool,
) -> torch.Tensor:
    """scores: (B,) on device; w = clamp(1 + alpha * h^gamma, min, max), h = clip((s-tau)/(1-tau),0,1)."""
    denom = max(1.0 - tau, 1e-8)
    h = ((scores - tau) / denom).clamp(0.0, 1.0)
    w = 1.0 + alpha * (h ** gamma)
    w = w.clamp(min_weight, max_weight)
    if normalize_in_batch:
        w = w / w.mean().clamp_min(1e-8)
    return w


def smart_sample_weights(
    scores: torch.Tensor,
    fov: torch.Tensor,
    tau: float,
    alpha: float,
    gamma: float,
    fov_min: float,
    max_weight: float,
    normalize_in_batch: bool,
) -> torch.Tensor:
    """FOV-gated oracle weights: only upweight when fov >= fov_min and scores > tau; else w=1."""
    denom = max(1.0 - tau, 1e-8)
    h = ((scores - tau) / denom).clamp(0.0, 1.0)
    w = 1.0 + alpha * (h ** gamma)
    w = w.clamp(1.0, max_weight)
    gate = (fov >= fov_min) & (scores > tau)
    w = torch.where(gate, w, torch.ones_like(w))
    if normalize_in_batch:
        w = w / w.mean().clamp_min(1e-8)
    return w


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
        "sci_res2_norm": torch.tensor([float(b["sci_res2_norm"]) for b in batch], dtype=torch.float32),
        "fov_ratio": torch.tensor([float(b.get("fov_ratio", 1.0)) for b in batch], dtype=torch.float32),
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
            out = model(img)
            logits = out[0] if isinstance(out, tuple) else out

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


@torch.no_grad()
def evaluate_loader_threshold_sweep(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ours: bool,
    thresholds: List[float],
) -> Dict[str, Any]:
    """Aggregate val/hard dice+recall at each probability threshold (diagnostic; primary metric stays 0.5 in evaluate_loader)."""
    model.eval()
    logits_chunks: List[torch.Tensor] = []
    mask_chunks: List[torch.Tensor] = []
    hard_flags: List[bool] = []

    for batch in loader:
        img = batch["image"].to(device)
        m = batch["mask"].to(device)
        hard = batch["is_hard"]
        if ours:
            out_m = model(img, seg_only=False)
            logits = out_m[0]
        else:
            out_m = model(img)
            logits = out_m[0] if isinstance(out_m, tuple) else out_m
        logits_chunks.append(logits.detach().cpu())
        mask_chunks.append(m.detach().cpu())
        hard_flags.extend(bool(x) for x in hard)

    if not logits_chunks:
        return {
            "thresholds": thresholds,
            "per_threshold": {},
            "best_hard_dice_threshold": None,
            "best_val_dice_threshold": None,
            "fixed_threshold_reporting": 0.5,
        }

    logits_all = torch.cat(logits_chunks, dim=0)
    mask_all = torch.cat(mask_chunks, dim=0)
    hard_arr = np.asarray(hard_flags, dtype=bool)

    per_t: Dict[str, Dict[str, float]] = {}
    hard_dice_by_t: List[tuple[float, float]] = []
    val_dice_by_t: List[tuple[float, float]] = []

    for t in thresholds:
        d, r = _per_sample_dice_recall(logits_all, mask_all, thresh=float(t))
        val_dice = float(np.mean(d)) if len(d) else float("nan")
        val_rec = float(np.mean(r)) if len(r) else float("nan")
        if hard_arr.any():
            hd = float(np.mean(d[hard_arr]))
            hr = float(np.mean(r[hard_arr]))
        else:
            hd = float("nan")
            hr = float("nan")
        key = f"{t:.2f}"
        per_t[key] = {
            "val_dice_mean": val_dice,
            "val_recall_mean": val_rec,
            "hard_dice_mean": hd,
            "hard_recall_mean": hr,
        }
        if not np.isnan(val_dice):
            val_dice_by_t.append((float(t), val_dice))
        if not np.isnan(hd):
            hard_dice_by_t.append((float(t), hd))

    best_hard_t = max(hard_dice_by_t, key=lambda x: x[1])[0] if hard_dice_by_t else None
    best_val_t = max(val_dice_by_t, key=lambda x: x[1])[0] if val_dice_by_t else None

    return {
        "thresholds": thresholds,
        "per_threshold": per_t,
        "best_hard_dice_threshold": best_hard_t,
        "best_val_dice_threshold": best_val_t,
        "fixed_threshold_reporting": 0.5,
        "note": "Primary reported metrics use threshold 0.5 in val_metrics.json; sweep is diagnostic only.",
    }


def plot_threshold_curves(threshold_metrics: Dict[str, Any], out_dir: Path) -> None:
    per = threshold_metrics.get("per_threshold") or {}
    if not per:
        return
    keys = sorted(per.keys(), key=lambda k: float(k))
    xs = [float(k) for k in keys]
    val_d = [per[k]["val_dice_mean"] for k in keys]
    val_r = [per[k]["val_recall_mean"] for k in keys]
    hard_d = [per[k]["hard_dice_mean"] for k in keys]
    hard_r = [per[k]["hard_recall_mean"] for k in keys]

    def _one(ys: List[float], ylabel: str, fname: str) -> None:
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o", markersize=4)
        plt.xlabel("probability threshold")
        plt.ylabel(ylabel)
        plt.title(f"Threshold sweep: {ylabel}")
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=160)
        plt.close()

    _one(val_d, "val_dice_mean", "threshold_curve_val_dice.png")
    _one(val_r, "val_recall_mean", "threshold_curve_val_recall.png")
    _one(hard_d, "hard_dice_mean", "threshold_curve_hard_dice.png")
    _one(hard_r, "hard_recall_mean", "threshold_curve_hard_recall.png")


@torch.no_grad()
def evaluate_difficulty_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tau: float,
    out_dir: Path,
) -> Dict[str, Any]:
    """Collect predicted vs GT difficulty scores on val; compute stats, scatter, histogram."""
    model.eval()
    preds: List[float] = []
    gts: List[float] = []
    is_hard_flags: List[bool] = []

    for batch in loader:
        img = batch["image"].to(device)
        gt = batch["sci_res2_norm"]
        hard = batch["is_hard"]
        out = model(img)
        if isinstance(out, tuple):
            diff = out[1].squeeze(1).detach().cpu()
        else:
            continue
        preds.extend(diff.numpy().ravel().tolist())
        gts.extend(gt.numpy().ravel().tolist() if isinstance(gt, torch.Tensor) else [float(g) for g in gt])
        is_hard_flags.extend(bool(h) for h in hard)

    if len(preds) < 3:
        return {"error": "too few samples"}

    pa = np.asarray(preds, dtype=np.float64)
    ga = np.asarray(gts, dtype=np.float64)
    ha = np.asarray(is_hard_flags, dtype=bool)

    pearson_r = float(pearsonr(ga, pa)[0]) if np.std(pa) > 1e-12 and np.std(ga) > 1e-12 else None
    spearman_r = float(spearmanr(ga, pa).correlation) if np.std(pa) > 1e-12 else None

    n = len(pa)
    above_tau = int((pa > tau).sum())
    above_tau_on_hard = int((pa[ha] > tau).sum()) if ha.any() else 0
    n_hard = int(ha.sum())

    stats: Dict[str, Any] = {
        "val_sci_corr_pearson": pearson_r,
        "val_sci_corr_spearman": spearman_r,
        "pred_sci_mean": float(pa.mean()),
        "pred_sci_std": float(pa.std()),
        "gt_sci_mean": float(ga.mean()),
        "gt_sci_std": float(ga.std()),
        "pct_pred_above_tau": round(above_tau / n, 4) if n else None,
        "pct_pred_above_tau_on_true_hard": round(above_tau_on_hard / n_hard, 4) if n_hard else None,
        "n_val": n,
        "n_hard": n_hard,
        "tau": tau,
    }
    save_json(stats, out_dir / "difficulty_stats.json")

    plt.figure(figsize=(5, 5))
    plt.scatter(ga, pa, s=6, alpha=0.4)
    plt.xlabel("GT sci_res2_norm")
    plt.ylabel("Predicted difficulty")
    plt.title(f"Pred vs GT (Pearson={pearson_r or 0:.3f})")
    mn = min(float(ga.min()), float(pa.min()))
    mx = max(float(ga.max()), float(pa.max()))
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "pred_vs_gt_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(pa, bins=30, alpha=0.6, label="predicted")
    plt.hist(ga, bins=30, alpha=0.4, label="GT")
    plt.axvline(tau, color="r", linestyle="--", linewidth=1, label=f"tau={tau}")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.title("Difficulty score distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "pred_sci_hist.png", dpi=160)
    plt.close()

    return stats


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


def _plot_oracle_weighted_curves(rows: List[dict], path: Path) -> None:
    ep = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(ep, [r["train_seg_loss"] for r in rows], label="train_seg_loss")
    axes[0].plot(ep, [r["val_dice_mean"] for r in rows], label="val_dice_mean")
    axes[0].set_ylabel("loss / dice")
    axes[0].legend(loc="best")
    axes[0].set_title("Task 3 v2 oracle-weighted (baseline)")
    axes[1].plot(ep, [r["val_recall_mean"] for r in rows], label="val_recall_mean", color="C2")
    axes[1].plot(ep, [r["hard_dice_mean"] for r in rows], label="hard_dice_mean", color="C4")
    wm = [r.get("train_weight_mean", float("nan")) for r in rows]
    axes[1].plot(ep, wm, label="train_weight_mean", color="C3", linestyle="--")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("metrics / weight")
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
    threshold_sweep_thresholds: Optional[List[float]] = None,
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
                out_b = model(img)
                logits = out_b[0] if isinstance(out_b, tuple) else out_b
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
            "train_weight_mean": 1.0,
            "train_weight_std": 0.0,
            "train_weight_max": 1.0,
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

    if threshold_sweep_thresholds:
        tm = evaluate_loader_threshold_sweep(
            model, val_loader, device, ours=ours, thresholds=threshold_sweep_thresholds
        )
        save_json(tm, out_dir / "threshold_metrics.json")
        plot_threshold_curves(tm, out_dir)


def _effective_oracle_score_gamma(weight_mode: str, gamma_from_config: float) -> float:
    """Map weight_mode to exponent in w = 1 + alpha * h^gamma (score-only oracle weights)."""
    wm = str(weight_mode).lower()
    if wm == "oracle_true_linear":
        return 1.0
    if wm == "oracle_quadratic":
        return 2.0
    if wm in ("oracle_linear", "oracle_smart"):
        return float(gamma_from_config)
    raise ValueError(f"unknown weight_mode for gamma mapping: {weight_mode}")


def train_task3_baseline_oracle_weighted(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    total_epochs: int,
    lr: float,
    weight_decay: float,
    tau: float,
    alpha: float,
    gamma: float,
    difficulty_warmup_epochs: int,
    min_weight: float,
    max_weight: float,
    normalize_weights_in_batch: bool,
    save_weight_stats: bool,
    shuffle_scores_in_batch: bool = False,
    weight_mode: str = "oracle_linear",
    fov_min: float = 0.8,
    threshold_sweep_thresholds: Optional[List[float]] = None,
    debug_save_first_weighted_batch: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wm = str(weight_mode).lower()
    if wm not in ("oracle_linear", "oracle_smart", "oracle_true_linear", "oracle_quadratic"):
        raise ValueError("weight_mode must be oracle_linear | oracle_true_linear | oracle_quadratic | oracle_smart")
    gamma_cfg = float(gamma)
    gamma_eff = _effective_oracle_score_gamma(wm, gamma_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, Any]] = []
    weight_by_epoch: List[Dict[str, Any]] = []
    best_dice = -1.0
    best_state: Optional[dict] = None
    first_weighted_batch_saved = False

    for epoch in range(total_epochs):
        use_weights = epoch >= difficulty_warmup_epochs
        model.train()
        ep_seg = 0.0
        n_batches = 0
        all_w: List[float] = []

        for batch_idx, batch in enumerate(train_loader):
            img = batch["image"].to(device)
            m = batch["mask"].to(device)
            scores = batch["sci_res2_norm"].to(device)
            fov = batch["fov_ratio"].to(device)
            opt.zero_grad(set_to_none=True)

            logits = model(img)
            li = seg_loss_per_sample(logits, m)
            if use_weights:
                if shuffle_scores_in_batch:
                    perm = torch.randperm(scores.size(0), device=scores.device)
                    scores = scores[perm]
                if wm == "oracle_smart":
                    w = smart_sample_weights(
                        scores, fov, tau, alpha, gamma_eff, fov_min, max_weight, normalize_weights_in_batch
                    )
                else:
                    w = oracle_sample_weights(
                        scores, tau, alpha, gamma_eff, min_weight, max_weight, normalize_weights_in_batch
                    )
                if (
                    debug_save_first_weighted_batch
                    and not first_weighted_batch_saved
                    and batch_idx == 0
                ):
                    with torch.no_grad():
                        w_if_linear = oracle_sample_weights(
                            scores, tau, alpha, gamma_eff, min_weight, max_weight, normalize_weights_in_batch
                        )
                        w_if_smart = smart_sample_weights(
                            scores, fov, tau, alpha, gamma_eff, fov_min, max_weight, normalize_weights_in_batch
                        )
                    dbg: Dict[str, Any] = {
                        "weight_mode_used": wm,
                        "epoch": epoch,
                        "batch_index": batch_idx,
                        "shuffle_scores_in_batch": shuffle_scores_in_batch,
                        "tau": tau,
                        "alpha": alpha,
                        "gamma": gamma_eff,
                        "gamma_from_config": gamma_cfg,
                        "fov_min": fov_min,
                        "min_weight": min_weight,
                        "max_weight": max_weight,
                        "normalize_weights_in_batch": normalize_weights_in_batch,
                        "sci_res2_norm": scores.detach().float().cpu().tolist(),
                        "fov_ratio": fov.detach().float().cpu().tolist(),
                        "w_used": w.detach().float().cpu().tolist(),
                        "w_if_oracle_linear": w_if_linear.detach().float().cpu().tolist(),
                        "w_if_oracle_smart": w_if_smart.detach().float().cpu().tolist(),
                        "patch_name": list(batch.get("patch_name", [])),
                    }
                    save_json(dbg, out_dir / "debug_first_weighted_batch.json")
                    first_weighted_batch_saved = True
                loss = (li * w).sum() / w.sum().clamp_min(1e-8)
                all_w.extend(w.detach().float().cpu().numpy().tolist())
            else:
                loss = li.mean()
            loss.backward()
            opt.step()
            ep_seg += float(loss.detach())
            n_batches += 1

        train_seg_mean = ep_seg / max(n_batches, 1)
        val_metrics = evaluate_loader(model, val_loader, device, ours=False)
        if use_weights and all_w:
            arr = np.asarray(all_w, dtype=np.float64)
            tw_m, tw_s, tw_x = float(arr.mean()), float(arr.std()), float(arr.max())
        else:
            tw_m, tw_s, tw_x = 1.0, 0.0, 1.0

        row = {
            "epoch": epoch,
            "phase": "oracle_weighted" if use_weights else "weight_warmup",
            "train_seg_loss": train_seg_mean,
            "val_dice_mean": val_metrics["val_dice_mean"],
            "val_recall_mean": val_metrics["val_recall_mean"],
            "hard_dice_mean": val_metrics["hard_dice_mean"],
            "hard_recall_mean": val_metrics["hard_recall_mean"],
            "val_sci_corr": None,
            "train_weight_mean": tw_m,
            "train_weight_std": tw_s,
            "train_weight_max": tw_x,
            "weight_mode": wm,
        }
        history.append(row)
        if save_weight_stats:
            weight_by_epoch.append(
                {
                    "epoch": epoch,
                    "train_weight_mean": tw_m,
                    "train_weight_std": tw_s,
                    "train_weight_max": tw_x,
                }
            )

        vd = val_metrics["val_dice_mean"]
        if not np.isnan(vd) and vd > best_dice:
            best_dice = vd
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"model_state_dict": model.state_dict(), "best_val_dice": best_dice}, out_dir / "best_model.pt")
    save_json(history, out_dir / "history.json")
    _plot_oracle_weighted_curves(history, out_dir / "train_curves.png")

    final_val = evaluate_loader(model, val_loader, device, ours=False)
    if shuffle_scores_in_batch:
        final_val["loss_mode"] = "shuffle_oracle_ablation"
    elif wm == "oracle_smart":
        final_val["loss_mode"] = "oracle_smart"
    elif wm == "oracle_true_linear":
        final_val["loss_mode"] = "oracle_true_linear"
    elif wm == "oracle_quadratic":
        final_val["loss_mode"] = "oracle_quadratic"
    else:
        final_val["loss_mode"] = "oracle_linear"
    final_val["val_sci_corr"] = None
    save_json(final_val, out_dir / "val_metrics.json")

    hard_json = {
        "hard_dice_mean": final_val["hard_dice_mean"],
        "hard_recall_mean": final_val["hard_recall_mean"],
        "n_hard_patches_note": "subset of val where is_hard=True (top sci_res2_norm)",
        "loss_mode": final_val["loss_mode"],
    }
    save_json(hard_json, out_dir / "hard_patch_metrics.json")

    weight_stats: Dict[str, Any] = {
        "tau": tau,
        "alpha": alpha,
        "gamma": gamma_eff,
        "gamma_from_config": gamma_cfg,
        "warmup_epochs": difficulty_warmup_epochs,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "normalize_weights_in_batch": normalize_weights_in_batch,
        "shuffle_scores_in_batch": shuffle_scores_in_batch,
        "weight_mode": wm,
        "fov_min": fov_min,
        "debug_save_first_weighted_batch": debug_save_first_weighted_batch,
    }
    if save_weight_stats:
        weight_stats["by_epoch"] = weight_by_epoch
    save_json(weight_stats, out_dir / "weight_stats.json")

    if threshold_sweep_thresholds:
        tm = evaluate_loader_threshold_sweep(
            model, val_loader, device, ours=False, thresholds=threshold_sweep_thresholds
        )
        save_json(tm, out_dir / "threshold_metrics.json")
        plot_threshold_curves(tm, out_dir)


@torch.no_grad()
def _evaluate_probe_difficulty(
    probe: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tau: float,
    out_dir: Path,
) -> Dict[str, Any]:
    """Collect frozen-probe predictions vs GT on val; reuse evaluate_difficulty_predictions logic."""
    probe.eval()
    preds: List[float] = []
    gts: List[float] = []
    is_hard_flags: List[bool] = []

    for batch in loader:
        img = batch["image"].to(device)
        gt = batch["sci_res2_norm"]
        hard = batch["is_hard"]
        out = probe(img)
        pred = out["sci_pred"].squeeze(1).detach().cpu().clamp(0.0, 1.0)
        preds.extend(pred.numpy().ravel().tolist())
        gts.extend(gt.numpy().ravel().tolist() if isinstance(gt, torch.Tensor) else [float(g) for g in gt])
        is_hard_flags.extend(bool(h) for h in hard)

    if len(preds) < 3:
        return {"error": "too few samples"}

    pa = np.asarray(preds, dtype=np.float64)
    ga = np.asarray(gts, dtype=np.float64)
    ha = np.asarray(is_hard_flags, dtype=bool)

    pearson_r = float(pearsonr(ga, pa)[0]) if np.std(pa) > 1e-12 and np.std(ga) > 1e-12 else None
    spearman_r = float(spearmanr(ga, pa).correlation) if np.std(pa) > 1e-12 else None

    n = len(pa)
    above_tau = int((pa > tau).sum())
    above_tau_on_hard = int((pa[ha] > tau).sum()) if ha.any() else 0
    n_hard = int(ha.sum())

    stats: Dict[str, Any] = {
        "val_sci_corr_pearson": pearson_r,
        "val_sci_corr_spearman": spearman_r,
        "pred_sci_mean": float(pa.mean()),
        "pred_sci_std": float(pa.std()),
        "gt_sci_mean": float(ga.mean()),
        "gt_sci_std": float(ga.std()),
        "pct_pred_above_tau": round(above_tau / n, 4) if n else None,
        "pct_pred_above_tau_on_true_hard": round(above_tau_on_hard / n_hard, 4) if n_hard else None,
        "n_val": n,
        "n_hard": n_hard,
        "tau": tau,
    }
    save_json(stats, out_dir / "difficulty_stats.json")

    plt.figure(figsize=(5, 5))
    plt.scatter(ga, pa, s=6, alpha=0.4)
    plt.xlabel("GT sci_res2_norm")
    plt.ylabel("Predicted difficulty (frozen probe)")
    plt.title(f"Probe vs GT (Pearson={pearson_r or 0:.3f})")
    mn = min(float(ga.min()), float(pa.min()))
    mx = max(float(ga.max()), float(pa.max()))
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "pred_vs_gt_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(pa, bins=30, alpha=0.6, label="predicted (probe)")
    plt.hist(ga, bins=30, alpha=0.4, label="GT")
    plt.axvline(tau, color="r", linestyle="--", linewidth=1, label=f"tau={tau}")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.title("Frozen probe difficulty distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "pred_sci_hist.png", dpi=160)
    plt.close()

    return stats


def train_task3_baseline_probe_weighted(
    model: nn.Module,
    probe: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    total_epochs: int,
    lr: float,
    weight_decay: float,
    tau: float,
    alpha: float,
    gamma: float,
    difficulty_warmup_epochs: int,
    min_weight: float,
    max_weight: float,
    normalize_weights_in_batch: bool,
    save_weight_stats: bool,
    ramp_epochs: int = 0,
    threshold_sweep_thresholds: Optional[List[float]] = None,
    debug_save_first_weighted_batch: bool = False,
) -> None:
    """Per-sample weights from frozen SCEProbeCNN predictions with gradual ramp-up."""
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, Any]] = []
    weight_by_epoch: List[Dict[str, Any]] = []
    best_dice = -1.0
    best_state: Optional[dict] = None
    first_weighted_batch_saved = False

    t_warmup = int(difficulty_warmup_epochs)
    t_ramp = int(ramp_epochs)

    def ramp_lambda(ep: int) -> float:
        if ep < t_warmup:
            return 0.0
        elapsed = ep - t_warmup
        if t_ramp <= 0 or elapsed >= t_ramp:
            return 1.0
        return elapsed / t_ramp

    for epoch in range(total_epochs):
        use_weights = epoch >= difficulty_warmup_epochs
        lam_t = ramp_lambda(epoch)
        model.train()
        ep_loss = 0.0
        n_batches = 0
        all_w: List[float] = []

        for batch_idx, batch in enumerate(train_loader):
            img = batch["image"].to(device)
            m = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                scores = probe(img)["sci_pred"].squeeze(1).clamp(0.0, 1.0)

            logits = model(img)
            li = seg_loss_per_sample(logits, m)
            if use_weights:
                w_raw = oracle_sample_weights(
                    scores, tau, alpha, gamma, min_weight, max_weight, normalize_weights_in_batch
                )
                w = 1.0 + lam_t * (w_raw - 1.0)
                w = w.clamp(1.0, max_weight)
                loss = (li * w).sum() / w.sum().clamp_min(1e-8)
                all_w.extend(w.detach().float().cpu().numpy().tolist())

                if debug_save_first_weighted_batch and not first_weighted_batch_saved and batch_idx == 0:
                    gt_scores = batch.get("sci_res2_norm")
                    dbg: Dict[str, Any] = {
                        "weight_mode_used": "frozen_probe",
                        "epoch": epoch,
                        "batch_index": batch_idx,
                        "phase": "probe_weighted",
                        "ramp_lambda": lam_t,
                        "tau": tau,
                        "alpha": alpha,
                        "gamma": gamma,
                        "min_weight": min_weight,
                        "max_weight": max_weight,
                        "predicted_scores": scores.detach().float().cpu().tolist(),
                        "sci_res2_norm_gt": gt_scores.tolist() if gt_scores is not None else None,
                        "w_raw": w_raw.detach().float().cpu().tolist(),
                        "w_effective": w.detach().float().cpu().tolist(),
                        "patch_name": list(batch.get("patch_name", [])),
                    }
                    save_json(dbg, out_dir / "debug_first_weighted_batch.json")
                    first_weighted_batch_saved = True
            else:
                loss = li.mean()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            n_batches += 1

        train_loss_mean = ep_loss / max(n_batches, 1)
        val_metrics = evaluate_loader(model, val_loader, device, ours=False)
        if use_weights and all_w:
            arr = np.asarray(all_w, dtype=np.float64)
            tw_m, tw_s, tw_x = float(arr.mean()), float(arr.std()), float(arr.max())
        else:
            tw_m, tw_s, tw_x = 1.0, 0.0, 1.0

        row = {
            "epoch": epoch,
            "phase": "probe_weighted" if use_weights else "weight_warmup",
            "ramp_lambda": round(lam_t, 4) if use_weights else None,
            "train_seg_loss": train_loss_mean,
            "val_dice_mean": val_metrics["val_dice_mean"],
            "val_recall_mean": val_metrics["val_recall_mean"],
            "hard_dice_mean": val_metrics["hard_dice_mean"],
            "hard_recall_mean": val_metrics["hard_recall_mean"],
            "val_sci_corr": None,
            "train_weight_mean": tw_m,
            "train_weight_std": tw_s,
            "train_weight_max": tw_x,
        }
        history.append(row)
        if save_weight_stats:
            weight_by_epoch.append(
                {
                    "epoch": epoch,
                    "train_weight_mean": tw_m,
                    "train_weight_std": tw_s,
                    "train_weight_max": tw_x,
                    "ramp_lambda": round(lam_t, 4) if use_weights else None,
                }
            )

        vd = val_metrics["val_dice_mean"]
        if not np.isnan(vd) and vd > best_dice:
            best_dice = vd
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"model_state_dict": model.state_dict(), "best_val_dice": best_dice}, out_dir / "best_model.pt")
    save_json(history, out_dir / "history.json")
    _plot_oracle_weighted_curves(history, out_dir / "train_curves.png")

    final_val = evaluate_loader(model, val_loader, device, ours=False)
    final_val["loss_mode"] = "frozen_probe_weighted"
    final_val["val_sci_corr"] = None
    save_json(final_val, out_dir / "val_metrics.json")

    hard_json = {
        "hard_dice_mean": final_val["hard_dice_mean"],
        "hard_recall_mean": final_val["hard_recall_mean"],
        "n_hard_patches_note": "subset of val where is_hard=True (top sci_res2_norm)",
        "loss_mode": "frozen_probe_weighted",
    }
    save_json(hard_json, out_dir / "hard_patch_metrics.json")

    diff_stats = _evaluate_probe_difficulty(probe, val_loader, device, tau, out_dir)
    final_val["difficulty_stats"] = diff_stats

    weight_stats: Dict[str, Any] = {
        "tau": tau,
        "alpha": alpha,
        "gamma": gamma,
        "warmup_epochs": difficulty_warmup_epochs,
        "ramp_epochs": ramp_epochs,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "normalize_weights_in_batch": normalize_weights_in_batch,
        "score_source": "frozen_probe",
    }
    if save_weight_stats:
        weight_stats["by_epoch"] = weight_by_epoch
    save_json(weight_stats, out_dir / "weight_stats.json")

    if threshold_sweep_thresholds:
        tm = evaluate_loader_threshold_sweep(
            model, val_loader, device, ours=False, thresholds=threshold_sweep_thresholds
        )
        save_json(tm, out_dir / "threshold_metrics.json")
        plot_threshold_curves(tm, out_dir)


def train_task3_joint_difficulty(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    total_epochs: int,
    lr: float,
    weight_decay: float,
    tau: float,
    alpha: float,
    gamma: float,
    difficulty_warmup_epochs: int,
    min_weight: float,
    max_weight: float,
    normalize_weights_in_batch: bool,
    save_weight_stats: bool,
    joint_calib_epochs: int,
    joint_oracle_weight_epochs: int,
    lambda_diff: float,
    pred_weight_detach: bool,
    ramp_epochs: int = 0,
    threshold_sweep_thresholds: Optional[List[float]] = None,
    debug_save_first_weighted_batch: bool = False,
) -> None:
    """
    UNet + difficulty head: calibrate head with uniform seg + aux loss, optional oracle-weighted
    phase, then per-sample weights from predicted difficulty with gradual ramp-up.

    Ramp-up: in the first `ramp_epochs` of the pred_weight phase, lambda_t linearly
    rises from 0 to 1 so that w_eff = 1 + lambda_t * (w_raw - 1).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, Any]] = []
    weight_by_epoch: List[Dict[str, Any]] = []
    best_dice = -1.0
    best_state: Optional[dict] = None
    first_pred_batch_saved = False

    t_calib = int(joint_calib_epochs)
    t_oracle_w = int(joint_oracle_weight_epochs)
    t_dw = int(difficulty_warmup_epochs)
    t_ramp = int(ramp_epochs)
    t_pred_start = t_dw + t_calib + t_oracle_w
    if t_pred_start > total_epochs:
        raise ValueError(
            "difficulty_warmup_epochs + joint_calib_epochs + joint_oracle_weight_epochs must be <= total_epochs"
        )

    def phase_for_epoch(ep: int) -> str:
        if ep < t_dw:
            return "weight_warmup"
        if ep < t_dw + t_calib:
            return "joint_calib"
        if ep < t_dw + t_calib + t_oracle_w:
            return "joint_oracle_weight"
        return "joint_pred_weight"

    def ramp_lambda(ep: int) -> float:
        """Ramp coefficient for the pred_weight phase: 0->1 over ramp_epochs."""
        if ep < t_pred_start:
            return 0.0
        elapsed = ep - t_pred_start
        if t_ramp <= 0 or elapsed >= t_ramp:
            return 1.0
        return elapsed / t_ramp

    for epoch in range(total_epochs):
        ph = phase_for_epoch(epoch)
        lam_t = ramp_lambda(epoch)
        model.train()
        ep_loss = 0.0
        n_batches = 0
        all_w: List[float] = []

        for batch_idx, batch in enumerate(train_loader):
            img = batch["image"].to(device)
            m = batch["mask"].to(device)
            gt_scores = batch["sci_res2_norm"].to(device)
            opt.zero_grad(set_to_none=True)

            logits, diff_pred = model(img)
            d = diff_pred.squeeze(1)
            li = seg_loss_per_sample(logits, m)

            if ph == "weight_warmup":
                loss = li.mean()
            elif ph == "joint_calib":
                loss = li.mean() + lambda_diff * F.smooth_l1_loss(d, gt_scores)
            elif ph == "joint_oracle_weight":
                w = oracle_sample_weights(
                    gt_scores, tau, alpha, gamma, min_weight, max_weight, normalize_weights_in_batch
                )
                loss = (li * w).sum() / w.sum().clamp_min(1e-8) + lambda_diff * F.smooth_l1_loss(d, gt_scores)
                all_w.extend(w.detach().float().cpu().numpy().tolist())
            else:
                s_for_w = d.detach() if pred_weight_detach else d
                s_for_w = s_for_w.clamp(0.0, 1.0)
                w_raw = oracle_sample_weights(
                    s_for_w, tau, alpha, gamma, min_weight, max_weight, normalize_weights_in_batch
                )
                w = 1.0 + lam_t * (w_raw - 1.0)
                w = w.clamp(1.0, max_weight)
                loss = (li * w).sum() / w.sum().clamp_min(1e-8) + lambda_diff * F.smooth_l1_loss(d, gt_scores)
                all_w.extend(w.detach().float().cpu().numpy().tolist())

                if debug_save_first_weighted_batch and not first_pred_batch_saved and batch_idx == 0:
                    dbg: Dict[str, Any] = {
                        "weight_mode_used": "joint_pred_weight",
                        "epoch": epoch,
                        "batch_index": batch_idx,
                        "phase": ph,
                        "ramp_lambda": lam_t,
                        "tau": tau,
                        "alpha": alpha,
                        "gamma": gamma,
                        "min_weight": min_weight,
                        "max_weight": max_weight,
                        "pred_weight_detach": pred_weight_detach,
                        "normalize_weights_in_batch": normalize_weights_in_batch,
                        "sci_res2_norm_gt": gt_scores.detach().float().cpu().tolist(),
                        "predicted_difficulty": d.detach().float().cpu().tolist(),
                        "w_raw": w_raw.detach().float().cpu().tolist(),
                        "w_effective": w.detach().float().cpu().tolist(),
                        "patch_name": list(batch.get("patch_name", [])),
                    }
                    save_json(dbg, out_dir / "debug_first_weighted_batch.json")
                    first_pred_batch_saved = True

            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            n_batches += 1

        train_loss_mean = ep_loss / max(n_batches, 1)
        val_metrics = evaluate_loader(model, val_loader, device, ours=False)
        if ph in ("joint_oracle_weight", "joint_pred_weight") and all_w:
            arr = np.asarray(all_w, dtype=np.float64)
            tw_m, tw_s, tw_x = float(arr.mean()), float(arr.std()), float(arr.max())
        else:
            tw_m, tw_s, tw_x = 1.0, 0.0, 1.0

        row = {
            "epoch": epoch,
            "phase": ph,
            "ramp_lambda": round(lam_t, 4) if ph == "joint_pred_weight" else None,
            "train_seg_loss": train_loss_mean,
            "val_dice_mean": val_metrics["val_dice_mean"],
            "val_recall_mean": val_metrics["val_recall_mean"],
            "hard_dice_mean": val_metrics["hard_dice_mean"],
            "hard_recall_mean": val_metrics["hard_recall_mean"],
            "val_sci_corr": None,
            "train_weight_mean": tw_m,
            "train_weight_std": tw_s,
            "train_weight_max": tw_x,
        }
        history.append(row)
        if save_weight_stats:
            weight_by_epoch.append(
                {
                    "epoch": epoch,
                    "train_weight_mean": tw_m,
                    "train_weight_std": tw_s,
                    "train_weight_max": tw_x,
                    "phase": ph,
                    "ramp_lambda": round(lam_t, 4) if ph == "joint_pred_weight" else None,
                }
            )

        vd = val_metrics["val_dice_mean"]
        if not np.isnan(vd) and vd > best_dice:
            best_dice = vd
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({"model_state_dict": model.state_dict(), "best_val_dice": best_dice}, out_dir / "best_model.pt")
    save_json(history, out_dir / "history.json")
    _plot_oracle_weighted_curves(history, out_dir / "train_curves.png")

    final_val = evaluate_loader(model, val_loader, device, ours=False)
    final_val["loss_mode"] = "joint_difficulty"
    final_val["val_sci_corr"] = None
    save_json(final_val, out_dir / "val_metrics.json")

    hard_json = {
        "hard_dice_mean": final_val["hard_dice_mean"],
        "hard_recall_mean": final_val["hard_recall_mean"],
        "n_hard_patches_note": "subset of val where is_hard=True (top sci_res2_norm)",
        "loss_mode": "joint_difficulty",
    }
    save_json(hard_json, out_dir / "hard_patch_metrics.json")

    diff_stats = evaluate_difficulty_predictions(model, val_loader, device, tau, out_dir)
    final_val["difficulty_stats"] = diff_stats

    weight_stats: Dict[str, Any] = {
        "tau": tau,
        "alpha": alpha,
        "gamma": gamma,
        "difficulty_warmup_epochs": difficulty_warmup_epochs,
        "joint_calib_epochs": joint_calib_epochs,
        "joint_oracle_weight_epochs": joint_oracle_weight_epochs,
        "ramp_epochs": ramp_epochs,
        "lambda_diff": lambda_diff,
        "pred_weight_detach": pred_weight_detach,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "normalize_weights_in_batch": normalize_weights_in_batch,
        "score_source": "joint_head",
    }
    if save_weight_stats:
        weight_stats["by_epoch"] = weight_by_epoch
    save_json(weight_stats, out_dir / "weight_stats.json")

    if threshold_sweep_thresholds:
        tm = evaluate_loader_threshold_sweep(
            model, val_loader, device, ours=False, thresholds=threshold_sweep_thresholds
        )
        save_json(tm, out_dir / "threshold_metrics.json")
        plot_threshold_curves(tm, out_dir)

