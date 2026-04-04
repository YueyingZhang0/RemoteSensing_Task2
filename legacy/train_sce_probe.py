#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a minimal CNN probe to regress sci_res2_norm from patch RGB images.
Uses image-level split by sample_id (GroupShuffleSplit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else float("nan")
    spearman = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else float("nan")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }


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


class SCEProbe(nn.Module):
    """Light CNN + GAP + MLP -> scalar."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.b1 = ConvBlock(in_ch, 32)
        self.b2 = ConvBlock(32, 64)
        self.b3 = ConvBlock(64, 128)
        self.b4 = ConvBlock(128, 128)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.pool(self.b3(x))
        x = self.pool(self.b4(x))
        x = self.gap(x).flatten(1)
        return self.mlp(x).squeeze(1)


class PatchSCIDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        target_col: str,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.target_col = target_col
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        name = str(row["patch_name"])
        path = self.images_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing patch image: {path}")
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment:
            if np.random.rand() > 0.5:
                arr = np.fliplr(arr).copy()
            if np.random.rand() > 0.5:
                arr = np.flipud(arr).copy()
        x = torch.from_numpy(arr).permute(2, 0, 1)
        y = float(row[self.target_col])
        return x, torch.tensor(y, dtype=torch.float32), name


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    total = 0.0
    n = 0
    preds: List[float] = []
    gts: List[float] = []
    names: List[str] = []
    for xb, yb, nb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total += loss.item() * xb.size(0)
        n += xb.size(0)
        preds.extend(pred.cpu().numpy().tolist())
        gts.extend(yb.cpu().numpy().tolist())
        names.extend(list(nb))
    avg_loss = total / max(n, 1)
    return avg_loss, np.array(preds), np.array(gts), names


def save_scatter_gt_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str, target_col: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, s=12)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    plt.xlabel(f"GT {target_col}")
    plt.ylabel(f"Pred {target_col}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_train_curve(train_losses: List[float], val_losses: List[float], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("SmoothL1 loss")
    plt.legend()
    plt.title("Training curves")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_ranked_predictions(
    names: List[str],
    preds: np.ndarray,
    df_meta: pd.DataFrame,
    out_path: Path,
    descending: bool,
    top_k: int,
) -> None:
    order = np.argsort(-preds if descending else preds)
    lines = []
    for i in order[:top_k]:
        name = names[int(i)]
        p = float(preds[int(i)])
        sub = df_meta[df_meta["patch_name"] == name]
        if len(sub) == 0:
            lines.append(f"{name}\tpred={p:.6f}\t(no metadata row)\n")
            continue
        r = sub.iloc[0]
        lines.append(
            f"{name}\tpred={p:.6f}\tgt={float(r.get('sci_res2_norm', float('nan'))):.6f}\t"
            f"vessel_area={int(r['vessel_area'])}\tfov_ratio={float(r['fov_ratio']):.4f}\t"
            f"junctions={int(r['junction_count'])}\tend_eff={int(r['endpoint_eff'])}\n"
        )
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SCE probe CNN for sci_res2_norm regression.")
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="outputs/task1_topo_v3/patch_metadata.csv",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="outputs/task1_topo_v3/patches/images",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/task2_sce_probe")
    parser.add_argument("--target_col", type=str, default="sci_res2_norm")
    parser.add_argument("--group_col", type=str, default="sample_id")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metadata_csv.is_file():
        raise FileNotFoundError(f"metadata_csv not found: {metadata_csv}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    df = pd.read_csv(metadata_csv)
    required = ["patch_name", args.target_col, args.group_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.random_state)
    train_idx, val_idx = next(gss.split(df, groups=df[args.group_col]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    split_info = {
        "train_sample_ids": sorted(df_train[args.group_col].unique().tolist()),
        "val_sample_ids": sorted(df_val[args.group_col].unique().tolist()),
        "random_state": args.random_state,
        "val_ratio": args.val_ratio,
    }
    (out_dir / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    device = torch.device(args.device)
    train_ds = PatchSCIDataset(df_train, images_dir, args.target_col, augment=True)
    val_ds = PatchSCIDataset(df_val, images_dir, args.target_col, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SCEProbe().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val = float("inf")
    best_epoch = -1
    patience_left = args.early_stop_patience
    best_state: Optional[dict] = None

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(tr)
        val_losses.append(va)
        if va < best_val - 1e-8:
            best_val = va
            best_epoch = epoch
            patience_left = args.early_stop_patience
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = out_dir / "best_sce_branch.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_col": args.target_col,
            "epoch": best_epoch,
            "val_loss": best_val,
        },
        ckpt_path,
    )

    train_loss_final, pred_tr, gt_tr, _ = evaluate(model, train_loader, criterion, device)
    val_loss_final, pred_val, gt_val, names_val = evaluate(model, val_loader, criterion, device)

    train_metrics = compute_metrics(gt_tr, pred_tr)
    val_metrics = compute_metrics(gt_val, pred_val)

    metrics = {
        "config": {
            "metadata_csv": str(metadata_csv),
            "images_dir": str(images_dir),
            "target_col": args.target_col,
            "epochs_ran": len(train_losses),
            "best_epoch": best_epoch,
        },
        "train": {"loss_smooth_l1": train_loss_final, **train_metrics},
        "val": {"loss_smooth_l1": val_loss_final, **val_metrics},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_scatter_gt_pred(
        gt_val,
        pred_val,
        out_dir / "best_scatter.png",
        "SCE probe: val pred vs GT",
        args.target_col,
    )
    save_scatter_gt_pred(
        gt_val,
        pred_val,
        out_dir / "pred_vs_gt.png",
        "SCE probe: val pred vs GT",
        args.target_col,
    )
    save_train_curve(train_losses, val_losses, out_dir / "train_curve.png")

    write_ranked_predictions(names_val, pred_val, df_val, out_dir / "highest_pred.txt", descending=True, top_k=20)
    write_ranked_predictions(names_val, pred_val, df_val, out_dir / "lowest_pred.txt", descending=False, top_k=20)

    print("=== SCE probe training done ===")
    print(f"best epoch: {best_epoch}, best val loss: {best_val:.6f}")
    print(f"val Pearson={val_metrics['pearson']:.4f} Spearman={val_metrics['spearman']:.4f}")
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
