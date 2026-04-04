#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    spearman = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def fit_and_eval_single_feature(df_train, df_val, feature_col, target_col, degree=1):
    x_train = df_train[[feature_col]].values
    y_train = df_train[target_col].values
    x_val = df_val[[feature_col]].values
    y_val = df_val[target_col].values

    if degree == 1:
        model = LinearRegression()
    else:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lin", LinearRegression())
        ])

    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)

    train_metrics = compute_metrics(y_train, pred_train)
    val_metrics = compute_metrics(y_val, pred_val)

    return model, train_metrics, val_metrics, pred_val


def fit_and_eval_multi_feature(df_train, df_val, feature_cols, target_col, degree=1):
    x_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    x_val = df_val[feature_cols].values
    y_val = df_val[target_col].values

    if degree == 1:
        model = LinearRegression()
    else:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lin", LinearRegression())
        ])

    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)

    train_metrics = compute_metrics(y_train, pred_train)
    val_metrics = compute_metrics(y_val, pred_val)

    return model, train_metrics, val_metrics, pred_val


def save_scatter(x, y_true, y_pred, out_path, title, xlabel, ylabel="sci_res"):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y_true, alpha=0.5, label="GT")
    plt.scatter(x, y_pred, alpha=0.5, label="Pred")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_pred_vs_gt(y_true, y_pred, out_path, title, target_col="sci_res"):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel(f"GT {target_col}")
    plt.ylabel(f"Pred {target_col}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="sci_res2_norm")
    parser.add_argument("--group_col", type=str, default="sample_id")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata_csv)

    required_cols = ["vessel_area", "fov_ratio", args.target_col, args.group_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # image-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.random_state)
    train_idx, val_idx = next(gss.split(df, groups=df[args.group_col]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    results = {}
    target = args.target_col

    single_baselines = [
        ("area_linear",    "vessel_area", 1),
        ("area_quadratic", "vessel_area", 2),
        ("fov_linear",     "fov_ratio",   1),
        ("fov_quadratic",  "fov_ratio",   2),
    ]

    for name, feat, deg in single_baselines:
        model, train_m, val_m, pred_val = fit_and_eval_single_feature(
            df_train, df_val, feature_col=feat, target_col=target, degree=deg
        )
        results[name] = {"train": train_m, "val": val_m}
        save_scatter(
            df_val[feat].values, df_val[target].values, pred_val,
            out_dir / f"{name}_scatter.png",
            f"{name} baseline", feat, ylabel=target,
        )
        save_pred_vs_gt(
            df_val[target].values, pred_val,
            out_dir / f"{name}_pred_vs_gt.png",
            f"{name}: pred vs gt", target_col=target,
        )

    # area+fov combined baselines
    multi_feats = ["vessel_area", "fov_ratio"]
    for deg, suffix in [(1, "area_fov_linear"), (2, "area_fov_quadratic")]:
        model, train_m, val_m, pred_val = fit_and_eval_multi_feature(
            df_train, df_val, feature_cols=multi_feats, target_col=target, degree=deg
        )
        results[suffix] = {"train": train_m, "val": val_m}
        save_pred_vs_gt(
            df_val[target].values, pred_val,
            out_dir / f"{suffix}_pred_vs_gt.png",
            f"{suffix}: pred vs gt", target_col=target,
        )

    with open(out_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=== Train/Val split ===")
    print(f"target column: {target}")
    print(f"train patches: {len(df_train)}")
    print(f"val patches:   {len(df_val)}")
    print(f"train images:  {df_train[args.group_col].nunique()}")
    print(f"val images:    {df_val[args.group_col].nunique()}")

    header = f"{'Baseline':<25s} {'Pearson':>8s} {'Spearman':>8s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}"
    sep = "-" * len(header)
    print(f"\n=== Validation results (target: {target}) ===")
    print(header)
    print(sep)
    for name, item in results.items():
        m = item["val"]
        print(
            f"{name:<25s} {m['pearson']:>8.4f} {m['spearman']:>8.4f} "
            f"{m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}"
        )


if __name__ == "__main__":
    main()