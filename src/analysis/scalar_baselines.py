from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.training.metrics import regression_metrics
from src.utils.io import save_json
from src.utils.plotting import plot_scatter_gt_pred, plot_scatter_feature


def _fit_eval(x_train, y_train, x_val, y_val, degree: int):
    if degree == 1:
        model = LinearRegression()
    else:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lin", LinearRegression()),
        ])
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)
    return pred_train, pred_val


def run_scalar_regression_baselines(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    target_col: str,
    output_dir: str | Path,
) -> Dict[str, dict]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    y_train = df_train[target_col].values
    y_val = df_val[target_col].values
    results: Dict[str, dict] = {}

    single = [
        ("area_linear", "vessel_area", 1),
        ("area_quadratic", "vessel_area", 2),
        ("fov_linear", "fov_ratio", 1),
        ("fov_quadratic", "fov_ratio", 2),
    ]
    for name, feat, deg in single:
        x_tr = df_train[[feat]].values
        x_va = df_val[[feat]].values
        pred_tr, pred_va = _fit_eval(x_tr, y_train, x_va, y_val, deg)
        results[name] = {
            "train": regression_metrics(y_train, pred_tr),
            "val": regression_metrics(y_val, pred_va),
        }
        plot_scatter_feature(
            x_va.ravel(), y_val, pred_va, out / f"{name}_scatter.png",
            f"{name} baseline", feat, target_col,
        )
        plot_scatter_gt_pred(
            y_val, pred_va, out / f"{name}_pred_vs_gt.png",
            f"{name}: pred vs gt", target_col,
        )

    multi_feats = ["vessel_area", "fov_ratio"]
    for deg, name in [(1, "area_fov_linear"), (2, "area_fov_quadratic")]:
        x_tr = df_train[multi_feats].values
        x_va = df_val[multi_feats].values
        pred_tr, pred_va = _fit_eval(x_tr, y_train, x_va, y_val, deg)
        results[name] = {
            "train": regression_metrics(y_train, pred_tr),
            "val": regression_metrics(y_val, pred_va),
        }
        plot_scatter_gt_pred(
            y_val, pred_va, out / f"{name}_pred_vs_gt.png",
            f"{name}: pred vs gt", target_col,
        )

    save_json(results, out / "baseline_results.json")

    print("=== Scalar baseline validation results ===")
    hdr = f"{'Baseline':<25s} {'Pearson':>8s} {'Spearman':>8s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for name, item in results.items():
        m = item["val"]
        print(f"{name:<25s} {m['pearson']:>8.4f} {m['spearman']:>8.4f} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")

    return results
