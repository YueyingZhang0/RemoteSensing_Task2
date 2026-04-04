from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_true)
    pear = float(pearsonr(y_true, y_pred)[0]) if n > 1 else float("nan")
    spear = float(spearmanr(y_true, y_pred)[0]) if n > 1 else float("nan")
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "pearson": pear,
        "spearman": spear,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
