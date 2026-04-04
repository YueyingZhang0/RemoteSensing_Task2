from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.utils.io import save_json


def make_group_train_val_split(
    df: pd.DataFrame,
    group_col: str = "sample_id",
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df[group_col]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    return df_train, df_val


def mark_hard_patches(
    df_val: pd.DataFrame,
    target_col: str,
    hard_ratio: float = 0.2,
) -> pd.DataFrame:
    threshold = float(np.quantile(df_val[target_col].values, 1.0 - hard_ratio))
    df_val = df_val.copy()
    df_val["is_hard"] = df_val[target_col] >= threshold
    return df_val


def save_split_info(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    group_col: str,
    random_state: int,
    val_ratio: float,
    path: str | Path,
) -> None:
    info = {
        "train_sample_ids": sorted(df_train[group_col].unique().tolist()),
        "val_sample_ids": sorted(df_val[group_col].unique().tolist()),
        "random_state": random_state,
        "val_ratio": val_ratio,
        "train_patches": len(df_train),
        "val_patches": len(df_val),
    }
    save_json(info, path)


def load_split_info(path: str | Path) -> dict:
    from src.utils.io import load_json
    return load_json(path)
