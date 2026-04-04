from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.utils.io import load_json, save_json


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


def make_nested_group_split(
    df: pd.DataFrame,
    group_col: str,
    val_ratio: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group-wise split of rows in df into disjoint train/val; both subsets stay inside df."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df[group_col]))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )


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
    return load_json(path)


def load_or_create_group_split(
    df: pd.DataFrame,
    group_col: str,
    val_ratio: float,
    random_state: int,
    split_json: str | Path | None,
    project_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_ids = set(df[group_col].map(lambda x: str(x)).unique())

    path: Optional[Path] = None
    if split_json:
        path = Path(split_json)
        if not path.is_absolute():
            path = project_root / path

    if path is not None and path.is_file():
        info = load_split_info(path)
        train_ids = set(str(x) for x in info["train_sample_ids"])
        val_ids = set(str(x) for x in info["val_sample_ids"])
        if train_ids & val_ids:
            raise ValueError(f"Train/val sample_id overlap in {path}")
        if train_ids | val_ids != all_ids:
            raise ValueError(
                f"split_json does not match metadata rows: expected {len(all_ids)} groups, "
                f"got train+val={len(train_ids | val_ids)}"
            )
        m = df[group_col].map(lambda x: str(x))
        df_tr = df[m.isin(train_ids)].reset_index(drop=True)
        df_va = df[m.isin(val_ids)].reset_index(drop=True)
        return df_tr, df_va

    df_tr, df_va = make_group_train_val_split(df, group_col, val_ratio, random_state)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_split_info(df_tr, df_va, group_col, random_state, val_ratio, path)
    return df_tr, df_va