#!/usr/bin/env python3
"""Generate 2 new group splits for cross-split validation."""
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.training.splits import make_group_train_val_split, save_split_info

meta = pd.read_csv(ROOT / "outputs" / "task1_topo_v3" / "patch_metadata.csv")
group_col = "sample_id"
val_ratio = 0.2

for rs in [100, 200]:
    df_tr, df_va = make_group_train_val_split(meta, group_col, val_ratio, random_state=rs)
    out = ROOT / "outputs" / "task3_split" / f"split_info_rs{rs}.json"
    save_split_info(df_tr, df_va, group_col, rs, val_ratio, out)
    tr_ids = sorted(df_tr[group_col].unique().tolist())
    va_ids = sorted(df_va[group_col].unique().tolist())
    print(f"Split rs={rs}: train={len(df_tr)} patches ({len(tr_ids)} groups), val={len(df_va)} patches ({len(va_ids)} groups)")
    print(f"  Train IDs: {tr_ids}")
    print(f"  Val IDs:   {va_ids}")
    print(f"  Saved: {out}")
