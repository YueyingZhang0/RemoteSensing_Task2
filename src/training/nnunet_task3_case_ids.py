"""nnU-Net Task3 export: case_id assignment (must match export_task3_for_nnunet_v2)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def sanitize_nnunet_case_stem(patch_name: str) -> str:
    stem = Path(patch_name).stem
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_")
    return s or "case"


def build_nnunet_case_rows(df_tr: pd.DataFrame, df_va: pd.DataFrame) -> List[Dict[str, Any]]:
    """Same ordering and disambiguation as export: train rows first, then val."""
    seen: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for split, df in (("train", df_tr), ("val", df_va)):
        for _, row in df.iterrows():
            name = str(row["patch_name"])
            cid = sanitize_nnunet_case_stem(name)
            base = cid
            n = 0
            while cid in seen:
                n += 1
                cid = f"{base}_{n}"
            seen.add(cid)
            rows.append({"patch_name": name, "case_id": cid, "split": split})
    return rows


def patch_name_to_case_id(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    return {r["patch_name"]: r["case_id"] for r in rows}
