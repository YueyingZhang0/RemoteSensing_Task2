"""Resolve HRF/all image / manual1 / FOV paths for a given stem (e.g. 01_dr)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict


def hrf_case_paths(stem: str, project_root: Path) -> Dict[str, Path]:
    base = project_root / "HRF" / "all"
    img_dir = base / "images"
    ip: Path | None = None
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        c = img_dir / f"{stem}{ext}"
        if c.is_file():
            ip = c
            break
    if ip is None:
        raise FileNotFoundError(f"No image for stem {stem!r} under {img_dir}")
    gt = base / "manual1" / f"{stem}.tif"
    fov = base / "mask" / f"{stem}_mask.tif"
    if not gt.is_file():
        raise FileNotFoundError(gt)
    if not fov.is_file():
        raise FileNotFoundError(fov)
    return {"case_id": stem, "image": ip, "gt": gt, "fov": fov}
