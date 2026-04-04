#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_sci_targets_stable_v3.py

Upgrade of build_sci_targets_stable_v2.py.

What this script does
---------------------
1) Extract DRIVE patches with FOV filtering
2) Compute patch-level vessel geometry + topology statistics
3) Build multiple SCI targets:
   - sci_raw          : skeleton_len / vessel_area
   - sci_base         : log1p(sci_raw)
   - sci_topo         : sci_base enhanced by topology density
   - sci_res          : sci_topo with vessel-area trend removed
   - sci_res2         : sci_res with FOV-ratio trend removed
4) Export patch_metadata.csv
5) Optionally save patch crops
6) Save histograms / boxplots / scatter diagnostics / ranked examples / collages

Recommended starting settings
-----------------------------
    patch_size = 128
    stride = 64
    min_fov_ratio = 0.8
    min_vessel_pixels = 120
    target for Task 2/3 = sci_res2_norm

Important note
--------------
For a quick Task-1 diagnostic run, this script fits the area-trend and FOV-trend
residualization using all retained patches. For formal Task-2/3 training and
validation, fit both trend models on the training split only, then apply those
fitted models to validation/test patches.
"""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import pearsonr, spearmanr
from skimage.morphology import skeletonize


# -----------------------------
# Basic I/O helpers
# -----------------------------


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_gray(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# File matching / patch helpers
# -----------------------------


def numeric_key(name: str) -> str:
    nums = re.findall(r"\d+", name)
    return nums[0] if nums else Path(name).stem


def build_file_map(folder: Path) -> Dict[str, Path]:
    files = [p for p in folder.iterdir() if p.is_file()]
    mapping: Dict[str, Path] = {}
    for p in files:
        mapping[numeric_key(p.name)] = p
    return mapping


def binarize_mask(arr: np.ndarray, thr: int = 127) -> np.ndarray:
    return (arr > thr).astype(np.uint8)


def safe_crop(arr: np.ndarray, y: int, x: int, patch_size: int) -> np.ndarray:
    return arr[y:y + patch_size, x:x + patch_size]


def compute_fov_ratio(fov_patch: np.ndarray) -> float:
    return float(fov_patch.mean())


def sliding_positions(height: int, width: int, patch_size: int, stride: int) -> List[Tuple[int, int]]:
    ys = list(range(0, max(height - patch_size + 1, 1), stride))
    xs = list(range(0, max(width - patch_size + 1, 1), stride))

    if len(ys) == 0 or ys[-1] != height - patch_size:
        ys.append(max(height - patch_size, 0))
    if len(xs) == 0 or xs[-1] != width - patch_size:
        xs.append(max(width - patch_size, 0))

    positions: List[Tuple[int, int]] = []
    for y in ys:
        for x in xs:
            positions.append((y, x))
    return positions


# -----------------------------
# Topology helpers
# -----------------------------


def erode_valid_region(valid_bin: np.ndarray, margin: int = 3) -> np.ndarray:
    """
    Erode valid FOV to avoid counting fake endpoints/junctions near patch/FOV borders.
    """
    if margin <= 0:
        return valid_bin.astype(np.uint8)

    structure = np.ones((3, 3), dtype=np.uint8)
    out = valid_bin.astype(np.uint8).copy()
    for _ in range(margin):
        out = ndi.binary_erosion(out, structure=structure).astype(np.uint8)
    return out


def count_neighbors_8(skel: np.ndarray) -> np.ndarray:
    """
    Count 8-neighbors for each skeleton pixel.
    """
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    conv = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    neigh = np.where(skel > 0, conv - 10, 0)
    return neigh


def count_node_components(node_mask: np.ndarray) -> int:
    """
    Count endpoint/junction nodes more robustly than raw pixel counting.
    Adjacent node pixels are merged into one node component.
    """
    if node_mask.sum() == 0:
        return 0
    structure = np.ones((3, 3), dtype=np.uint8)
    _, n = ndi.label(node_mask.astype(np.uint8), structure=structure)
    return int(n)


def compute_topology_stats(
    vessel_patch: np.ndarray,
    fov_patch: np.ndarray,
    border_margin: int = 3,
) -> Dict[str, float]:
    """
    Compute vessel geometry + topology statistics inside valid FOV.

    Returns:
        vessel_area
        skeleton_len
        junction_count
        endpoint_count
        component_count
        endpoint_eff = max(endpoint_count - 2 * component_count, 0)
    """
    vessel_bin = (vessel_patch > 0).astype(np.uint8)
    valid_bin = (fov_patch > 0).astype(np.uint8)

    vessel_valid = vessel_bin * valid_bin
    vessel_area = int(vessel_valid.sum())

    if vessel_area == 0:
        return {
            "vessel_area": 0,
            "skeleton_len": 0,
            "junction_count": 0,
            "endpoint_count": 0,
            "component_count": 0,
            "endpoint_eff": 0,
        }

    skel = skeletonize(vessel_valid.astype(bool)).astype(np.uint8)
    skeleton_len = int(skel.sum())

    valid_inner = erode_valid_region(valid_bin, margin=border_margin)
    skel_inner = skel * valid_inner

    if skel_inner.sum() == 0:
        return {
            "vessel_area": vessel_area,
            "skeleton_len": skeleton_len,
            "junction_count": 0,
            "endpoint_count": 0,
            "component_count": 0,
            "endpoint_eff": 0,
        }

    neighbor_count = count_neighbors_8(skel_inner)

    endpoint_mask = ((skel_inner > 0) & (neighbor_count == 1)).astype(np.uint8)
    junction_mask = ((skel_inner > 0) & (neighbor_count >= 3)).astype(np.uint8)

    endpoint_count = count_node_components(endpoint_mask)
    junction_count = count_node_components(junction_mask)

    structure = np.ones((3, 3), dtype=np.uint8)
    _, component_count = ndi.label(skel_inner.astype(np.uint8), structure=structure)
    component_count = int(component_count)

    endpoint_eff = max(endpoint_count - 2 * component_count, 0)

    return {
        "vessel_area": vessel_area,
        "skeleton_len": skeleton_len,
        "junction_count": junction_count,
        "endpoint_count": endpoint_count,
        "component_count": component_count,
        "endpoint_eff": endpoint_eff,
    }


# -----------------------------
# SCI target builders
# -----------------------------


def compute_sci_legacy(
    vessel_area: int,
    skeleton_len: int,
    eps: float = 1e-5,
    area_ref: float = 150.0,
) -> Tuple[float, float]:
    """
    Legacy-compatible target family kept for downstream compatibility.
    Returns:
        sci_raw    = skeleton_len / vessel_area
        sci_stable = log1p(sci_raw) * min(1, vessel_area / area_ref)
    """
    if vessel_area <= 0 or skeleton_len <= 0:
        return 0.0, 0.0

    sci_raw = skeleton_len / (vessel_area + eps)
    thinness = math.log1p(sci_raw)
    area_factor = min(1.0, vessel_area / area_ref)
    sci_stable = thinness * area_factor
    return float(sci_raw), float(sci_stable)


def compute_sci_topo(
    vessel_patch: np.ndarray,
    fov_patch: np.ndarray,
    eps: float = 1e-5,
    border_margin: int = 3,
    wj: float = 1.0,
    we: float = 0.35,
    wc: float = 1.5,
    lambda_topo: float = 2.0,
) -> Dict[str, float]:
    """
    New target family:
        sci_base = log1p(L / (A + eps))
        topo_density = wj * J/L + we * Eeff/L + wc * max(C-1,0)/L
        sci_topo = sci_base * (1 + lambda_topo * topo_density)
    """
    stats = compute_topology_stats(
        vessel_patch=vessel_patch,
        fov_patch=fov_patch,
        border_margin=border_margin,
    )

    A = stats["vessel_area"]
    L = stats["skeleton_len"]
    J = stats["junction_count"]
    Eeff = stats["endpoint_eff"]
    C = stats["component_count"]

    if A <= 0 or L <= 0:
        stats["sci_raw"] = 0.0
        stats["sci_base"] = 0.0
        stats["topo_density"] = 0.0
        stats["sci_topo"] = 0.0
        return stats

    sci_raw = L / (A + eps)
    sci_base = math.log1p(sci_raw)

    topo_density = (
        wj * J / (L + eps)
        + we * Eeff / (L + eps)
        + wc * max(C - 1, 0) / (L + eps)
    )
    sci_topo = sci_base * (1.0 + lambda_topo * topo_density)

    stats["sci_raw"] = float(sci_raw)
    stats["sci_base"] = float(sci_base)
    stats["topo_density"] = float(topo_density)
    stats["sci_topo"] = float(sci_topo)
    return stats


# -----------------------------
# Residualization / normalization
# -----------------------------


def fit_area_trend_linear(area_arr: np.ndarray, sci_arr: np.ndarray) -> np.ndarray:
    x = area_arr.astype(np.float64)
    X = np.stack([np.ones_like(x), x], axis=1)
    beta, *_ = np.linalg.lstsq(X, sci_arr.astype(np.float64), rcond=None)
    return beta


def predict_area_trend_linear(area_arr: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x = area_arr.astype(np.float64)
    return beta[0] + beta[1] * x


def fit_area_trend_log_linear(area_arr: np.ndarray, sci_arr: np.ndarray) -> np.ndarray:
    x = np.log1p(area_arr.astype(np.float64))
    X = np.stack([np.ones_like(x), x], axis=1)
    beta, *_ = np.linalg.lstsq(X, sci_arr.astype(np.float64), rcond=None)
    return beta


def predict_area_trend_log_linear(area_arr: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x = np.log1p(area_arr.astype(np.float64))
    return beta[0] + beta[1] * x


def fit_area_trend(area_arr: np.ndarray, sci_arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return fit_area_trend_linear(area_arr, sci_arr)
    if mode == "log_linear":
        return fit_area_trend_log_linear(area_arr, sci_arr)
    raise ValueError(f"Unsupported area trend mode: {mode}")


def predict_area_trend(area_arr: np.ndarray, beta: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return predict_area_trend_linear(area_arr, beta)
    if mode == "log_linear":
        return predict_area_trend_log_linear(area_arr, beta)
    raise ValueError(f"Unsupported area trend mode: {mode}")


def fit_poly_trend(x_arr: np.ndarray, y_arr: np.ndarray, degree: int) -> np.ndarray:
    x = x_arr.astype(np.float64)
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    X = np.stack(cols, axis=1)
    beta, *_ = np.linalg.lstsq(X, y_arr.astype(np.float64), rcond=None)
    return beta


def predict_poly_trend(x_arr: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x = x_arr.astype(np.float64)
    degree = len(beta) - 1
    y = np.zeros_like(x, dtype=np.float64)
    for d in range(degree + 1):
        y += beta[d] * (x ** d)
    return y


def fit_fov_trend(fov_arr: np.ndarray, sci_arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return fit_poly_trend(fov_arr, sci_arr, degree=1)
    if mode == "quadratic":
        return fit_poly_trend(fov_arr, sci_arr, degree=2)
    raise ValueError(f"Unsupported FOV trend mode: {mode}")


def predict_fov_trend(fov_arr: np.ndarray, beta: np.ndarray, mode: str) -> np.ndarray:
    if mode in {"linear", "quadratic"}:
        return predict_poly_trend(fov_arr, beta)
    raise ValueError(f"Unsupported FOV trend mode: {mode}")


def robust_minmax(values: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    lo = np.percentile(values, q_low)
    hi = np.percentile(values, q_high)
    clipped = np.clip(values, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-8)


# -----------------------------
# Plotting / reports
# -----------------------------


def plot_histogram(values: np.ndarray, save_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=40)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_boxplot(values: np.ndarray, save_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.boxplot(
        values,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red"),
    )
    plt.title(title)
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    save_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    alpha: float = 0.35,
) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=10, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def summarize_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    p = float(pearsonr(x, y)[0])
    s = float(spearmanr(x, y)[0])
    return p, s


def save_ranked_examples(rows: List[dict], save_path: Path, top_k: int = 10) -> None:
    rank_specs = [
        ("sci_raw", True),
        ("sci_topo", True),
        ("sci_res", True),
        ("sci_res2", True),
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        for key, descending in rank_specs:
            rows_sorted = sorted(rows, key=lambda r: r[key], reverse=descending)

            f.write(f"=== Top {key} patches ===\n")
            for row in rows_sorted[:top_k]:
                f.write(
                    f'{row["patch_name"]}\t{key}={row[key]:.6f}\t'
                    f'vessel_area={row["vessel_area"]}\tsk_len={row["skeleton_len"]}\t'
                    f'junctions={row["junction_count"]}\tend_eff={row["endpoint_eff"]}\t'
                    f'components={row["component_count"]}\tfov_ratio={row["fov_ratio"]}\n'
                )

            f.write(f"\n=== Bottom {key} patches ===\n")
            for row in rows_sorted[-top_k:]:
                f.write(
                    f'{row["patch_name"]}\t{key}={row[key]:.6f}\t'
                    f'vessel_area={row["vessel_area"]}\tsk_len={row["skeleton_len"]}\t'
                    f'junctions={row["junction_count"]}\tend_eff={row["endpoint_eff"]}\t'
                    f'components={row["component_count"]}\tfov_ratio={row["fov_ratio"]}\n'
                )
            f.write("\n")


def visualize_top_bottom_patches(rows: List[dict], output_dir: Path, top_k: int = 10) -> None:
    patch_dir = output_dir / "patches" / "images"
    vis_dir = output_dir / "top_bottom_vis"
    ensure_dir(vis_dir)

    def make_collage(rows_subset: List[dict], title_suffix: str) -> None:
        patches = []
        for row in rows_subset[:top_k]:
            p = patch_dir / row["patch_name"]
            if p.exists():
                im = Image.open(p).convert("RGB")
                patches.append(np.array(im))

        if not patches:
            return

        n_rows = 2
        n_cols = 5
        h, w = patches[0].shape[:2]
        collage = np.zeros((n_rows * h, n_cols * w, 3), dtype=np.uint8)

        for idx, im in enumerate(patches):
            r = idx // n_cols
            c = idx % n_cols
            collage[r * h:(r + 1) * h, c * w:(c + 1) * w, :] = im

        Image.fromarray(collage).save(vis_dir / f"{title_suffix}.png")

    for key in ["sci_raw", "sci_topo", "sci_res", "sci_res2"]:
        rows_sorted = sorted(rows, key=lambda r: r[key], reverse=True)
        make_collage(rows_sorted, f"top10_{key}")
        make_collage(list(reversed(rows_sorted)), f"bottom10_{key}")


def save_summary_report(
    output_dir: Path,
    args: argparse.Namespace,
    total_patches: int,
    skip_fov: int,
    skip_vessel: int,
    rows: List[dict],
    area_beta: np.ndarray,
    area_trend_mode: str,
    fov_beta: np.ndarray,
    fov_trend_mode: str,
) -> None:
    area_arr = np.array([r["vessel_area"] for r in rows], dtype=np.float64)
    sci_stable_arr = np.array([r["sci_stable"] for r in rows], dtype=np.float64)
    sci_topo_arr = np.array([r["sci_topo"] for r in rows], dtype=np.float64)
    sci_res_arr = np.array([r["sci_res"] for r in rows], dtype=np.float64)
    sci_res2_arr = np.array([r["sci_res2"] for r in rows], dtype=np.float64)
    fov_arr = np.array([r["fov_ratio"] for r in rows], dtype=np.float64)

    p_area_stable, s_area_stable = summarize_correlation(area_arr, sci_stable_arr)
    p_area_topo, s_area_topo = summarize_correlation(area_arr, sci_topo_arr)
    p_area_res, s_area_res = summarize_correlation(area_arr, sci_res_arr)
    p_area_res2, s_area_res2 = summarize_correlation(area_arr, sci_res2_arr)
    p_fov_topo, s_fov_topo = summarize_correlation(fov_arr, sci_topo_arr)
    p_fov_res, s_fov_res = summarize_correlation(fov_arr, sci_res_arr)
    p_fov_res2, s_fov_res2 = summarize_correlation(fov_arr, sci_res2_arr)

    report_path = output_dir / "summary_stats.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Task 1 summary ===\n")
        f.write(f"patch_size={args.patch_size}\n")
        f.write(f"stride={args.stride}\n")
        f.write(f"min_fov_ratio={args.min_fov_ratio}\n")
        f.write(f"min_vessel_pixels={args.min_vessel_pixels}\n")
        f.write(f"border_margin={args.border_margin}\n")
        f.write(f"wj={args.wj}\n")
        f.write(f"we={args.we}\n")
        f.write(f"wc={args.wc}\n")
        f.write(f"lambda_topo={args.lambda_topo}\n")
        f.write(f"legacy_area_ref={args.area_ref}\n")
        f.write(f"area_trend_mode={area_trend_mode}\n")
        f.write(f"area_trend_beta={area_beta.tolist()}\n")
        f.write(f"fov_trend_mode={fov_trend_mode}\n")
        f.write(f"fov_trend_beta={fov_beta.tolist()}\n\n")

        f.write(f"total_patches={total_patches}\n")
        f.write(f"filtered_by_fov={skip_fov}\n")
        f.write(f"filtered_by_vessel_area={skip_vessel}\n")
        f.write(f"valid_patches={len(rows)}\n\n")

        f.write("=== Correlation diagnostics ===\n")
        f.write(f"Pearson(vessel_area, sci_stable)={p_area_stable:.6f}\n")
        f.write(f"Spearman(vessel_area, sci_stable)={s_area_stable:.6f}\n")
        f.write(f"Pearson(vessel_area, sci_topo)={p_area_topo:.6f}\n")
        f.write(f"Spearman(vessel_area, sci_topo)={s_area_topo:.6f}\n")
        f.write(f"Pearson(vessel_area, sci_res)={p_area_res:.6f}\n")
        f.write(f"Spearman(vessel_area, sci_res)={s_area_res:.6f}\n")
        f.write(f"Pearson(vessel_area, sci_res2)={p_area_res2:.6f}\n")
        f.write(f"Spearman(vessel_area, sci_res2)={s_area_res2:.6f}\n")
        f.write(f"Pearson(fov_ratio, sci_topo)={p_fov_topo:.6f}\n")
        f.write(f"Spearman(fov_ratio, sci_topo)={s_fov_topo:.6f}\n")
        f.write(f"Pearson(fov_ratio, sci_res)={p_fov_res:.6f}\n")
        f.write(f"Spearman(fov_ratio, sci_res)={s_fov_res:.6f}\n")
        f.write(f"Pearson(fov_ratio, sci_res2)={p_fov_res2:.6f}\n")
        f.write(f"Spearman(fov_ratio, sci_res2)={s_fov_res2:.6f}\n\n")

        def write_quantiles(name: str, arr: np.ndarray) -> None:
            q = np.percentile(arr, [0, 1, 5, 25, 50, 75, 95, 99, 100])
            f.write(
                f"{name}: min={q[0]:.6f}, p1={q[1]:.6f}, p5={q[2]:.6f}, "
                f"q1={q[3]:.6f}, median={q[4]:.6f}, q3={q[5]:.6f}, "
                f"p95={q[6]:.6f}, p99={q[7]:.6f}, max={q[8]:.6f}\n"
            )

        f.write("=== Distribution summary ===\n")
        write_quantiles("sci_stable", sci_stable_arr)
        write_quantiles("sci_topo", sci_topo_arr)
        write_quantiles("sci_res", sci_res_arr)
        write_quantiles("sci_res2", sci_res2_arr)


# -----------------------------
# Main builder
# -----------------------------


def build_metadata(
    images_dir: Path,
    masks_dir: Path,
    fov_dir: Path,
    output_dir: Path,
    patch_size: int,
    stride: int,
    min_fov_ratio: float,
    min_vessel_pixels: int,
    area_ref: float,
    save_patches: bool,
    border_margin: int,
    wj: float,
    we: float,
    wc: float,
    lambda_topo: float,
    area_trend_mode: str,
    fov_trend_mode: str,
) -> None:
    ensure_dir(output_dir)

    patch_img_dir = output_dir / "patches" / "images"
    patch_mask_dir = output_dir / "patches" / "masks"
    patch_fov_dir = output_dir / "patches" / "fov"
    if save_patches:
        ensure_dir(patch_img_dir)
        ensure_dir(patch_mask_dir)
        ensure_dir(patch_fov_dir)

    img_map = build_file_map(images_dir)
    msk_map = build_file_map(masks_dir)
    fov_map = build_file_map(fov_dir)

    print("🔍 Matched image IDs:", list(img_map.keys())[:10])
    print("🔍 Matched mask IDs :", list(msk_map.keys())[:10])
    print("🔍 Matched FOV IDs  :", list(fov_map.keys())[:10])

    shared_ids = sorted(set(img_map.keys()) & set(msk_map.keys()) & set(fov_map.keys()))
    print("🔍 Number of shared IDs:", len(shared_ids))
    if not shared_ids:
        raise RuntimeError("No matched image / mask / FOV triplets found.")

    rows: List[dict] = []
    total_patches = 0
    skip_fov = 0
    skip_vessel = 0

    for sid in shared_ids:
        img_path = img_map[sid]
        msk_path = msk_map[sid]
        fov_path = fov_map[sid]

        img = load_rgb(img_path)
        vessel = binarize_mask(load_gray(msk_path), thr=127)
        fov = binarize_mask(load_gray(fov_path), thr=127)

        print(f"\n🖼️ Processing sample {sid}")
        print(f"  image shape: {img.shape}, vessel shape: {vessel.shape}, fov shape: {fov.shape}")
        print(f"  full-image FOV valid ratio: {np.mean(fov > 0):.4f}")
        print(f"  total vessel pixels: {np.sum(vessel > 0)}")

        h, w = vessel.shape
        positions = sliding_positions(h, w, patch_size, stride)
        print(f"  number of patch positions: {len(positions)}")

        for y, x in positions:
            total_patches += 1

            img_patch = safe_crop(img, y, x, patch_size)
            vessel_patch = safe_crop(vessel, y, x, patch_size)
            fov_patch = safe_crop(fov, y, x, patch_size)

            fov_ratio = compute_fov_ratio(fov_patch)
            if fov_ratio < min_fov_ratio:
                skip_fov += 1
                continue

            topo_stats = compute_sci_topo(
                vessel_patch=vessel_patch,
                fov_patch=fov_patch,
                eps=1e-5,
                border_margin=border_margin,
                wj=wj,
                we=we,
                wc=wc,
                lambda_topo=lambda_topo,
            )

            vessel_area = int(topo_stats["vessel_area"])
            skeleton_len = int(topo_stats["skeleton_len"])
            if vessel_area < min_vessel_pixels:
                skip_vessel += 1
                continue

            sci_raw_legacy, sci_stable_legacy = compute_sci_legacy(
                vessel_area=vessel_area,
                skeleton_len=skeleton_len,
                eps=1e-5,
                area_ref=area_ref,
            )

            patch_name = f"{sid}_y{y}_x{x}.png"

            if save_patches:
                save_rgb(patch_img_dir / patch_name, img_patch)
                save_gray(patch_mask_dir / patch_name, vessel_patch * 255)
                save_gray(patch_fov_dir / patch_name, fov_patch * 255)

            row = {
                "sample_id": sid,
                "patch_name": patch_name,
                "image_path": str(img_path),
                "mask_path": str(msk_path),
                "fov_path": str(fov_path),
                "patch_y": y,
                "patch_x": x,
                "patch_size": patch_size,
                "stride": stride,
                "fov_ratio": round(fov_ratio, 6),

                # Topology stats
                "vessel_area": vessel_area,
                "skeleton_len": skeleton_len,
                "junction_count": int(topo_stats["junction_count"]),
                "endpoint_count": int(topo_stats["endpoint_count"]),
                "component_count": int(topo_stats["component_count"]),
                "endpoint_eff": int(topo_stats["endpoint_eff"]),

                # Legacy-compatible target family
                "sci_raw": float(sci_raw_legacy),
                "sci_stable": float(sci_stable_legacy),

                # New target family
                "sci_base": float(topo_stats["sci_base"]),
                "topo_density": float(topo_stats["topo_density"]),
                "sci_topo": float(topo_stats["sci_topo"]),
            }
            rows.append(row)

    print("\n📊 Summary:")
    print(f"total patches: {total_patches}")
    print(f"filtered by FOV ratio: {skip_fov}")
    print(f"filtered by vessel area: {skip_vessel}")
    print(f"final valid patches: {len(rows)}")

    if not rows:
        raise RuntimeError("No valid patches remained. Please relax thresholds.")

    sci_raw_arr = np.array([r["sci_raw"] for r in rows], dtype=np.float32)
    sci_stable_arr = np.array([r["sci_stable"] for r in rows], dtype=np.float32)
    sci_base_arr = np.array([r["sci_base"] for r in rows], dtype=np.float32)
    topo_density_arr = np.array([r["topo_density"] for r in rows], dtype=np.float32)
    sci_topo_arr = np.array([r["sci_topo"] for r in rows], dtype=np.float32)
    area_arr = np.array([r["vessel_area"] for r in rows], dtype=np.float32)
    fov_arr = np.array([r["fov_ratio"] for r in rows], dtype=np.float32)

    # Residualize sci_topo against vessel_area -> sci_res
    area_beta = fit_area_trend(area_arr, sci_topo_arr, mode=area_trend_mode)
    sci_area_trend_arr = predict_area_trend(area_arr, area_beta, mode=area_trend_mode).astype(np.float32)
    sci_res_arr = (sci_topo_arr - sci_area_trend_arr).astype(np.float32)

    # Residualize sci_res against fov_ratio -> sci_res2
    fov_beta = fit_fov_trend(fov_arr, sci_res_arr, mode=fov_trend_mode)
    sci_fov_trend_arr = predict_fov_trend(fov_arr, fov_beta, mode=fov_trend_mode).astype(np.float32)
    sci_res2_arr = (sci_res_arr - sci_fov_trend_arr).astype(np.float32)

    # Normalized versions
    sci_raw_norm = robust_minmax(sci_raw_arr)
    sci_stable_norm = robust_minmax(sci_stable_arr)
    sci_base_norm = robust_minmax(sci_base_arr)
    topo_density_norm = robust_minmax(topo_density_arr)
    sci_topo_norm = robust_minmax(sci_topo_arr)
    sci_res_norm = robust_minmax(sci_res_arr)
    sci_res2_norm = robust_minmax(sci_res2_arr)

    for i, row in enumerate(rows):
        row["sci_raw_norm"] = float(sci_raw_norm[i])
        row["sci_stable_norm"] = float(sci_stable_norm[i])
        row["sci_base_norm"] = float(sci_base_norm[i])
        row["topo_density_norm"] = float(topo_density_norm[i])
        row["sci_topo_norm"] = float(sci_topo_norm[i])
        row["sci_area_trend"] = float(sci_area_trend_arr[i])
        row["sci_res"] = float(sci_res_arr[i])
        row["sci_res_norm"] = float(sci_res_norm[i])
        row["sci_fov_trend"] = float(sci_fov_trend_arr[i])
        row["sci_res2"] = float(sci_res2_arr[i])
        row["sci_res2_norm"] = float(sci_res2_norm[i])

    csv_path = output_dir / "patch_metadata.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Histograms / boxplots
    plot_histogram(sci_raw_arr, output_dir / "hist_sci_raw.png", "SCI raw distribution")
    plot_histogram(sci_stable_arr, output_dir / "hist_sci_stable.png", "SCI stable (legacy) distribution")
    plot_histogram(sci_base_arr, output_dir / "hist_sci_base.png", "SCI base distribution")
    plot_histogram(topo_density_arr, output_dir / "hist_topo_density.png", "Topology density distribution")
    plot_histogram(sci_topo_arr, output_dir / "hist_sci_topo.png", "SCI topo distribution")
    plot_histogram(sci_res_arr, output_dir / "hist_sci_res.png", "SCI residual distribution")
    plot_histogram(sci_res2_arr, output_dir / "hist_sci_res2.png", "SCI residual-2 distribution")

    plot_histogram(sci_raw_norm, output_dir / "hist_sci_raw_norm.png", "SCI raw normalized distribution")
    plot_histogram(sci_stable_norm, output_dir / "hist_sci_stable_norm.png", "SCI stable normalized distribution")
    plot_histogram(sci_base_norm, output_dir / "hist_sci_base_norm.png", "SCI base normalized distribution")
    plot_histogram(topo_density_norm, output_dir / "hist_topo_density_norm.png", "Topology density normalized distribution")
    plot_histogram(sci_topo_norm, output_dir / "hist_sci_topo_norm.png", "SCI topo normalized distribution")
    plot_histogram(sci_res_norm, output_dir / "hist_sci_res_norm.png", "SCI residual normalized distribution")
    plot_histogram(sci_res2_norm, output_dir / "hist_sci_res2_norm.png", "SCI residual-2 normalized distribution")

    plot_boxplot(sci_raw_arr, output_dir / "box_sci_raw.png", "SCI raw boxplot")
    plot_boxplot(sci_stable_arr, output_dir / "box_sci_stable.png", "SCI stable boxplot")
    plot_boxplot(sci_base_arr, output_dir / "box_sci_base.png", "SCI base boxplot")
    plot_boxplot(topo_density_arr, output_dir / "box_topo_density.png", "Topology density boxplot")
    plot_boxplot(sci_topo_arr, output_dir / "box_sci_topo.png", "SCI topo boxplot")
    plot_boxplot(sci_res_arr, output_dir / "box_sci_res.png", "SCI residual boxplot")
    plot_boxplot(sci_res2_arr, output_dir / "box_sci_res2.png", "SCI residual-2 boxplot")

    plot_boxplot(sci_raw_norm, output_dir / "box_sci_raw_norm.png", "SCI raw normalized boxplot")
    plot_boxplot(sci_stable_norm, output_dir / "box_sci_stable_norm.png", "SCI stable normalized boxplot")
    plot_boxplot(sci_base_norm, output_dir / "box_sci_base_norm.png", "SCI base normalized boxplot")
    plot_boxplot(topo_density_norm, output_dir / "box_topo_density_norm.png", "Topology density normalized boxplot")
    plot_boxplot(sci_topo_norm, output_dir / "box_sci_topo_norm.png", "SCI topo normalized boxplot")
    plot_boxplot(sci_res_norm, output_dir / "box_sci_res_norm.png", "SCI residual normalized boxplot")
    plot_boxplot(sci_res2_norm, output_dir / "box_sci_res2_norm.png", "SCI residual-2 normalized boxplot")

    # Scatter diagnostics
    plot_scatter(
        area_arr,
        sci_stable_arr,
        output_dir / "scatter_area_vs_sci_stable.png",
        "Vessel area vs SCI stable (legacy)",
        "Vessel area",
        "SCI stable (legacy)",
    )
    plot_scatter(
        area_arr,
        sci_topo_arr,
        output_dir / "scatter_area_vs_sci_topo.png",
        "Vessel area vs SCI topo",
        "Vessel area",
        "SCI topo",
    )
    plot_scatter(
        area_arr,
        sci_res_arr,
        output_dir / "scatter_area_vs_sci_res.png",
        "Vessel area vs SCI residual",
        "Vessel area",
        "SCI residual",
    )
    plot_scatter(
        area_arr,
        sci_res2_arr,
        output_dir / "scatter_area_vs_sci_res2.png",
        "Vessel area vs SCI residual-2",
        "Vessel area",
        "SCI residual-2",
    )
    plot_scatter(
        fov_arr,
        sci_res_arr,
        output_dir / "scatter_fov_vs_sci_res.png",
        "FOV ratio vs SCI residual",
        "FOV ratio",
        "SCI residual",
    )
    plot_scatter(
        fov_arr,
        sci_res2_arr,
        output_dir / "scatter_fov_vs_sci_res2.png",
        "FOV ratio vs SCI residual-2",
        "FOV ratio",
        "SCI residual-2",
    )

    save_ranked_examples(rows, output_dir / "top_bottom_examples.txt")
    visualize_top_bottom_patches(rows, output_dir)

    fake_args = argparse.Namespace(
        patch_size=patch_size,
        stride=stride,
        min_fov_ratio=min_fov_ratio,
        min_vessel_pixels=min_vessel_pixels,
        border_margin=border_margin,
        wj=wj,
        we=we,
        wc=wc,
        lambda_topo=lambda_topo,
        area_ref=area_ref,
    )
    save_summary_report(
        output_dir=output_dir,
        args=fake_args,
        total_patches=total_patches,
        skip_fov=skip_fov,
        skip_vessel=skip_vessel,
        rows=rows,
        area_beta=area_beta,
        area_trend_mode=area_trend_mode,
        fov_beta=fov_beta,
        fov_trend_mode=fov_trend_mode,
    )

    print(f"\n[Done] CSV saved to: {csv_path}")
    print(f"[Done] Valid patch count: {len(rows)}")
    print(f"[Done] Recommended target column for Task 2/3: sci_res2_norm")
    print("✅ Task 1 finished.")


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build patch-level SCI targets with topology enhancement, area residualization, and FOV residualization."
    )
    parser.add_argument("--images_dir", type=str, required=True, help="Directory with DRIVE training images.")
    parser.add_argument("--masks_dir", type=str, required=True, help="Directory with GT vessel masks (1st_manual).")
    parser.add_argument("--fov_dir", type=str, required=True, help="Directory with FOV masks.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")

    parser.add_argument("--patch_size", type=int, default=128, help="Patch size.")
    parser.add_argument("--stride", type=int, default=64, help="Sliding-window stride.")
    parser.add_argument("--min_fov_ratio", type=float, default=0.8, help="Minimum valid FOV ratio.")
    parser.add_argument("--min_vessel_pixels", type=int, default=120, help="Minimum vessel pixels.")
    parser.add_argument("--save_patches", action="store_true", help="Whether to save patch images / masks / FOV masks.")

    parser.add_argument("--area_ref", type=float, default=150.0, help="Legacy SCI_stable area reference A0.")

    parser.add_argument("--border_margin", type=int, default=3, help="Ignore endpoints/junctions within this many erosion steps from FOV border.")
    parser.add_argument("--wj", type=float, default=1.0, help="Weight for junction density.")
    parser.add_argument("--we", type=float, default=0.35, help="Weight for effective endpoint density.")
    parser.add_argument("--wc", type=float, default=1.5, help="Weight for component fragmentation term.")
    parser.add_argument("--lambda_topo", type=float, default=2.0, help="Overall strength of topology enhancement.")

    parser.add_argument(
        "--area_trend_mode",
        type=str,
        default="log_linear",
        choices=["linear", "log_linear"],
        help="Trend model used for removing vessel-area effect from sci_topo.",
    )
    parser.add_argument(
        "--fov_trend_mode",
        type=str,
        default="quadratic",
        choices=["linear", "quadratic"],
        help="Trend model used for removing FOV-ratio effect from sci_res.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_metadata(
        images_dir=Path(args.images_dir),
        masks_dir=Path(args.masks_dir),
        fov_dir=Path(args.fov_dir),
        output_dir=Path(args.output_dir),
        patch_size=args.patch_size,
        stride=args.stride,
        min_fov_ratio=args.min_fov_ratio,
        min_vessel_pixels=args.min_vessel_pixels,
        area_ref=args.area_ref,
        save_patches=args.save_patches,
        border_margin=args.border_margin,
        wj=args.wj,
        we=args.we,
        wc=args.wc,
        lambda_topo=args.lambda_topo,
        area_trend_mode=args.area_trend_mode,
        fov_trend_mode=args.fov_trend_mode,
    )


if __name__ == "__main__":
    main()
