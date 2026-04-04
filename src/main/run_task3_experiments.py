#!/usr/bin/env python3
"""Batch Task 3 runs: multi-seed baseline/oracle, tau sweep, shuffle ablation (same split_json)."""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from src.utils.io import load_yaml


def _deep_set(d: dict, path: str, value) -> None:
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _write_merged_config(base_path: Path, overrides: dict) -> Path:
    cfg = load_yaml(base_path) if base_path.is_file() else {}
    cfg = copy.deepcopy(cfg)
    for k, v in overrides.items():
        if "." in k:
            _deep_set(cfg, k, v)
        else:
            cfg[k] = v
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="task3_exp_")
    Path(tmp).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    import os

    os.close(fd)
    return Path(tmp)


def _run_train(cfg_file: Path, root: Path) -> None:
    cmd = [sys.executable, "-m", "src.main.train_task3", "--config", str(cfg_file), "--model", "baseline"]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--base_config", type=str, default="configs/task3.yaml")
    p.add_argument("--oracle_config", type=str, default="configs/task3_v2_oracle_weighted.yaml")
    p.add_argument(
        "--v2_1_suite",
        action="store_true",
        help="Run Task 3 v2.1 ablations (baseline + oracle_linear + oracle_smart + samplers) on same split.",
    )
    p.add_argument("--out_root", type=str, default="outputs/task3_experiments")
    p.add_argument("--seeds", type=str, default="", help="Comma-separated random seeds (baseline + oracle).")
    p.add_argument("--tau_sweep", type=str, default="", help="Comma-separated tau values for oracle runs.")
    p.add_argument("--alpha_sweep", type=str, default="", help="Comma-separated alpha values for oracle runs.")
    p.add_argument("--gamma_sweep", type=str, default="", help="Comma-separated gamma values for oracle runs.")
    p.add_argument("--run_shuffle_ablation", action="store_true")
    args = p.parse_args()

    root = Path(args.root).resolve()
    base_cfg = (root / args.base_config).resolve()
    oracle_cfg = (root / args.oracle_config).resolve()
    out_root = root / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    tmp_files: list[Path] = []

    try:
        if args.v2_1_suite:
            out_root = root / args.out_root
            if args.out_root == "outputs/task3_experiments":
                out_root = root / "outputs" / "task3_v2_1_experiments"
            out_root.mkdir(parents=True, exist_ok=True)
            suite_cfgs = [
                ("baseline", root / "configs" / "task3_v2_1_baseline.yaml"),
                ("oracle_linear", root / "configs" / "task3_v2_1_oracle_linear.yaml"),
                ("oracle_smart", root / "configs" / "task3_v2_1_oracle_smart.yaml"),
                ("oracle_linear_sampler", root / "configs" / "task3_v2_1_oracle_linear_sampler.yaml"),
                ("oracle_smart_sampler", root / "configs" / "task3_v2_1_oracle_smart_sampler.yaml"),
            ]
            for name, cfgp in suite_cfgs:
                if not cfgp.is_file():
                    raise FileNotFoundError(f"v2.1 suite missing config: {cfgp}")
                print(f"\n=== v2.1 suite: {name} ===\n")
                _run_train(cfgp.resolve(), root)
            summ_dir = out_root
            summ_dir.mkdir(parents=True, exist_ok=True)
            compare_cmd = [
                sys.executable,
                "-m",
                "src.analysis.compare_task3_v2_to_baseline",
                "--root",
                str(root),
                "--v2_1_table",
                "--v2_1_out_json",
                str(summ_dir / "comparison_summary.json"),
                "--v2_1_out_csv",
                str(summ_dir / "comparison_summary.csv"),
            ]
            print("+", " ".join(compare_cmd))
            subprocess.run(compare_cmd, cwd=root, check=True)
            print(f"\nDone v2.1 suite. Outputs under {out_root} and comparison_summary.*")
            return

        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        for s in seeds:
            odir = out_root / f"baseline_seed{s}"
            tmp = _write_merged_config(base_cfg, {"random_state": s, "output_dir_baseline": str(odir.relative_to(root))})
            tmp_files.append(tmp)
            _run_train(tmp, root)

            odir2 = out_root / f"oracle_seed{s}"
            tmp2 = _write_merged_config(
                oracle_cfg,
                {"random_state": s, "output_dir": str(odir2.relative_to(root)), "difficulty_weighting.mode": "oracle"},
            )
            tmp_files.append(tmp2)
            _run_train(tmp2, root)

        taus = [float(x.strip()) for x in args.tau_sweep.split(",") if x.strip()]
        for i, tau in enumerate(taus):
            odir = out_root / f"oracle_tau{str(tau).replace('.', '_')}"
            tmp = _write_merged_config(
                oracle_cfg,
                {
                    "output_dir": str(odir.relative_to(root)),
                    "difficulty_weighting.mode": "oracle",
                    "difficulty_weighting.tau": tau,
                },
            )
            tmp_files.append(tmp)
            _run_train(tmp, root)

        alphas = [float(x.strip()) for x in args.alpha_sweep.split(",") if x.strip()]
        for a in alphas:
            odir = out_root / f"oracle_alpha{str(a).replace('.', '_')}"
            tmp = _write_merged_config(
                oracle_cfg,
                {
                    "output_dir": str(odir.relative_to(root)),
                    "difficulty_weighting.mode": "oracle",
                    "difficulty_weighting.alpha": a,
                },
            )
            tmp_files.append(tmp)
            _run_train(tmp, root)

        gammas = [float(x.strip()) for x in args.gamma_sweep.split(",") if x.strip()]
        for g in gammas:
            odir = out_root / f"oracle_gamma{str(g).replace('.', '_')}"
            tmp = _write_merged_config(
                oracle_cfg,
                {
                    "output_dir": str(odir.relative_to(root)),
                    "difficulty_weighting.mode": "oracle",
                    "difficulty_weighting.gamma": g,
                },
            )
            tmp_files.append(tmp)
            _run_train(tmp, root)

        if args.run_shuffle_ablation:
            odir = out_root / "oracle_shuffle_ablation"
            tmp = _write_merged_config(
                oracle_cfg,
                {
                    "output_dir": str(odir.relative_to(root)),
                    "difficulty_weighting.mode": "oracle",
                    "difficulty_weighting.shuffle_scores_in_batch": True,
                },
            )
            tmp_files.append(tmp)
            _run_train(tmp, root)
    finally:
        for t in tmp_files:
            if t.is_file():
                t.unlink(missing_ok=True)

    print(f"Done. Outputs under {out_root}")


if __name__ == "__main__":
    main()
