#!/usr/bin/env python3
"""
Print oracle_linear vs oracle_smart sample weights on toy batches.

Group A: all fov=1 — with v2.1 default hyperparams, smart matches linear (no FOV gate fires on low-FOV high-SCI).
Group B: mixed fov — linear upweights both high-s; smart only upweights the high-FOV sample.

Run: python -m src.analysis.print_oracle_weight_toys
"""
from __future__ import annotations

import torch

from src.training.task3_engine import oracle_sample_weights, smart_sample_weights


def _h(scores: torch.Tensor, tau: float) -> torch.Tensor:
    denom = max(1.0 - tau, 1e-8)
    return ((scores - tau) / denom).clamp(0.0, 1.0)


def _print_group(
    name: str,
    s: list[float],
    fov: list[float],
    *,
    tau: float,
    alpha: float,
    gamma: float,
    fov_min: float,
    min_weight: float,
    max_weight: float,
    expect_match: str,
) -> None:
    scores = torch.tensor(s, dtype=torch.float32)
    fov_t = torch.tensor(fov, dtype=torch.float32)
    h = _h(scores, tau)
    w_lin_n = oracle_sample_weights(
        scores, tau, alpha, gamma, min_weight, max_weight, normalize_in_batch=False
    )
    w_smt_n = smart_sample_weights(
        scores, fov_t, tau, alpha, gamma, fov_min, max_weight, normalize_in_batch=False
    )
    w_lin_y = oracle_sample_weights(
        scores, tau, alpha, gamma, min_weight, max_weight, normalize_in_batch=True
    )
    w_smt_y = smart_sample_weights(
        scores, fov_t, tau, alpha, gamma, fov_min, max_weight, normalize_in_batch=True
    )

    print(f"\n=== {name} ===")
    print(f"expectation: {expect_match}")
    print(f"tau={tau} alpha={alpha} gamma={gamma} fov_min={fov_min} min_w={min_weight} max_w={max_weight}")
    print("idx  s      fov    h      w_lin(norm=0) w_smt(norm=0) | w_lin(norm=1) w_smt(norm=1)")
    for i in range(len(s)):
        print(
            f"{i:2d}  {s[i]:.2f}   {fov[i]:.2f}   {h[i].item():.4f}   "
            f"{w_lin_n[i].item():.6f}      {w_smt_n[i].item():.6f}      | "
            f"{w_lin_y[i].item():.6f}      {w_smt_y[i].item():.6f}"
        )
    match_n = torch.allclose(w_lin_n, w_smt_n, atol=1e-6)
    match_y = torch.allclose(w_lin_y, w_smt_y, atol=1e-6)
    print(f"allclose (no batch norm): {match_n}")
    print(f"allclose (batch norm):    {match_y}")


def main() -> None:
    tau, alpha, gamma = 0.60, 0.80, 2.0
    fov_min = 0.80
    min_w, max_w = 1.0, 1.80

    _print_group(
        "Group A (user toy: all high FOV)",
        s=[0.2, 0.6, 0.8, 1.0],
        fov=[1.0, 1.0, 1.0, 1.0],
        tau=tau,
        alpha=alpha,
        gamma=gamma,
        fov_min=fov_min,
        min_weight=min_w,
        max_weight=max_w,
        expect_match="smart == linear elementwise (FOV gate never forces w=1 on a sample that linear would upweight).",
    )
    _print_group(
        "Group B (low FOV + high SCI on sample 0)",
        s=[0.9, 0.9],
        fov=[0.5, 1.0],
        tau=tau,
        alpha=alpha,
        gamma=gamma,
        fov_min=fov_min,
        min_weight=min_w,
        max_weight=max_w,
        expect_match="smart != linear: first sample gated to w=1, second upweighted.",
    )


if __name__ == "__main__":
    main()
