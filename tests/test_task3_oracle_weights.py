"""Unit tests: oracle_linear vs oracle_smart sample weights must differ when FOV gate applies."""
from __future__ import annotations

import torch

from src.training.task3_engine import (
    _effective_oracle_score_gamma,
    oracle_sample_weights,
    smart_sample_weights,
)


def test_effective_gamma_modes() -> None:
    assert _effective_oracle_score_gamma("oracle_true_linear", 99.0) == 1.0
    assert _effective_oracle_score_gamma("oracle_quadratic", 0.5) == 2.0
    assert _effective_oracle_score_gamma("oracle_linear", 1.5) == 1.5


def test_smart_differs_from_linear_when_low_fov_high_sci() -> None:
    tau, alpha, gamma = 0.60, 0.80, 2.0
    fov_min, min_w, max_w = 0.80, 1.0, 1.80
    scores = torch.tensor([0.9, 0.9], dtype=torch.float32)
    fov = torch.tensor([0.5, 1.0], dtype=torch.float32)
    w_lin = oracle_sample_weights(scores, tau, alpha, gamma, min_w, max_w, normalize_in_batch=False)
    w_smt = smart_sample_weights(scores, fov, tau, alpha, gamma, fov_min, max_w, normalize_in_batch=False)
    assert not torch.allclose(w_lin, w_smt, atol=1e-5)


def test_smart_matches_linear_all_high_fov_user_toy() -> None:
    tau, alpha, gamma = 0.60, 0.80, 2.0
    fov_min, min_w, max_w = 0.80, 1.0, 1.80
    scores = torch.tensor([0.2, 0.6, 0.8, 1.0], dtype=torch.float32)
    fov = torch.ones_like(scores)
    w_lin = oracle_sample_weights(scores, tau, alpha, gamma, min_w, max_w, normalize_in_batch=False)
    w_smt = smart_sample_weights(scores, fov, tau, alpha, gamma, fov_min, max_w, normalize_in_batch=False)
    assert torch.allclose(w_lin, w_smt, atol=1e-6)


def test_with_batch_normalize_mixed_fov_still_differs() -> None:
    tau, alpha, gamma = 0.60, 0.80, 2.0
    fov_min, min_w, max_w = 0.80, 1.0, 1.80
    scores = torch.tensor([0.9, 0.9], dtype=torch.float32)
    fov = torch.tensor([0.5, 1.0], dtype=torch.float32)
    w_lin = oracle_sample_weights(scores, tau, alpha, gamma, min_w, max_w, normalize_in_batch=True)
    w_smt = smart_sample_weights(scores, fov, tau, alpha, gamma, fov_min, max_w, normalize_in_batch=True)
    assert not torch.allclose(w_lin, w_smt, atol=1e-5)
