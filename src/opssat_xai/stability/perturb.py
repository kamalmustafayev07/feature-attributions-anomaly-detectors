"""Noisy copies x + N(0, sigma^2 I) in standardized space and the class-flip filter."""

from __future__ import annotations

from typing import Any

import numpy as np

from opssat_xai.contracts import check_matrix


def perturb_copies(x: np.ndarray, n_copies: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """``n_copies`` rows ``x + eps`` with ``eps ~ N(0, sigma^2)`` independently per feature."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[None, :] + rng.normal(0.0, float(sigma), size=(int(n_copies), x.shape[0]))


def retained_mask(detector: Any, x_clean: np.ndarray, copies: np.ndarray) -> np.ndarray:
    """True for copies whose ``predict`` equals the clean prediction (copies that flip are not explained)."""
    clean = bool(detector.predict(np.asarray(x_clean, dtype=np.float64).reshape(1, -1))[0])
    return detector.predict(check_matrix(copies)) == clean
