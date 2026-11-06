"""Percentile bootstrap (B = 2000 by default) for means, seeded through SeedSequence."""

from __future__ import annotations

import numpy as np

from opssat_xai.seeds import rng


def bootstrap_mean(x: np.ndarray, B: int = 2000, seed: int = 0, alpha: float = 0.05, key: str = "mean") -> dict[str, float]:
    """Mean of ``x`` with a percentile bootstrap CI over its elements (NaNs are dropped first)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    n = x.shape[0]
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0, "B": int(B), "sd": float("nan")}
    generator = rng(seed, "bootstrap", key)
    idx = generator.integers(0, n, size=(int(B), n))
    means = x[idx].mean(axis=1)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "mean": float(x.mean()),
        "ci_low": float(np.percentile(means, lo)),
        "ci_high": float(np.percentile(means, hi)),
        "n": int(n),
        "B": int(B),
        "sd": float(x.std(ddof=1)) if n > 1 else float("nan"),
    }


def bootstrap_paired_mean_diff(a: np.ndarray, b: np.ndarray, B: int = 2000, seed: int = 0, alpha: float = 0.05, key: str = "diff") -> dict[str, float]:
    """Paired bootstrap on ``a - b`` (rows resampled together)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same length")
    return bootstrap_mean(a - b, B=B, seed=seed, alpha=alpha, key=key)
