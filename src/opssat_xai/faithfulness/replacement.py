"""Replacement strategies in standardized space.

R1: x_j <- median_j of the standardized train nominals (deterministic, one row).
R2: x_j <- a value drawn uniformly from the empirical train-nominal values of feature j, independently per feature,
    ``n_draws`` rows whose scores the caller averages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opssat_xai.contracts import check_matrix


@dataclass
class ReplacementStrategy:
    """``replace(x, feature_idx, rng) -> (m, d)`` with ``m = 1`` (R1) or ``n_draws`` (R2)."""

    name: str
    reference: np.ndarray | None = None   # R1: (d,) medians
    pool: np.ndarray | None = None        # R2: (n_train_nominal, d) empirical values
    n_draws: int = 1

    @property
    def m(self) -> int:
        """Rows produced per call."""
        return 1 if self.name == "R1" else int(self.n_draws)

    def replace(self, x: np.ndarray, feature_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return ``m`` copies of ``x`` with the features in ``feature_idx`` replaced."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        idx = np.asarray(feature_idx, dtype=int).reshape(-1)
        if self.name == "R1":
            if self.reference is None:
                raise ValueError("R1 needs a reference vector")
            out = x[None, :].copy()
            out[0, idx] = self.reference[idx]
            return out
        if self.name == "R2":
            if self.pool is None:
                raise ValueError("R2 needs a pool of empirical values")
            out = np.repeat(x[None, :], self.n_draws, axis=0)
            if idx.size:
                rows = rng.integers(0, self.pool.shape[0], size=(self.n_draws, idx.size))
                out[:, idx] = self.pool[rows, idx[None, :]]
            return out
        raise ValueError(f"Unknown replacement strategy '{self.name}'")

    def fully_replaced(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """All features replaced (deletion k = d, insertion k = 0)."""
        return self.replace(x, np.arange(x.shape[-1]), rng)


def make_strategy(name: str, X_train_nominal: np.ndarray, n_draws: int = 20) -> ReplacementStrategy:
    """Build R1 (per-feature median) or R2 (empirical marginal sampler) from the standardized train nominals."""
    X = check_matrix(X_train_nominal, "X_train_nominal")
    if name == "R1":
        return ReplacementStrategy(name="R1", reference=np.median(X, axis=0))
    if name == "R2":
        return ReplacementStrategy(name="R2", pool=X.copy(), n_draws=int(n_draws))
    raise ValueError(f"Unknown replacement strategy '{name}' (expected R1 or R2)")
