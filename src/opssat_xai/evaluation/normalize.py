"""Rank (empirical CDF) and z normalization of detector scores against the validation-nominal reference scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ScoreNormalizer:
    """``rank(s) = |{r_i <= s}| / |R|`` in [0, 1] and ``z(s) = (s - mean_R) / std_R``."""

    reference_sorted: np.ndarray
    mean: float
    std: float

    @classmethod
    def fit(cls, reference_scores: np.ndarray) -> "ScoreNormalizer":
        """Fit on the reference scores (validation nominals)."""
        r = np.sort(np.asarray(reference_scores, dtype=np.float64).reshape(-1))
        if r.size < 2:
            raise ValueError("need at least two reference scores")
        std = float(r.std(ddof=1))
        if std == 0.0:
            raise ValueError("reference scores are constant; z-normalization is undefined")
        return cls(reference_sorted=r, mean=float(r.mean()), std=std)

    def rank(self, scores: np.ndarray) -> np.ndarray:
        """Empirical CDF of the reference evaluated at ``scores`` (1.0 = above every reference score)."""
        s = np.asarray(scores, dtype=np.float64)
        return np.searchsorted(self.reference_sorted, s, side="right") / self.reference_sorted.size

    def z(self, scores: np.ndarray) -> np.ndarray:
        """Standardized score against the reference mean and SD."""
        return (np.asarray(scores, dtype=np.float64) - self.mean) / self.std

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {"reference_sorted": self.reference_sorted.tolist(), "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoreNormalizer":
        """Inverse of :meth:`to_dict`."""
        return cls(reference_sorted=np.asarray(d["reference_sorted"], dtype=np.float64), mean=float(d["mean"]), std=float(d["std"]))


def rank_saturation_fraction(normalizer: ScoreNormalizer, scores: np.ndarray) -> float:
    """Fraction of ``scores`` whose rank-normalized value is exactly 1.0 (saturation check, decided once at C3)."""
    return float(np.mean(normalizer.rank(scores) >= 1.0))
