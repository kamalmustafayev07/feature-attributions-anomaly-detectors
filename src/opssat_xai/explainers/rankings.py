"""Rankings derived from attributions (stable argsort so ties are deterministic)."""

from __future__ import annotations

import numpy as np


def rank(attr: np.ndarray) -> np.ndarray:
    """Signed ranking: feature indices sorted by attribution descending, shape ``(n, d)``."""
    return np.argsort(-np.asarray(attr, dtype=np.float64), axis=1, kind="stable")


def rank_abs(attr: np.ndarray) -> np.ndarray:
    """Absolute-value ranking (secondary; saved alongside, not used for headline results)."""
    return np.argsort(-np.abs(np.asarray(attr, dtype=np.float64)), axis=1, kind="stable")


def top_k(ranking: np.ndarray, k: int) -> np.ndarray:
    """First ``k`` entries of each row of a ranking."""
    return np.asarray(ranking)[:, :k]
