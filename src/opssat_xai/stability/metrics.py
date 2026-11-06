"""Per-copy stability metrics against the clean attribution: Jaccard@3/@5 (signed rankings), Spearman, relative L2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from opssat_xai.explainers.rankings import rank


def jaccard_topk(ranking_a: np.ndarray, ranking_b: np.ndarray, k: int) -> float:
    """Jaccard overlap of the top-k feature sets of two 1-D rankings."""
    a = set(np.asarray(ranking_a).reshape(-1)[:k].tolist())
    b = set(np.asarray(ranking_b).reshape(-1)[:k].tolist())
    union = a | b
    return float(len(a & b) / len(union)) if union else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two attribution vectors (NaN if either is constant)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def rel_l2(a_clean: np.ndarray, a_copy: np.ndarray) -> float:
    """``||a' - a||_2 / ||a||_2`` (NaN if the clean attribution is all zero)."""
    a = np.asarray(a_clean, dtype=np.float64).reshape(-1)
    b = np.asarray(a_copy, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(a)
    return float(np.linalg.norm(b - a) / norm) if norm > 0 else float("nan")


def stability_metrics(attr_clean: np.ndarray, attr_copy: np.ndarray, top_ks: tuple[int, ...] = (3, 5)) -> dict[str, float]:
    """All per-copy metrics for one (clean, copy) pair of attribution vectors."""
    a = np.asarray(attr_clean, dtype=np.float64).reshape(1, -1)
    b = np.asarray(attr_copy, dtype=np.float64).reshape(1, -1)
    ra, rb = rank(a)[0], rank(b)[0]
    out: dict[str, float] = {f"jaccard{k}": jaccard_topk(ra, rb, k) for k in top_ks}
    out["spearman"] = spearman(a, b)
    out["rel_l2"] = rel_l2(a, b)
    return out


def per_anomaly_means(stability_df: pd.DataFrame, metrics: tuple[str, ...] = ("jaccard3", "jaccard5", "spearman", "rel_l2")) -> pd.DataFrame:
    """Mean of each metric over the RETAINED copies of each (anomaly, detector, explainer)."""
    kept = stability_df[stability_df["retained"].astype(bool)]
    return kept.groupby(["detector", "explainer", "anomaly_id"], as_index=False)[list(metrics)].mean()


def retention_summary(stability_df: pd.DataFrame) -> pd.DataFrame:
    """Retention rate per detector (copies not discarded by the class-flip filter)."""
    one = stability_df.drop_duplicates(["detector", "anomaly_id", "copy_id"])
    g = one.groupby("detector")["retained"]
    return pd.DataFrame({"detector": g.size().index, "n_copies": g.size().to_numpy(),
                         "n_retained": g.sum().to_numpy(), "retention_rate": g.mean().to_numpy()})
