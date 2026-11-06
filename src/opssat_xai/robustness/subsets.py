"""Disagreement subset (SHAP vs DEV top-1 differs or top-3 Jaccard < 0.34) and false-positive subset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from opssat_xai.explainers.rankings import rank
from opssat_xai.stability.metrics import jaccard_topk


def disagreement_mask(attr_a: np.ndarray, attr_b: np.ndarray, top_k: int = 3, jaccard_threshold: float = 0.34) -> np.ndarray:
    """Rows where the top-1 features differ OR the top-k Jaccard is below the threshold."""
    ra, rb = rank(attr_a), rank(attr_b)
    top1_differs = ra[:, 0] != rb[:, 0]
    jac = np.array([jaccard_topk(ra[i], rb[i], top_k) for i in range(ra.shape[0])])
    return top1_differs | (jac < jaccard_threshold)


def false_positive_indices(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Positions of nominal rows predicted anomalous."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(bool)
    return np.flatnonzero((y == 0) & pred)


def top_features_table(attr: np.ndarray, feature_names: list[str], row_ids: np.ndarray, k: int = 3, prefix: str = "top") -> pd.DataFrame:
    """Top-k signed features per row as a table with columns ``{prefix}1..{prefix}k``."""
    r = rank(attr)
    data = {"row_id": np.asarray(row_ids)}
    for j in range(k):
        data[f"{prefix}{j + 1}"] = [feature_names[i] for i in r[:, j]]
    return pd.DataFrame(data)
