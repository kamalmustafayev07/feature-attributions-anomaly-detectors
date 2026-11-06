"""Spearman correlation on train nominals and grouping at |rho| >= threshold (connected components)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from opssat_xai.contracts import check_matrix


def spearman_matrix(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Pairwise Spearman correlation matrix as a labelled DataFrame."""
    X = check_matrix(X)
    rho = spearmanr(X).statistic
    rho = np.atleast_2d(np.asarray(rho, dtype=np.float64))
    return pd.DataFrame(rho, index=feature_names, columns=feature_names)


def correlated_groups(corr: pd.DataFrame, threshold: float = 0.8) -> list[list[str]]:
    """Connected components of the graph with an edge where ``|rho| >= threshold`` (single linkage).

    Singletons are included so every feature belongs to exactly one group. Groups are sorted by their first member
    in the original feature order.
    """
    names = list(corr.columns)
    parent = list(range(len(names)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    values = corr.to_numpy(dtype=np.float64)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.isfinite(values[i, j]) and abs(values[i, j]) >= threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i, n in enumerate(names):
        groups.setdefault(find(i), []).append(n)
    return sorted(groups.values(), key=lambda g: names.index(g[0]))


def group_attributions(attr: np.ndarray, groups: list[list[str]], feature_names: list[str]) -> np.ndarray:
    """Sum attributions within each group -> ``(n, n_groups)``."""
    attr = np.asarray(attr, dtype=np.float64)
    index = {n: i for i, n in enumerate(feature_names)}
    out = np.zeros((attr.shape[0], len(groups)))
    for g_i, group in enumerate(groups):
        out[:, g_i] = attr[:, [index[n] for n in group]].sum(axis=1)
    return out


def top1_agreement(attr_a: np.ndarray, attr_b: np.ndarray) -> float:
    """Fraction of rows whose top-1 (signed) feature agrees between two attribution matrices."""
    a = np.argmax(np.asarray(attr_a), axis=1)
    b = np.argmax(np.asarray(attr_b), axis=1)
    return float(np.mean(a == b))
