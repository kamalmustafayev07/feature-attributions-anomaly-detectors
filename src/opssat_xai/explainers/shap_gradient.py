"""GradientSHAP (expected gradients) on the autoencoder's scalar MSE score module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr

from opssat_xai.contracts import check_attribution, check_matrix
from opssat_xai.seeds import int_seed, rng


def select_background(X_train_nominal: np.ndarray, size: int = 100, seed: int = 0) -> np.ndarray:
    """``size`` train-nominal rows drawn without replacement (seeded)."""
    X = check_matrix(X_train_nominal)
    size = int(min(size, X.shape[0]))
    idx = np.sort(rng(seed, "gradient_background").choice(X.shape[0], size=size, replace=False))
    return X[idx]


class GradientShapExplainer:
    """``shap.GradientExplainer(score_module, background)``; explains the detector's own score."""

    name = "shap"
    method = "shap_gradient"

    def __init__(self, score_module: torch.nn.Module, background: np.ndarray, nsamples: int = 200, seed: int = 0) -> None:
        self.module = score_module.eval()
        self.background = torch.tensor(check_matrix(background), dtype=torch.float32)
        self.nsamples = int(nsamples)
        self.seed = int(seed)
        # TODO(verify): GradientExplainer constructor signature for the installed shap/torch versions
        self.explainer = shap.GradientExplainer(self.module, self.background)

    def attribute(self, X: np.ndarray, row_ids: np.ndarray | None = None) -> np.ndarray:
        """Expected-gradients attributions, seeded (torch + numpy) before the call."""
        X = check_matrix(X)
        s = int_seed(self.seed, "gradient")
        torch.manual_seed(s)
        np.random.seed(s)
        # TODO(verify): shap_values(X, nsamples=..., rseed=...) signature and return type (array vs list)
        sv = self.explainer.shap_values(torch.tensor(X, dtype=torch.float32), nsamples=self.nsamples, rseed=s)
        arr = sv[0] if isinstance(sv, list) else sv
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return check_attribution(arr.reshape(X.shape), X)


def compare_attributions(attr_a: np.ndarray, attr_b: np.ndarray, row_ids: np.ndarray | None = None) -> pd.DataFrame:
    """Per-row Spearman correlation between two attribution matrices (e.g. gradient vs kernel SHAP)."""
    a = np.asarray(attr_a, dtype=np.float64)
    b = np.asarray(attr_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("attribution matrices must have the same shape")
    ids = np.arange(a.shape[0]) if row_ids is None else np.asarray(row_ids)
    rows = []
    for i in range(a.shape[0]):
        rho = spearmanr(a[i], b[i]).statistic
        rows.append({"row_id": ids[i], "spearman": float(rho) if np.isfinite(rho) else np.nan,
                     "sum_a": float(a[i].sum()), "sum_b": float(b[i].sum())})
    return pd.DataFrame(rows)
