"""KernelSHAP on ``det.score`` (so positive = pushes the anomaly score up) with a k-means background."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import shap

from opssat_xai.contracts import check_attribution, check_matrix
from opssat_xai.seeds import int_seed


def make_background(X_train_nominal: np.ndarray, size: int = 100) -> Any:
    """``shap.kmeans`` summary of the standardized train nominals (weighted centroids)."""
    X = check_matrix(X_train_nominal)
    # TODO(verify): shap.kmeans uses sklearn KMeans with a fixed random_state internally in the installed version
    return shap.kmeans(X, int(min(size, X.shape[0])))


class KernelShapExplainer:
    """``shap.KernelExplainer(score_fn, background)``; one call per row, numpy seeded per row."""

    name = "shap"
    method = "shap_kernel"

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], np.ndarray],
        background: Any,
        n_features: int,
        nsamples: int = 2048,
        l1_reg: str = "num_features",
        seed: int = 0,
    ) -> None:
        self.score_fn = score_fn
        self.background = background
        self.n_features = int(n_features)
        self.nsamples = int(nsamples)
        self.l1_reg = f"num_features({self.n_features})" if l1_reg == "num_features" else l1_reg
        self.seed = int(seed)
        self.explainer = shap.KernelExplainer(score_fn, background)

    def attribute(self, X: np.ndarray, row_ids: np.ndarray | None = None) -> np.ndarray:
        """Explain each row with ``np.random.seed(int_seed(seed, 'kernel', row_id))`` set just before the call."""
        X = check_matrix(X)
        n, d = X.shape
        ids = np.arange(n) if row_ids is None else np.asarray(row_ids)
        out = np.empty((n, d), dtype=np.float64)
        for i in range(n):
            np.random.seed(int_seed(self.seed, "kernel", int(ids[i])))
            sv = self.explainer.shap_values(X[i:i + 1], nsamples=self.nsamples, l1_reg=self.l1_reg, silent=True)
            # TODO(verify): return shape of KernelExplainer.shap_values for a scalar-output function in the installed shap
            out[i] = np.asarray(sv, dtype=np.float64).reshape(-1)[:d]
        return check_attribution(out, X)

    def expected_value(self) -> float:
        """Mean score over the background (the SHAP reference)."""
        return float(np.asarray(self.explainer.expected_value).reshape(-1)[0])
