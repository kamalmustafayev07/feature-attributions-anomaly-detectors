"""TreeSHAP for the Isolation Forest. TreeSHAP explains the mean path length h(x); shorter = more anomalous,
so the attribution toward anomaly is ``-phi``."""

from __future__ import annotations

from typing import Any

import numpy as np
import shap
from sklearn.ensemble._iforest import _average_path_length  # TODO(verify): private sklearn import; check it still exists in 1.9.0

from opssat_xai.contracts import check_attribution, check_matrix


def _shap_values_2d(explainer: shap.TreeExplainer, X: np.ndarray) -> np.ndarray:
    phi = np.asarray(explainer.shap_values(X))
    if phi.ndim == 3:  # TODO(verify): shap version returning (n, d, n_outputs) for single-output forests
        phi = phi[:, :, 0]
    return phi.astype(np.float64)


class TreeShapExplainer:
    """``shap.TreeExplainer(iforest.model)`` with no background (path-dependent, exact for the trees)."""

    name = "shap"
    method = "shap_tree"

    def __init__(self, detector: Any) -> None:
        if getattr(detector, "model", None) is None:
            raise RuntimeError("TreeShapExplainer needs a fitted IsolationForestDetector")
        self.detector = detector
        self.explainer = shap.TreeExplainer(detector.model)

    def attribute(self, X: np.ndarray, row_ids: np.ndarray | None = None) -> np.ndarray:
        """``-shap_values(X)``: positive = shorter path = pushes the anomaly score up."""
        X = check_matrix(X)
        return check_attribution(-_shap_values_2d(self.explainer, X), X)


def check_additivity(detector: Any, X: np.ndarray, tol: float = 1e-6) -> dict[str, Any]:
    """Verify ``score_samples(X) == -2**(-(phi.sum(1) + expected_value) / c(max_samples_))`` on ``X``.

    Mirrors shap's own IsolationForest test. Returns the max absolute difference and a pass flag.
    """
    X = check_matrix(X)
    model = detector.model
    explainer = shap.TreeExplainer(model)
    phi = _shap_values_2d(explainer, X)
    expected = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)[0]
    c = float(_average_path_length(np.array([model.max_samples_]))[0])
    predicted = -(2.0 ** (-(phi.sum(axis=1) + expected) / c))
    actual = np.asarray(model.score_samples(X), dtype=np.float64)
    max_abs_diff = float(np.max(np.abs(actual - predicted)))
    return {"n_rows": int(X.shape[0]), "max_abs_diff": max_abs_diff, "tol": float(tol), "passed": bool(max_abs_diff <= tol)}
