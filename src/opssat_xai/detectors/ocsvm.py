"""One-Class SVM detector (RBF): score = -decision_function; nu/gamma chosen on the validation set by PR-AUC."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.svm import OneClassSVM

from opssat_xai.contracts import assert_not_test_labels, check_matrix
from opssat_xai.detectors.base import fit_threshold, predict_from_scores, require_fitted


class OneClassSVMDetector:
    """sklearn OneClassSVM behind the common detector contract."""

    name = "ocsvm"

    def __init__(
        self,
        nu: float = 0.05,
        gamma: float | str = "scale",
        kernel: str = "rbf",
        tol: float = 1e-4,
        cache_size: float = 500,
    ) -> None:
        self.params: dict[str, Any] = dict(nu=float(nu), gamma=gamma, kernel=kernel, tol=float(tol), cache_size=cache_size)
        self.model: OneClassSVM | None = None
        self.threshold: float | None = None

    def fit(self, X_nominal: np.ndarray) -> "OneClassSVMDetector":
        """Fit on nominal rows only."""
        X = check_matrix(X_nominal, "X_nominal")
        self.model = OneClassSVM(**self.params).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """``-decision_function(X)``: higher = more anomalous."""
        require_fitted(self.model, self.name)
        return -np.asarray(self.model.decision_function(check_matrix(X)), dtype=np.float64)

    def set_threshold_from_validation(self, X_val: np.ndarray, y_val: np.ndarray, rule: str = "max_f1") -> float:
        """Choose and store the threshold on the validation set."""
        return fit_threshold(self, X_val, y_val, rule)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``score(X) >= threshold``."""
        return predict_from_scores(self.score(X), self.threshold, self.name)


def select_hyperparameters(
    X_train_nominal: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    grid: dict[str, list[Any]],
    kernel: str = "rbf",
    tol: float = 1e-4,
    cache_size: float = 500,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Grid search nu x gamma by validation PR-AUC (average precision). Returns ``(best_params, grid_table)``.

    Ties keep the first grid entry (grid order is nu-major, gamma-minor as configured).
    """
    assert_not_test_labels(y_val, "ocsvm.select_hyperparameters")
    y_val = np.asarray(y_val).astype(int)
    rows: list[dict[str, Any]] = []
    for nu in grid["nu"]:
        for gamma in grid["gamma"]:
            det = OneClassSVMDetector(nu=nu, gamma=gamma, kernel=kernel, tol=tol, cache_size=cache_size).fit(X_train_nominal)
            pr_auc = float(average_precision_score(y_val, det.score(X_val)))
            rows.append({"nu": float(nu), "gamma": gamma, "val_pr_auc": pr_auc})
    table = pd.DataFrame(rows)
    best_i = int(np.argmax(table["val_pr_auc"].to_numpy()))
    best = {"nu": float(table.loc[best_i, "nu"]), "gamma": table.loc[best_i, "gamma"]}
    if isinstance(best["gamma"], (float, np.floating)):
        best["gamma"] = float(best["gamma"])
    return best, table
