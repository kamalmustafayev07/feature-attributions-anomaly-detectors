"""Isolation Forest detector: score = -score_samples (higher = more anomalous)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from opssat_xai.contracts import check_matrix
from opssat_xai.detectors.base import fit_threshold, predict_from_scores, require_fitted


class IsolationForestDetector:
    """sklearn IsolationForest behind the common detector contract."""

    name = "iforest"

    def __init__(
        self,
        n_estimators: int = 300,
        max_samples: int | str = 256,
        contamination: float | str = "auto",
        max_features: float = 1.0,
        random_state: int = 0,
        n_jobs: int = 1,
    ) -> None:
        self.params = dict(
            n_estimators=int(n_estimators), max_samples=max_samples, contamination=contamination,
            max_features=float(max_features), random_state=int(random_state), n_jobs=int(n_jobs),
        )
        self.model: IsolationForest | None = None
        self.threshold: float | None = None

    def fit(self, X_nominal: np.ndarray) -> "IsolationForestDetector":
        """Fit on nominal rows only."""
        X = check_matrix(X_nominal, "X_nominal")
        self.model = IsolationForest(**self.params).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """``-score_samples(X)``: higher = more anomalous."""
        require_fitted(self.model, self.name)
        return -np.asarray(self.model.score_samples(check_matrix(X)), dtype=np.float64)

    def set_threshold_from_validation(self, X_val: np.ndarray, y_val: np.ndarray, rule: str = "max_f1") -> float:
        """Choose and store the threshold on the validation set."""
        return fit_threshold(self, X_val, y_val, rule)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``score(X) >= threshold``."""
        return predict_from_scores(self.score(X), self.threshold, self.name)
