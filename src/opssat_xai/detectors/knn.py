"""kNN distance detector (optional fallback): score = mean Euclidean distance to the k nearest train nominals."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from opssat_xai.contracts import check_matrix
from opssat_xai.detectors.base import fit_threshold, predict_from_scores, require_fitted


class KNNDetector:
    """NearestNeighbors(k) on the train nominals; higher mean distance = more anomalous."""

    name = "knn"

    def __init__(self, k: int = 5) -> None:
        self.k = int(k)
        self.model: NearestNeighbors | None = None
        self.threshold: float | None = None

    def fit(self, X_nominal: np.ndarray) -> "KNNDetector":
        """Index the nominal rows."""
        X = check_matrix(X_nominal, "X_nominal")
        self.model = NearestNeighbors(n_neighbors=self.k).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Mean distance to the k nearest indexed (train-nominal) rows. Note: a train row scores its own distance 0."""
        require_fitted(self.model, self.name)
        dist, _ = self.model.kneighbors(check_matrix(X), n_neighbors=self.k)
        return np.asarray(dist.mean(axis=1), dtype=np.float64)

    def set_threshold_from_validation(self, X_val: np.ndarray, y_val: np.ndarray, rule: str = "max_f1") -> float:
        """Choose and store the threshold on the validation set."""
        return fit_threshold(self, X_val, y_val, rule)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``score(X) >= threshold``."""
        return predict_from_scores(self.score(X), self.threshold, self.name)
