"""Plain helper functions shared by the detector classes (no base-class hierarchy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from opssat_xai.contracts import assert_not_test_labels, check_matrix
from opssat_xai.detectors.thresholds import choose_threshold
from opssat_xai.io import resolve


def fit_threshold(detector: Any, X_val: np.ndarray, y_val: np.ndarray, rule: str = "max_f1") -> float:
    """Score the validation set, choose the threshold by ``rule`` and store it on ``detector``."""
    assert_not_test_labels(y_val, f"{detector.name}.set_threshold_from_validation")
    scores = detector.score(check_matrix(X_val, "X_val"))
    detector.threshold = choose_threshold(scores, np.asarray(y_val).astype(int), rule)
    return float(detector.threshold)


def predict_from_scores(scores: np.ndarray, threshold: float | None, name: str) -> np.ndarray:
    """``score >= threshold`` as a boolean array; raises if no threshold has been set."""
    if threshold is None:
        raise RuntimeError(f"{name}: call set_threshold_from_validation before predict")
    return np.asarray(scores, dtype=np.float64) >= float(threshold)


def require_fitted(model: Any, name: str) -> None:
    """Raise if ``fit`` has not been called."""
    if model is None:
        raise RuntimeError(f"{name}: call fit(X_nominal) first")


def save_detector(detector: Any, path: str | Path) -> Path:
    """Persist a scikit-learn based detector object with joblib."""
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, p)
    return p


def load_detector(path: str | Path) -> Any:
    """Load a detector persisted by :func:`save_detector`."""
    return joblib.load(resolve(path))
