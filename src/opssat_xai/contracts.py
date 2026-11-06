"""Interface contracts shared by both halves of the project (change only by agreement).

Array conventions
-----------------
* Feature matrix ``X``: ``np.ndarray`` of shape ``(n, d)``, ``float64``, already standardized by the
  scaler fitted on the train nominals. ``d`` is 18 for the main run and 15 for ablation A. Column order
  is ``feature_names`` as exposed by the loader (configs/data.yaml ``features.names`` minus ``features.exclude``).
* Detector scores: shape ``(n,)``, HIGHER = MORE ANOMALOUS.
* Attributions: shape ``(n, d)``, POSITIVE = pushes the anomaly score up.
* Rankings: ``np.argsort(-attr, axis=1)`` (signed, descending); ``ranking_abs`` uses ``|attr|``.
* Replacement: ``replace(x_row, feature_idx, rng) -> (m, d)`` with ``m = 1`` for R1 and ``m = n_draws`` for R2;
  the caller averages the ``m`` scores.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

# Long-format faithfulness record (one CSV row per perturbation step). AUDC is computed from these rows.
FAITHFULNESS_COLUMNS: tuple[str, ...] = (
    "anomaly_id", "detector", "explainer", "repeat", "strategy", "direction", "k",
    "score_raw", "score_rank", "score_z",
)

# Stability record (one CSV row per noisy copy per explainer).
STABILITY_COLUMNS: tuple[str, ...] = (
    "anomaly_id", "detector", "explainer", "copy_id", "retained", "jaccard3", "jaccard5", "spearman", "rel_l2",
)

STRATEGIES: tuple[str, ...] = ("R1", "R2")
DIRECTIONS: tuple[str, ...] = ("deletion", "insertion")


@runtime_checkable
class Detector(Protocol):
    """Anomaly detector contract. ``fit`` must only ever receive nominal rows."""

    name: str
    threshold: float | None

    def fit(self, X_nominal: np.ndarray) -> "Detector": ...

    def score(self, X: np.ndarray) -> np.ndarray: ...

    def set_threshold_from_validation(self, X_val: np.ndarray, y_val: np.ndarray, rule: str = "max_f1") -> float: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Explainer(Protocol):
    """Explainer contract: ``attribute(X) -> (n, d)`` with positive = pushes the anomaly score up."""

    name: str

    def attribute(self, X: np.ndarray, row_ids: np.ndarray | None = None) -> np.ndarray: ...


class TestLabels(np.ndarray):
    """ndarray subclass that tags the TEST labels so no selection/threshold function can consume them.

    Scripts wrap ``y_test`` with :func:`mark_test_labels`; every threshold or hyper-parameter selection
    function calls :func:`assert_not_test_labels` on the labels it receives. Slicing/views keep the tag.
    """

    def __new__(cls, input_array: np.ndarray) -> "TestLabels":
        obj = np.asarray(input_array).view(cls)
        obj.is_test_labels = True
        return obj

    def __array_finalize__(self, obj: object) -> None:
        self.is_test_labels = getattr(obj, "is_test_labels", True)


def mark_test_labels(y_test: np.ndarray) -> TestLabels:
    """Tag the test labels so that they can never be passed to a selection function."""
    return TestLabels(np.asarray(y_test))


def assert_not_test_labels(y: np.ndarray, where: str) -> None:
    """Raise if ``y`` carries the test-label tag (guards against test-set thresholding / tuning)."""
    if isinstance(y, TestLabels) or getattr(y, "is_test_labels", False):
        raise AssertionError(
            f"Test labels were passed to '{where}'. Thresholds and hyper-parameters are chosen on the "
            "validation set only (no test-set thresholding)."
        )


def check_matrix(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Return ``X`` as a 2-D float64 array, raising a clear error otherwise."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, d); got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains NaN or inf")
    return X


def check_attribution(attr: np.ndarray, X: np.ndarray, name: str = "attribution") -> np.ndarray:
    """Validate an attribution array against the explained matrix (shape ``(n, d)``, finite)."""
    attr = np.asarray(attr, dtype=np.float64)
    if attr.shape != X.shape:
        raise ValueError(f"{name} shape {attr.shape} does not match X shape {X.shape}")
    if not np.all(np.isfinite(attr)):
        raise ValueError(f"{name} contains NaN or inf")
    return attr
