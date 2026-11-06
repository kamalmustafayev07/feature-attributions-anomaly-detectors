"""Threshold selection on the VALIDATION set only. Test labels are rejected by the tag in contracts.py."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

from opssat_xai.contracts import assert_not_test_labels


def max_f1_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Threshold on ``score >= t`` that maximizes F1 on ``(scores, y)``; ties go to the lowest threshold.

    Returns ``(threshold, f1)``.
    """
    assert_not_test_labels(y, "max_f1_threshold")
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("max_f1_threshold needs both classes in the validation labels")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]  # drop the artificial (1, 0) end point
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    best = int(np.argmax(f1))  # thresholds are increasing; first max = lowest threshold = highest recall
    return float(thresholds[best]), float(f1[best])


def fpr_threshold(scores_nominal: np.ndarray, fpr: float) -> float:
    """Threshold such that at most ``fpr`` of the given nominal scores satisfy ``score >= t``."""
    scores_nominal = np.asarray(scores_nominal, dtype=np.float64)
    if not 0.0 < fpr < 1.0:
        raise ValueError("fpr must be in (0, 1)")
    return float(np.quantile(scores_nominal, 1.0 - fpr, method="higher"))


def choose_threshold(scores: np.ndarray, y: np.ndarray, rule: str = "max_f1") -> float:
    """Dispatch on ``rule``; only ``max_f1`` is a primary rule (``fpr`` points are computed separately)."""
    assert_not_test_labels(y, f"choose_threshold({rule})")
    if rule == "max_f1":
        return max_f1_threshold(scores, y)[0]
    raise ValueError(f"Unknown threshold rule '{rule}' (supported: max_f1)")
