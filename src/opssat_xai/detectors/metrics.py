"""Detection metrics: PR-AUC (average precision), ROC-AUC, thresholded metrics, row bootstrap CIs, chance rule."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from opssat_xai.seeds import rng


def auc_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """PR-AUC via ``average_precision_score`` and ROC-AUC from raw scores."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    return {"pr_auc": float(average_precision_score(y, scores)), "roc_auc": float(roc_auc_score(y, scores))}


def threshold_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    """Precision, recall, F1, MCC, accuracy and TP/FP/FN/TN at a fixed prediction."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def bootstrap_auc_ci(
    y: np.ndarray, scores: np.ndarray, B: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Percentile bootstrap over rows for PR-AUC and ROC-AUC. Resamples lacking a class are redrawn."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(y)
    generator = rng(seed, "bootstrap_auc")
    pr = np.empty(B)
    roc = np.empty(B)
    for b in range(B):
        while True:
            idx = generator.integers(0, n, size=n)
            yb = y[idx]
            if 0 < yb.sum() < n:
                break
        pr[b] = average_precision_score(yb, scores[idx])
        roc[b] = roc_auc_score(yb, scores[idx])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    point = auc_metrics(y, scores)
    return {
        "pr_auc": point["pr_auc"],
        "pr_auc_ci_low": float(np.percentile(pr, lo)), "pr_auc_ci_high": float(np.percentile(pr, hi)),
        "roc_auc": point["roc_auc"],
        "roc_auc_ci_low": float(np.percentile(roc, lo)), "roc_auc_ci_high": float(np.percentile(roc, hi)),
        "bootstrap_B": int(B),
    }


def chance_pr_auc(y: np.ndarray) -> float:
    """Expected PR-AUC of a random scorer = positive rate."""
    y = np.asarray(y).astype(int)
    return float(y.mean())


def at_chance(ci_low: float, ci_high: float, chance: float) -> bool:
    """Proposal rule R1: a detector is 'at chance' if the PR-AUC CI contains the chance level."""
    return bool(ci_low <= chance <= ci_high)


def evaluate_detector(detector: Any, X: np.ndarray, y: np.ndarray, B: int, seed: int, alpha: float = 0.05) -> dict[str, Any]:
    """AUC metrics with CIs plus thresholded metrics at the detector's stored threshold."""
    scores = detector.score(X)
    out: dict[str, Any] = {"detector": detector.name, "n": int(len(y)), "n_positive": int(np.asarray(y).sum())}
    out.update(bootstrap_auc_ci(y, scores, B=B, seed=seed, alpha=alpha))
    out["threshold"] = float(detector.threshold) if detector.threshold is not None else None
    if detector.threshold is not None:
        out.update(threshold_metrics(y, scores >= detector.threshold))
    return out
