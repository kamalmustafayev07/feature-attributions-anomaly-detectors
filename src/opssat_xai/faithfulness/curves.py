"""Deletion and insertion curves for one anomaly, batched: all perturbed rows for all rankings of one anomaly
are built as one array and scored with ONE ``score`` call per (strategy, direction)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from opssat_xai.faithfulness.replacement import ReplacementStrategy


def features_to_replace(ranking: np.ndarray, k: int, direction: str) -> np.ndarray:
    """Deletion replaces the top-k; insertion starts fully replaced and reinstates the top-k originals."""
    ranking = np.asarray(ranking, dtype=int).reshape(-1)
    if direction == "deletion":
        return ranking[:k]
    if direction == "insertion":
        return ranking[k:]
    raise ValueError(f"Unknown direction '{direction}'")


def perturbed_rows(
    x: np.ndarray, ranking: np.ndarray, strategy: ReplacementStrategy, direction: str, rng: np.random.Generator
) -> np.ndarray:
    """All ``(K + 1) * m`` perturbed rows for ``k = 0..K`` (K = d), ordered by k then draw."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    K = x.shape[0]
    blocks = [strategy.replace(x, features_to_replace(ranking, k, direction), rng) for k in range(K + 1)]
    return np.vstack(blocks)


def score_curve(
    score_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    ranking: np.ndarray,
    strategy: ReplacementStrategy,
    direction: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Raw score per k (draws averaged), shape ``(K + 1,)``, one ``score_fn`` call."""
    rows = perturbed_rows(x, ranking, strategy, direction, rng)
    K = np.asarray(x).reshape(-1).shape[0]
    scores = np.asarray(score_fn(rows), dtype=np.float64).reshape(K + 1, strategy.m)
    return scores.mean(axis=1)


def curves_for_anomaly(
    score_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    rankings: Sequence[tuple[str, int, np.ndarray]],
    strategies: Sequence[ReplacementStrategy],
    directions: Sequence[str],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Curves for every ``(explainer, repeat, ranking)`` of one anomaly.

    Returns records ``{explainer, repeat, strategy, direction, k, score_raw}``; one score call per
    (strategy, direction) covers all rankings, so memory stays at one anomaly's batch.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    K = x.shape[0]
    records: list[dict[str, Any]] = []
    for strategy in strategies:
        for direction in directions:
            blocks = [perturbed_rows(x, r, strategy, direction, rng) for _, _, r in rankings]
            batch = np.vstack(blocks)
            scores = np.asarray(score_fn(batch), dtype=np.float64).reshape(len(rankings), K + 1, strategy.m).mean(axis=2)
            for (explainer, repeat, _), curve in zip(rankings, scores):
                for k in range(K + 1):
                    records.append({
                        "explainer": explainer, "repeat": int(repeat), "strategy": strategy.name,
                        "direction": direction, "k": int(k), "score_raw": float(curve[k]),
                    })
    return records
