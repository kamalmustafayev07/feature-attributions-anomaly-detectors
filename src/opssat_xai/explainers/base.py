"""Plain helpers shared by the explainer classes (no base-class hierarchy)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from opssat_xai.contracts import check_attribution, check_matrix
from opssat_xai.io import resolve


def score_function(detector: Any) -> Callable[[np.ndarray], np.ndarray]:
    """``lambda Z: detector.score(Z)`` accepting any array-like (what shap passes in)."""

    def f(Z: np.ndarray) -> np.ndarray:
        return detector.score(check_matrix(np.asarray(Z, dtype=np.float64)))

    return f


def explain_with_checkpoint(
    explainer: Any,
    X: np.ndarray,
    row_ids: np.ndarray,
    checkpoint_path: str | Path,
    every: int = 10,
    log: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Explain ``X`` row by row, saving ``.npz`` checkpoints every ``every`` rows and resuming if one exists.

    The checkpoint stores ``row_ids``, ``attr`` and a ``done`` mask; a checkpoint with different ``row_ids``
    is ignored (not resumed).
    """
    X = check_matrix(X)
    row_ids = np.asarray(row_ids)
    n, d = X.shape
    attr = np.full((n, d), np.nan)
    done = np.zeros(n, dtype=bool)
    path = resolve(checkpoint_path)
    if path.exists():
        with np.load(path, allow_pickle=False) as ck:
            if ck["attr"].shape == (n, d) and np.array_equal(ck["row_ids"], row_ids):
                attr, done = ck["attr"], ck["done"]
                if log:
                    log(f"resumed {int(done.sum())}/{n} rows from {path.name}")
    since_save = 0
    for i in range(n):
        if done[i]:
            continue
        attr[i] = explainer.attribute(X[i:i + 1], row_ids[i:i + 1])[0]
        done[i] = True
        since_save += 1
        if since_save >= every:
            _save_checkpoint(path, row_ids, attr, done)
            since_save = 0
            if log:
                log(f"checkpoint {int(done.sum())}/{n} rows -> {path.name}")
    _save_checkpoint(path, row_ids, attr, done)
    return check_attribution(attr, X)


def _save_checkpoint(path: Path, row_ids: np.ndarray, attr: np.ndarray, done: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, row_ids=row_ids, attr=attr, done=done)
    tmp.replace(path)
