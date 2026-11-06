"""Validation hold-out: 20% of the TRAINING NOMINALS (stratified by channel) + ALL training anomalies.

The remaining training nominals are the only rows ever passed to a detector's ``fit``. The test split is
``train == False`` and is never resampled. Indices are positions into the dataset.csv row order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from opssat_xai.io import read_json, write_json
from opssat_xai.seeds import rng


@dataclass
class Splits:
    """Row positions (into dataset.csv) of each role."""

    train_nominal: np.ndarray
    val_nominal: np.ndarray
    val_anomaly: np.ndarray
    test: np.ndarray
    seed: int
    validation_fraction: float
    stratify_by: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def val(self) -> np.ndarray:
        """Validation rows: held-out nominals followed by all training anomalies."""
        return np.concatenate([self.val_nominal, self.val_anomaly])


def stratified_holdout(idx: np.ndarray, strata: np.ndarray, fraction: float, generator: np.random.Generator) -> np.ndarray:
    """Return a subset of ``idx`` holding out ``fraction`` of the rows within each stratum (rounded, min 0)."""
    held: list[np.ndarray] = []
    for s in sorted(set(strata.tolist()), key=str):
        members = idx[strata == s]
        n_hold = int(round(fraction * len(members)))
        if n_hold == 0:
            continue
        chosen = generator.permutation(members)[:n_hold]
        held.append(np.sort(chosen))
    return np.sort(np.concatenate(held)) if held else np.array([], dtype=int)


def make_splits(y: np.ndarray, train: np.ndarray, channel: np.ndarray, cfg: dict[str, Any], seed: int) -> Splits:
    """Build the train-nominal / validation / test splits from the official ``train`` column."""
    y = np.asarray(y).astype(int)
    train = np.asarray(train).astype(bool)
    channel = np.asarray(channel)
    frac = float(cfg["splits"]["validation_fraction"])
    stratify_by = str(cfg["splits"].get("stratify_by", "channel"))
    all_idx = np.arange(len(y))
    train_nom = all_idx[train & (y == 0)]
    train_anom = all_idx[train & (y == 1)]
    test = all_idx[~train]
    generator = rng(seed, "validation_holdout")
    val_nom = stratified_holdout(train_nom, channel[train_nom], frac, generator)
    remaining = np.setdiff1d(train_nom, val_nom)
    splits = Splits(
        train_nominal=remaining,
        val_nominal=val_nom,
        val_anomaly=train_anom,
        test=test,
        seed=int(seed),
        validation_fraction=frac,
        stratify_by=stratify_by,
    )
    splits.counts = {
        "train_nominal": int(len(remaining)),
        "val_nominal": int(len(val_nom)),
        "val_anomaly": int(len(train_anom)),
        "val": int(len(val_nom) + len(train_anom)),
        "test": int(len(test)),
        "test_anomaly": int((y[test] == 1).sum()),
    }
    validate_splits(splits, y, train)
    return splits


def validate_splits(splits: Splits, y: np.ndarray, train: np.ndarray) -> None:
    """Assert no overlap, nominal-only train set, all training anomalies in validation, test untouched."""
    parts = [splits.train_nominal, splits.val_nominal, splits.val_anomaly, splits.test]
    union = np.concatenate(parts)
    if len(np.unique(union)) != len(union):
        raise AssertionError("Split indices overlap")
    if len(union) != len(y):
        raise AssertionError("Split indices do not cover every row exactly once")
    if np.any(y[splits.train_nominal] != 0) or np.any(y[splits.val_nominal] != 0):
        raise AssertionError("A nominal split contains anomalies")
    if not np.all(train[splits.train_nominal]) or not np.all(train[splits.val]):
        raise AssertionError("Validation/train-nominal rows must come from train == True")
    if np.any(train[splits.test]):
        raise AssertionError("Test rows must have train == False")
    expected_anom = np.flatnonzero(np.asarray(train) & (np.asarray(y) == 1))
    if not np.array_equal(np.sort(splits.val_anomaly), expected_anom):
        raise AssertionError("Validation must contain ALL training anomalies")


def save_splits(splits: Splits, path: str | Path) -> Path:
    """Write the splits to JSON (lists of row positions plus seed and counts)."""
    payload = asdict(splits)
    for k in ("train_nominal", "val_nominal", "val_anomaly", "test"):
        payload[k] = [int(i) for i in payload[k]]
    return write_json(path, payload)


def load_splits(path: str | Path) -> Splits:
    """Read splits written by :func:`save_splits`."""
    payload = read_json(path)
    return Splits(
        train_nominal=np.asarray(payload["train_nominal"], dtype=int),
        val_nominal=np.asarray(payload["val_nominal"], dtype=int),
        val_anomaly=np.asarray(payload["val_anomaly"], dtype=int),
        test=np.asarray(payload["test"], dtype=int),
        seed=int(payload["seed"]),
        validation_fraction=float(payload["validation_fraction"]),
        stratify_by=str(payload["stratify_by"]),
        counts={k: int(v) for k, v in payload.get("counts", {}).items()},
    )


def splits_equal(a: Splits, b: Splits) -> bool:
    """True if two Splits hold identical index arrays."""
    return all(
        np.array_equal(getattr(a, k), getattr(b, k)) for k in ("train_nominal", "val_nominal", "val_anomaly", "test")
    )
