"""StandardScaler fitted on the train nominals only; every downstream array is standardized float64."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from opssat_xai.config import active_feature_names, processed_dir
from opssat_xai.contracts import TestLabels, check_matrix, mark_test_labels
from opssat_xai.data.loader import DatasetArrays, dataset_arrays, load_dataset
from opssat_xai.data.splits import Splits, load_splits
from opssat_xai.io import resolve


def fit_scaler(X_train_nominal: np.ndarray, y_train_nominal: np.ndarray | None = None) -> StandardScaler:
    """Fit a StandardScaler on train-nominal rows. If labels are given, assert that none is an anomaly."""
    X = check_matrix(X_train_nominal, "X_train_nominal")
    if y_train_nominal is not None and np.any(np.asarray(y_train_nominal) != 0):
        raise AssertionError("fit_scaler received anomalous rows; the scaler must see train nominals only")
    return StandardScaler().fit(X)


def transform(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Standardize ``X`` and return float64."""
    return np.asarray(scaler.transform(check_matrix(X)), dtype=np.float64)


def save_scaler(scaler: StandardScaler, path: str | Path) -> Path:
    """Persist the scaler with joblib."""
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, p)
    return p


def load_scaler(path: str | Path) -> StandardScaler:
    """Load a scaler persisted by :func:`save_scaler`."""
    return joblib.load(resolve(path))


def feature_stats(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Per-feature summary (raw units) used for the metadata investigation."""
    X = check_matrix(X)
    q = np.quantile(X, [0.0, 0.25, 0.5, 0.75, 1.0], axis=0)
    return pd.DataFrame({
        "feature": feature_names,
        "mean": X.mean(axis=0), "std": X.std(axis=0, ddof=1),
        "min": q[0], "q25": q[1], "median": q[2], "q75": q[3], "max": q[4],
        "n_unique": [len(np.unique(X[:, j])) for j in range(X.shape[1])],
    })


@dataclass
class StandardizedData:
    """Everything a run needs after preprocessing, already standardized."""

    feature_names: list[str]
    X_train_nominal: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_val_nominal: np.ndarray
    X_test: np.ndarray
    y_test: TestLabels
    channel_val: np.ndarray
    channel_test: np.ndarray
    test_row_ids: np.ndarray       # positions into dataset.csv
    test_segment_ids: np.ndarray
    splits: Splits

    @property
    def d(self) -> int:
        """Number of active features."""
        return len(self.feature_names)


def build_standardized_data(arrays: DatasetArrays, splits: Splits, scaler: StandardScaler) -> StandardizedData:
    """Assemble the standardized train-nominal / validation / test arrays from raw arrays and splits."""
    X = arrays.X
    return StandardizedData(
        feature_names=list(arrays.feature_names),
        X_train_nominal=transform(scaler, X[splits.train_nominal]),
        X_val=transform(scaler, X[splits.val]),
        y_val=arrays.y[splits.val].astype(int),
        X_val_nominal=transform(scaler, X[splits.val_nominal]),
        X_test=transform(scaler, X[splits.test]),
        y_test=mark_test_labels(arrays.y[splits.test].astype(int)),
        channel_val=arrays.channel[splits.val],
        channel_test=arrays.channel[splits.test],
        test_row_ids=np.asarray(splits.test, dtype=int),
        test_segment_ids=arrays.segment_id[splits.test],
        splits=splits,
    )


def load_standardized_data(cfg: dict[str, Any]) -> StandardizedData:
    """Load dataset.csv, data/processed/splits.json and the run's scaler and return standardized arrays."""
    df = load_dataset(cfg)
    names = active_feature_names(cfg)
    arrays = dataset_arrays(df, cfg, names)
    splits = load_splits(cfg["splits"]["splits_file"])
    scaler = load_scaler(processed_dir(cfg) / "scaler.joblib")
    if getattr(scaler, "n_features_in_", len(names)) != len(names):
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features but the config selects {len(names)}; rerun scripts/prepare_data.py"
        )
    return build_standardized_data(arrays, splits, scaler)
