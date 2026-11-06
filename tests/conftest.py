"""Synthetic fixtures: a DataFrame with the dataset.csv schema (18 features, channel, anomaly, train), a config
pointing at it in a temp directory, and a linear scorer with a known feature ordering. No data is downloaded."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from opssat_xai.config import load_config
from opssat_xai.data.loader import md5sum

warnings.filterwarnings("ignore", category=UserWarning)  # IsolationForest max_samples > n_samples on tiny data

FEATURES = ["mean", "var", "std", "kurtosis", "skew", "n_peaks", "duration", "len", "len_weighted", "gaps_squared",
            "var_div_duration", "var_div_len", "smooth10_n_peaks", "smooth20_n_peaks", "diff_peaks", "diff2_peaks",
            "diff_var", "diff2_var"]
COUNTS = {"train_nominal": 120, "train_anomaly": 30, "test_nominal": 40, "test_anomaly": 12}


def make_synthetic_frame(counts: dict[str, int] = COUNTS, seed: int = 0) -> pd.DataFrame:
    """Nominals ~ N(0, 1), anomalies shifted on the first features; 4 channels; official train column."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, n in counts.items():
        is_train = key.startswith("train")
        is_anom = key.endswith("anomaly")
        X = rng.normal(size=(n, len(FEATURES)))
        if is_anom:
            X[:, :4] += 4.0
        for i in range(n):
            rows.append({"segment": f"{key}_{i}", "anomaly": int(is_anom), "channel": f"CH{i % 4}", "train": bool(is_train),
                         **{f: float(X[i, j]) for j, f in enumerate(FEATURES)}})
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    return make_synthetic_frame()


@pytest.fixture
def synthetic_cfg(tmp_path: Path, synthetic_frame: pd.DataFrame) -> dict[str, Any]:
    """Project config rewired to the synthetic CSV and a temp results/processed tree."""
    csv = tmp_path / "dataset.csv"
    synthetic_frame.to_csv(csv, index=False)
    cfg = copy.deepcopy(load_config("configs/base.yaml"))
    cfg["run_name"] = "test"
    cfg["data"]["dataset_file"] = str(csv)
    cfg["data"]["dataset_md5"] = md5sum(csv)
    cfg["data"]["n_rows"] = len(synthetic_frame)
    cfg["data"]["expected_counts"] = dict(COUNTS)
    cfg["paths"]["results"] = str(tmp_path / "results")
    cfg["paths"]["figures"] = str(tmp_path / "figures")
    cfg["paths"]["data_processed"] = str(tmp_path / "processed")
    cfg["splits"]["splits_file"] = str(tmp_path / "processed" / "splits.json")
    cfg["detectors"]["iforest"]["n_estimators"] = 50
    cfg["detectors"]["iforest"]["max_samples"] = 64
    cfg["detectors"]["autoencoder"]["max_epochs"] = 5
    return cfg


@pytest.fixture
def linear_scorer():
    """``score(X) = X @ w`` with strictly decreasing positive weights, so the true deletion ranking of a row x is
    ``argsort(-(w * x))``."""
    w = np.linspace(2.0, 0.1, len(FEATURES))

    def score(X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ w

    return score, w


@pytest.fixture
def standardized_arrays(synthetic_cfg: dict[str, Any]):
    """Standardized train-nominal / validation / test arrays built from the synthetic frame."""
    from opssat_xai.config import active_feature_names
    from opssat_xai.data.loader import dataset_arrays, load_dataset
    from opssat_xai.data.preprocess import build_standardized_data, fit_scaler
    from opssat_xai.data.splits import make_splits

    df = load_dataset(synthetic_cfg)
    arrays = dataset_arrays(df, synthetic_cfg, active_feature_names(synthetic_cfg))
    splits = make_splits(arrays.y, arrays.train, arrays.channel, synthetic_cfg, seed=synthetic_cfg["seed"])
    scaler = fit_scaler(arrays.X[splits.train_nominal], arrays.y[splits.train_nominal])
    return build_standardized_data(arrays, splits, scaler)
