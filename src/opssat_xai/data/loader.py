"""Load and validate OPSSAT-AD ``dataset.csv`` (Zenodo v2) against configs/data.yaml.

Run ``uv run python -m opssat_xai.data.loader --verify`` to print the MD5 check and the Table 1 counts.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opssat_xai.config import active_feature_names, build_parser, config_from_args
from opssat_xai.io import resolve


class DatasetValidationError(ValueError):
    """Raised when the dataset file does not match the pinned hash, shape, columns or class counts."""


@dataclass(frozen=True)
class DatasetArrays:
    """Raw (unscaled) arrays extracted from dataset.csv in file row order."""

    X: np.ndarray            # (n, d) float64, columns = feature_names
    y: np.ndarray            # (n,) int, 1 = anomaly
    train: np.ndarray        # (n,) bool, official split
    channel: np.ndarray      # (n,) object
    segment_id: np.ndarray   # (n,) object (row position if no id column is configured)
    feature_names: list[str]


def md5sum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """MD5 hex digest of a file."""
    h = hashlib.md5()
    with resolve(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def coerce_bool(series: pd.Series, name: str) -> np.ndarray:
    """Coerce a bool / 0-1 / 'True'-'False' column to a boolean array, raising on anything else."""
    if series.dtype == bool:
        return series.to_numpy()
    values = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    unknown = sorted(set(values.unique()) - set(mapping))
    if unknown:
        raise DatasetValidationError(f"Column '{name}' has non-boolean values {unknown[:5]}")
    return values.map(mapping).to_numpy(dtype=bool)


def validate_dataset(df: pd.DataFrame, cfg: dict[str, Any], md5: str | None = None) -> dict[str, int]:
    """Check hash, row count, configured columns and Table 1 class counts; return the observed counts."""
    data_cfg = cfg["data"]
    cols = data_cfg["columns"]
    # TODO(verify): column names below come from configs/data.yaml and must be confirmed against the real CSV header.
    expected_md5 = data_cfg.get("dataset_md5")
    if md5 is not None and expected_md5 and md5.lower() != str(expected_md5).lower():
        raise DatasetValidationError(f"dataset.csv MD5 mismatch: got {md5}, expected {expected_md5}")
    n_rows = int(data_cfg["n_rows"])
    if len(df) != n_rows:
        raise DatasetValidationError(f"dataset.csv has {len(df)} rows, expected {n_rows}")
    required = list(cfg["features"]["names"]) + [cols["channel"], cols["label"], cols["train"]]
    if cols.get("segment_id"):
        required.append(cols["segment_id"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"dataset.csv is missing configured columns {missing}; found {list(df.columns)}")
    train = coerce_bool(df[cols["train"]], cols["train"])
    y = (df[cols["label"]].to_numpy() == data_cfg.get("label_positive", 1)).astype(int)
    counts = {
        "train_nominal": int(((y == 0) & train).sum()),
        "train_anomaly": int(((y == 1) & train).sum()),
        "test_nominal": int(((y == 0) & ~train).sum()),
        "test_anomaly": int(((y == 1) & ~train).sum()),
    }
    expected = {k: int(v) for k, v in data_cfg["expected_counts"].items()}
    if counts != expected:
        raise DatasetValidationError(f"Class counts {counts} differ from the expected Table 1 counts {expected}")
    return counts


def load_dataset(cfg: dict[str, Any], path: str | Path | None = None, verify_md5: bool = True) -> pd.DataFrame:
    """Read dataset.csv and validate it (MD5, row count, columns, class counts). Raises on any mismatch."""
    p = resolve(path or cfg["data"]["dataset_file"])
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download dataset.csv from https://zenodo.org/records/15108715 into data/raw/ (see README)."
        )
    md5 = md5sum(p) if verify_md5 else None
    df = pd.read_csv(p)
    validate_dataset(df, cfg, md5=md5)
    return df


def dataset_arrays(df: pd.DataFrame, cfg: dict[str, Any], feature_names: list[str] | None = None) -> DatasetArrays:
    """Extract the feature matrix (explicit column list, never 'everything numeric') and metadata arrays."""
    cols = cfg["data"]["columns"]
    names = feature_names or active_feature_names(cfg)
    X = df[names].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(X)):
        bad = [n for n in names if not np.all(np.isfinite(df[n].to_numpy(dtype=np.float64)))]
        raise DatasetValidationError(f"Non-finite values in feature columns {bad}")
    y = (df[cols["label"]].to_numpy() == cfg["data"].get("label_positive", 1)).astype(int)
    train = coerce_bool(df[cols["train"]], cols["train"])
    channel = df[cols["channel"]].to_numpy()
    seg_col = cols.get("segment_id")
    segment_id = df[seg_col].to_numpy() if seg_col else np.arange(len(df))
    return DatasetArrays(X=X, y=y, train=train, channel=channel, segment_id=segment_id, feature_names=list(names))


def _main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = build_parser("Verify dataset.csv against configs/data.yaml")
    parser.add_argument("--verify", action="store_true", help="print the MD5 check and class counts")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    p = resolve(cfg["data"]["dataset_file"])
    md5 = md5sum(p)
    df = pd.read_csv(p)
    counts = validate_dataset(df, cfg, md5=md5)
    print(f"{p}: md5 {md5} matches configured {cfg['data']['dataset_md5']}")
    print(f"rows={len(df)} counts={counts}")
    print(f"feature columns present: {len(active_feature_names(cfg))} of {len(cfg['features']['names'])} configured")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
