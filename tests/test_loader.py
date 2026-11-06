import copy

import pandas as pd
import pytest

from opssat_xai.config import active_feature_names
from opssat_xai.data.loader import DatasetValidationError, dataset_arrays, load_dataset, validate_dataset


def test_loader_accepts_matching_file(synthetic_cfg, synthetic_frame):
    df = load_dataset(synthetic_cfg)
    assert len(df) == len(synthetic_frame)
    arrays = dataset_arrays(df, synthetic_cfg)
    assert arrays.X.shape == (len(df), 18)
    assert arrays.feature_names == active_feature_names(synthetic_cfg)
    assert arrays.train.dtype == bool and arrays.y.sum() == 42


def test_wrong_hash_raises(synthetic_cfg):
    cfg = copy.deepcopy(synthetic_cfg)
    cfg["data"]["dataset_md5"] = "0" * 32
    with pytest.raises(DatasetValidationError, match="MD5"):
        load_dataset(cfg)


def test_wrong_counts_raise(synthetic_cfg):
    cfg = copy.deepcopy(synthetic_cfg)
    cfg["data"]["expected_counts"]["train_nominal"] += 1
    with pytest.raises(DatasetValidationError, match="counts"):
        load_dataset(cfg)


def test_wrong_row_count_raises(synthetic_cfg):
    cfg = copy.deepcopy(synthetic_cfg)
    cfg["data"]["n_rows"] += 1
    with pytest.raises(DatasetValidationError, match="rows"):
        load_dataset(cfg)


def test_missing_column_raises(synthetic_cfg, synthetic_frame):
    df = synthetic_frame.drop(columns=["kurtosis"])
    with pytest.raises(DatasetValidationError, match="missing"):
        validate_dataset(df, synthetic_cfg)


def test_string_train_column_is_coerced(synthetic_cfg, synthetic_frame):
    df = synthetic_frame.copy()
    df["train"] = df["train"].map({True: "True", False: "False"})
    counts = validate_dataset(df, synthetic_cfg)
    assert counts == synthetic_cfg["data"]["expected_counts"]


def test_feature_exclusion(synthetic_cfg):
    cfg = copy.deepcopy(synthetic_cfg)
    cfg["features"]["exclude"] = ["len", "duration", "gaps_squared"]
    names = active_feature_names(cfg)
    assert len(names) == 15 and "len" not in names
    df = load_dataset(cfg)
    assert dataset_arrays(df, cfg).X.shape[1] == 15
