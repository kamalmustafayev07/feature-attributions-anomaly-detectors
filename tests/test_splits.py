import numpy as np
import pytest

from opssat_xai.config import active_feature_names
from opssat_xai.data.loader import dataset_arrays, load_dataset
from opssat_xai.data.preprocess import fit_scaler, transform
from opssat_xai.data.splits import load_splits, make_splits, save_splits, splits_equal


def _arrays(cfg):
    return dataset_arrays(load_dataset(cfg), cfg, active_feature_names(cfg))


def test_splits_roles_and_no_overlap(synthetic_cfg):
    a = _arrays(synthetic_cfg)
    s = make_splits(a.y, a.train, a.channel, synthetic_cfg, seed=1)
    union = np.concatenate([s.train_nominal, s.val_nominal, s.val_anomaly, s.test])
    assert len(np.unique(union)) == len(a.y)
    assert np.all(a.y[s.train_nominal] == 0)
    assert np.all(a.y[s.val_nominal] == 0)
    assert np.array_equal(np.sort(s.val_anomaly), np.flatnonzero(a.train & (a.y == 1)))
    assert np.array_equal(s.test, np.flatnonzero(~a.train))
    assert abs(len(s.val_nominal) - 0.2 * (a.train & (a.y == 0)).sum()) <= 2


def test_splits_stratified_and_seeded(synthetic_cfg):
    a = _arrays(synthetic_cfg)
    s1 = make_splits(a.y, a.train, a.channel, synthetic_cfg, seed=7)
    s2 = make_splits(a.y, a.train, a.channel, synthetic_cfg, seed=7)
    s3 = make_splits(a.y, a.train, a.channel, synthetic_cfg, seed=8)
    assert splits_equal(s1, s2) and not splits_equal(s1, s3)
    for ch in np.unique(a.channel):
        n_total = int(((a.channel == ch) & a.train & (a.y == 0)).sum())
        n_val = int((a.channel[s1.val_nominal] == ch).sum())
        assert abs(n_val - 0.2 * n_total) < 1


def test_splits_roundtrip(synthetic_cfg, tmp_path):
    a = _arrays(synthetic_cfg)
    s = make_splits(a.y, a.train, a.channel, synthetic_cfg, seed=1)
    save_splits(s, tmp_path / "splits.json")
    assert splits_equal(s, load_splits(tmp_path / "splits.json"))


def test_scaler_sees_no_anomalies(synthetic_cfg):
    a = _arrays(synthetic_cfg)
    s = make_splits(a.y, a.train, a.channel, synthetic_cfg, seed=1)
    scaler = fit_scaler(a.X[s.train_nominal], a.y[s.train_nominal])
    Z = transform(scaler, a.X[s.train_nominal])
    assert Z.dtype == np.float64
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-12)
    with pytest.raises(AssertionError):
        fit_scaler(a.X[s.val], a.y[s.val])
