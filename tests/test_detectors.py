import numpy as np
import pytest

from opssat_xai.contracts import Detector, mark_test_labels
from opssat_xai.detectors.autoencoder import AutoencoderDetector
from opssat_xai.detectors.iforest import IsolationForestDetector
from opssat_xai.detectors.knn import KNNDetector
from opssat_xai.detectors.metrics import at_chance, bootstrap_auc_ci, chance_pr_auc, threshold_metrics
from opssat_xai.detectors.ocsvm import OneClassSVMDetector, select_hyperparameters
from opssat_xai.detectors.thresholds import fpr_threshold, max_f1_threshold


def test_iforest_score_sign_matches_sklearn(standardized_arrays):
    d = standardized_arrays
    det = IsolationForestDetector(n_estimators=50, max_samples=64, random_state=0).fit(d.X_train_nominal)
    assert isinstance(det, Detector)
    assert np.allclose(det.score(d.X_val), -det.model.score_samples(d.X_val))
    det.set_threshold_from_validation(d.X_val, d.y_val)
    assert np.array_equal(det.predict(d.X_test), det.score(d.X_test) >= det.threshold)
    assert det.score(d.X_val[d.y_val == 1]).mean() > det.score(d.X_val[d.y_val == 0]).mean()


def test_ocsvm_score_sign_matches_sklearn(standardized_arrays):
    d = standardized_arrays
    det = OneClassSVMDetector(nu=0.05, gamma="scale").fit(d.X_train_nominal)
    assert np.allclose(det.score(d.X_val), -det.model.decision_function(d.X_val))


def test_knn_score(standardized_arrays):
    d = standardized_arrays
    det = KNNDetector(k=5).fit(d.X_train_nominal)
    s = det.score(d.X_val)
    assert s.shape == (len(d.y_val),) and np.all(s >= 0)
    assert s[d.y_val == 1].mean() > s[d.y_val == 0].mean()


def test_threshold_rejects_test_labels(standardized_arrays):
    d = standardized_arrays
    det = IsolationForestDetector(n_estimators=20, max_samples=64, random_state=0).fit(d.X_train_nominal)
    with pytest.raises(AssertionError, match="Test labels"):
        det.set_threshold_from_validation(d.X_test, d.y_test)
    with pytest.raises(AssertionError, match="Test labels"):
        max_f1_threshold(det.score(d.X_test), d.y_test)
    with pytest.raises(AssertionError, match="Test labels"):
        select_hyperparameters(d.X_train_nominal, d.X_test, d.y_test, {"nu": [0.1], "gamma": ["scale"]})
    assert mark_test_labels(np.array([0, 1]))[:1].is_test_labels


def test_max_f1_threshold_is_optimal():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95])
    y = np.array([0, 0, 0, 1, 1, 1])
    t, f1 = max_f1_threshold(scores, y)
    assert f1 == 1.0 and 0.3 < t <= 0.4
    assert fpr_threshold(np.arange(100.0), 0.05) >= 95


def test_ocsvm_grid_selects_by_validation_pr_auc(standardized_arrays):
    d = standardized_arrays
    best, table = select_hyperparameters(d.X_train_nominal, d.X_val, d.y_val, {"nu": [0.05, 0.2], "gamma": ["scale", 0.1]})
    assert len(table) == 4
    assert table["val_pr_auc"].max() == table[(table["nu"] == best["nu"]) & (table["gamma"].astype(str) == str(best["gamma"]))]["val_pr_auc"].iloc[0]


def test_autoencoder_deterministic_and_mse(standardized_arrays):
    d = standardized_arrays
    kw = dict(input_dim=d.d, hidden=(8, 4), max_epochs=5, seed=3)
    a = AutoencoderDetector(**kw).fit(d.X_train_nominal)
    b = AutoencoderDetector(**kw).fit(d.X_train_nominal)
    assert np.array_equal(a.score(d.X_val), b.score(d.X_val))
    recon = a.reconstruct(d.X_val)
    mse_np = ((recon.astype(np.float32) - d.X_val.astype(np.float32)) ** 2).mean(axis=1)
    assert np.allclose(a.score(d.X_val), mse_np, atol=1e-6)
    assert a.history["train_loss"].iloc[-1] <= a.history["train_loss"].iloc[0]
    assert next(a.net.parameters()).device.type == "cpu"


def test_autoencoder_save_load_roundtrip(standardized_arrays, tmp_path):
    d = standardized_arrays
    a = AutoencoderDetector(input_dim=d.d, max_epochs=3, seed=1).fit(d.X_train_nominal)
    a.set_threshold_from_validation(d.X_val, d.y_val)
    a.save(tmp_path / "ae.pt")
    b = AutoencoderDetector.load(tmp_path / "ae.pt")
    assert np.allclose(a.score(d.X_test), b.score(d.X_test)) and b.threshold == a.threshold


def test_metrics_and_chance_rule():
    y = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 1, 0, 1, 1])
    m = threshold_metrics(y, pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 3) and m["recall"] == 1.0
    ci = bootstrap_auc_ci(y, np.array([0.1, 0.2, 0.6, 0.3, 0.8, 0.9]), B=50, seed=0)
    assert ci["pr_auc_ci_low"] <= ci["pr_auc"] <= ci["pr_auc_ci_high"]
    assert chance_pr_auc(y) == pytest.approx(1 / 3)
    assert at_chance(0.2, 0.5, 1 / 3) and not at_chance(0.5, 0.9, 1 / 3)
