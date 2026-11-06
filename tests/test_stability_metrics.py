import numpy as np
import pandas as pd
import pytest

from opssat_xai.stability.metrics import jaccard_topk, per_anomaly_means, rel_l2, retention_summary, spearman, stability_metrics
from opssat_xai.stability.perturb import perturb_copies, retained_mask


def test_identical_rankings():
    a = np.array([5.0, 3.0, -1.0, 2.0, 0.5])
    m = stability_metrics(a, a)
    assert m["jaccard3"] == 1.0 and m["jaccard5"] == 1.0 and m["spearman"] == pytest.approx(1.0) and m["rel_l2"] == 0.0


def test_disjoint_and_partial():
    assert jaccard_topk(np.array([0, 1, 2, 3]), np.array([3, 2, 1, 0]), 2) == 0.0
    assert jaccard_topk(np.array([0, 1, 2, 3]), np.array([0, 2, 1, 3]), 3) == 1.0
    assert jaccard_topk(np.array([0, 1, 2, 3]), np.array([0, 3, 2, 1]), 2) == pytest.approx(1 / 3)
    assert spearman(np.arange(5.0), -np.arange(5.0)) == pytest.approx(-1.0)
    assert np.isnan(spearman(np.ones(5), np.arange(5.0)))
    assert rel_l2(np.array([3.0, 4.0]), np.array([3.0, 4.0 + 5.0])) == pytest.approx(1.0)


def test_perturb_and_retained():
    x = np.zeros(18)
    copies = perturb_copies(x, 2000, 0.05, np.random.default_rng(0))
    assert copies.shape == (2000, 18)
    assert abs(copies.std() - 0.05) < 0.005

    class Det:
        threshold = 0.0

        def predict(self, X):
            return np.asarray(X)[:, 0] >= 0.0

    kept = retained_mask(Det(), np.array([1.0] + [0.0] * 17), np.array([[1.0] + [0.0] * 17, [-1.0] + [0.0] * 17]))
    assert kept.tolist() == [True, False]


def test_summaries():
    df = pd.DataFrame({
        "anomaly_id": [1, 1, 1, 1], "detector": ["d"] * 4, "explainer": ["shap", "shap", "dev", "dev"], "copy_id": [0, 1, 0, 1],
        "retained": [True, False, True, False], "jaccard3": [1.0, np.nan, 0.5, np.nan], "jaccard5": [1.0, np.nan, 0.5, np.nan],
        "spearman": [1.0, np.nan, 0.0, np.nan], "rel_l2": [0.0, np.nan, 1.0, np.nan],
    })
    per = per_anomaly_means(df)
    assert len(per) == 2 and per[per["explainer"] == "dev"]["jaccard3"].iloc[0] == 0.5
    ret = retention_summary(df)
    assert ret["retention_rate"].iloc[0] == 0.5 and ret["n_copies"].iloc[0] == 2
