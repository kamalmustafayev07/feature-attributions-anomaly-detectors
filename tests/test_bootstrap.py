import numpy as np
import pandas as pd
import pytest

from opssat_xai.stats.bootstrap import bootstrap_mean, bootstrap_paired_mean_diff
from opssat_xai.stats.paired import cliffs_delta, cohen_dz, faithfulness_comparisons, holm_correction, paired_comparison, wilcoxon_p


def test_bootstrap_ci_covers_known_mean():
    rng = np.random.default_rng(123)
    covered = 0
    for trial in range(100):
        x = rng.normal(loc=0.3, scale=1.0, size=60)
        b = bootstrap_mean(x, B=500, seed=trial)
        covered += b["ci_low"] <= 0.3 <= b["ci_high"]
    assert covered >= 90  # nominal 95%; allow Monte-Carlo slack
    b = bootstrap_mean(np.arange(10.0), B=200, seed=0)
    assert b == bootstrap_mean(np.arange(10.0), B=200, seed=0) and b["mean"] == 4.5 and b["n"] == 10


def test_paired_and_effect_sizes():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a - 1.0
    d = bootstrap_paired_mean_diff(a, b, B=100, seed=0)
    assert d["mean"] == pytest.approx(1.0) and d["ci_low"] == pytest.approx(1.0) and d["ci_high"] == pytest.approx(1.0)
    assert cliffs_delta(a, b) > 0 and cliffs_delta(a, a) == 0.0
    assert np.isnan(cohen_dz(np.zeros(5))) and cohen_dz(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)
    assert wilcoxon_p(np.zeros(5)) == 1.0
    res = paired_comparison(shap_values=np.array([0.1, 0.2, 0.15]), other_values=np.array([0.5, 0.6, 0.55]), higher_is_better=False, B=100, seed=0, alpha=0.05, key="k")
    assert res["mean_diff"] == pytest.approx(0.4) and res["cliffs_delta"] == 1.0  # lower AUDC = better, positive favours SHAP


def test_holm():
    adj = holm_correction(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06])  # step-down: 0.01*3, max(0.03*2, 0.04*1) carried forward
    adj2 = holm_correction(np.array([0.01, np.nan, 0.5]))
    assert np.isnan(adj2[1]) and adj2[0] == pytest.approx(0.02) and adj2[2] == 0.5


def test_faithfulness_comparisons_orientation():
    rows = []
    for a in range(6):
        for e, v in (("shap", 0.2), ("dev", 0.4), ("rnd", 0.6)):
            rows.append({"anomaly_id": a, "detector": "d", "explainer": e, "strategy": "R1", "direction": "deletion", "audc_rank": v + 0.01 * a})
            rows.append({"anomaly_id": a, "detector": "d", "explainer": e, "strategy": "R1", "direction": "insertion", "audc_rank": 1 - v + 0.01 * a})
    t = faithfulness_comparisons(pd.DataFrame(rows), "audc_rank", B=100, seed=0, alpha=0.05)
    assert len(t) == 4 and (t["mean_diff"] > 0).all() and "p_holm" in t.columns
