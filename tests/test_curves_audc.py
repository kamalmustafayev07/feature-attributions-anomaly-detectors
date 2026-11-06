import numpy as np
import pandas as pd
import pytest

from opssat_xai.faithfulness.audc import audc, audc_table
from opssat_xai.faithfulness.curves import curves_for_anomaly, score_curve
from opssat_xai.faithfulness.replacement import make_strategy


def test_audc_constant_and_linear():
    assert audc(np.full(19, 0.37)) == pytest.approx(0.37)
    assert audc(np.linspace(1.0, 0.0, 19)) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        audc(np.array([1.0]))


def test_true_ranking_beats_random_on_linear_scorer(linear_scorer):
    score, w = linear_scorer
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(300, 18))
    r1 = make_strategy("R1", pool)
    x = np.abs(rng.normal(size=18)) + 0.5
    true_rank = np.argsort(-(w * x))
    rand = rng.permutation(18)
    true_curve = score_curve(score, x, true_rank, r1, "deletion", rng)
    rand_curve = score_curve(score, x, rand, r1, "deletion", rng)
    assert audc(true_curve) < audc(rand_curve)
    assert audc(score_curve(score, x, true_rank, r1, "insertion", rng)) > audc(score_curve(score, x, rand, r1, "insertion", rng))


def test_curves_for_anomaly_batches_rankings(linear_scorer):
    score, w = linear_scorer
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(100, 18))
    strategies = [make_strategy("R1", pool), make_strategy("R2", pool, n_draws=5)]
    x = rng.normal(size=18)
    rankings = [("shap", 0, np.argsort(-(w * x))), ("rnd", 0, rng.permutation(18)), ("rnd", 1, rng.permutation(18))]
    recs = curves_for_anomaly(score, x, rankings, strategies, ["deletion", "insertion"], np.random.default_rng(1))
    assert len(recs) == 3 * 2 * 2 * 19
    df = pd.DataFrame(recs)
    k0 = df[(df["k"] == 0) & (df["direction"] == "deletion")]["score_raw"]
    assert np.allclose(k0, score(x[None])[0])
    full = df[(df["k"] == 18) & (df["direction"] == "deletion") & (df["strategy"] == "R1")]["score_raw"]
    assert np.allclose(full, full.iloc[0])  # fully replaced vector identical across rankings


def test_audc_table_averages_repeats():
    rows = []
    for rep, offset in ((0, 0.0), (1, 1.0)):
        for k in range(3):
            rows.append({"anomaly_id": 1, "detector": "d", "explainer": "rnd", "repeat": rep, "strategy": "R1", "direction": "deletion",
                         "k": k, "score_raw": offset, "score_rank": offset, "score_z": offset})
    t = audc_table(pd.DataFrame(rows))
    assert len(t) == 1 and t["audc_rank"].iloc[0] == pytest.approx(0.5) and t["n_repeats"].iloc[0] == 2 and t["K"].iloc[0] == 2
