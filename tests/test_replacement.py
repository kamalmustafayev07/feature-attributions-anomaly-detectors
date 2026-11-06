import numpy as np
import pytest

from opssat_xai.faithfulness.curves import features_to_replace, score_curve
from opssat_xai.faithfulness.replacement import make_strategy


@pytest.fixture
def pool():
    rng = np.random.default_rng(0)
    return rng.normal(size=(200, 18)) + rng.normal(size=18) * 0.3  # non-zero medians


def test_r1_k0_and_kd(pool, linear_scorer):
    score, w = linear_scorer
    r1 = make_strategy("R1", pool)
    x = np.abs(np.random.default_rng(1).normal(size=18)) + 1
    ranking = np.argsort(-(w * x))
    curve = score_curve(score, x, ranking, r1, "deletion", np.random.default_rng(0))
    assert curve.shape == (19,)
    assert curve[0] == pytest.approx(score(x[None])[0])
    assert curve[18] == pytest.approx(score(np.median(pool, axis=0)[None])[0])
    ins = score_curve(score, x, ranking, r1, "insertion", np.random.default_rng(0))
    assert ins[0] == pytest.approx(curve[18]) and ins[18] == pytest.approx(curve[0])


def test_r1_replace_shape_and_values(pool):
    r1 = make_strategy("R1", pool)
    x = np.ones(18)
    out = r1.replace(x, np.array([0, 5]), np.random.default_rng(0))
    assert out.shape == (1, 18)
    assert out[0, 0] == np.median(pool[:, 0]) and out[0, 5] == np.median(pool[:, 5]) and out[0, 1] == 1.0


def test_r2_deterministic_under_seed_and_from_pool(pool):
    r2 = make_strategy("R2", pool, n_draws=20)
    x = np.full(18, 100.0)
    idx = np.array([2, 3])
    a = r2.replace(x, idx, np.random.default_rng(42))
    b = r2.replace(x, idx, np.random.default_rng(42))
    c = r2.replace(x, idx, np.random.default_rng(43))
    assert a.shape == (20, 18) and np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.all(a[:, 2][:, None] == pool[:, 2][None, :]).sum() == 0  # elementwise check below instead
    assert all(v in set(pool[:, 2]) for v in a[:, 2]) and all(v in set(pool[:, 3]) for v in a[:, 3])
    assert np.all(a[:, [0, 1, 4]] == 100.0)
    assert r2.replace(x, np.array([], dtype=int), np.random.default_rng(0)).shape == (20, 18)


def test_features_to_replace():
    ranking = np.array([4, 2, 0, 1, 3])
    assert features_to_replace(ranking, 2, "deletion").tolist() == [4, 2]
    assert features_to_replace(ranking, 2, "insertion").tolist() == [0, 1, 3]
    assert features_to_replace(ranking, 0, "insertion").tolist() == [4, 2, 0, 1, 3]
