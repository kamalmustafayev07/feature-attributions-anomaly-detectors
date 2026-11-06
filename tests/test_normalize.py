import numpy as np
import pytest

from opssat_xai.evaluation.normalize import ScoreNormalizer, rank_saturation_fraction


def test_rank_and_z():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    n = ScoreNormalizer.fit(ref)
    assert n.rank(np.array([4.0]))[0] == 1.0
    assert n.rank(np.array([0.5]))[0] == 0.0
    assert n.rank(np.array([2.5]))[0] == 0.5
    assert np.all((n.rank(np.linspace(-5, 10, 50)) >= 0) & (n.rank(np.linspace(-5, 10, 50)) <= 1))
    assert n.z(np.array([2.5]))[0] == pytest.approx(0.0)
    assert n.z(np.array([2.5 + ref.std(ddof=1)]))[0] == pytest.approx(1.0)
    assert rank_saturation_fraction(n, np.array([4.0, 5.0, 1.0, 2.0])) == 0.5
    assert ScoreNormalizer.from_dict(n.to_dict()).rank(np.array([3.0]))[0] == 0.75


def test_constant_reference_raises():
    with pytest.raises(ValueError):
        ScoreNormalizer.fit(np.ones(5))
