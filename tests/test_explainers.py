import numpy as np

from opssat_xai.contracts import Explainer
from opssat_xai.detectors.autoencoder import AutoencoderDetector
from opssat_xai.detectors.iforest import IsolationForestDetector
from opssat_xai.explainers.base import explain_with_checkpoint, score_function
from opssat_xai.explainers.dev import DevExplainer
from opssat_xai.explainers.rankings import rank, rank_abs, top_k
from opssat_xai.explainers.rnd import RndExplainer
from opssat_xai.explainers.shap_gradient import GradientShapExplainer, compare_attributions, select_background
from opssat_xai.explainers.shap_kernel import KernelShapExplainer, make_background
from opssat_xai.explainers.shap_tree import TreeShapExplainer, check_additivity


def test_treeshap_additivity_and_sign(standardized_arrays):
    d = standardized_arrays
    det = IsolationForestDetector(n_estimators=30, max_samples=64, random_state=0).fit(d.X_train_nominal)
    check = check_additivity(det, d.X_val[:20], tol=1e-6)
    assert check["passed"], check
    expl = TreeShapExplainer(det)
    assert isinstance(expl, Explainer)
    attr = expl.attribute(d.X_val[:5])
    phi = np.asarray(expl.explainer.shap_values(d.X_val[:5]))
    assert attr.shape == (5, d.d) and np.allclose(attr, -phi)


def test_kernelshap_irrelevant_feature_and_local_accuracy():
    rng = np.random.default_rng(0)
    X_bg = rng.normal(size=(50, 2))

    class Toy:
        name = "toy"

        def score(self, X):
            return 3.0 * np.asarray(X)[:, 0]  # feature 1 is irrelevant

    expl = KernelShapExplainer(score_function(Toy()), make_background(X_bg, 10), n_features=2, nsamples=200, seed=0)
    x = np.array([[2.0, 5.0]])
    attr = expl.attribute(x, np.array([0]))
    assert abs(attr[0, 1]) < 1e-6
    assert np.isclose(attr.sum() + expl.expected_value(), Toy().score(x)[0], atol=1e-3)
    assert np.array_equal(attr, expl.attribute(x, np.array([0])))  # seeded per row


def test_gradientshap_shape_and_comparison(standardized_arrays):
    d = standardized_arrays
    ae = AutoencoderDetector(input_dim=d.d, max_epochs=3, seed=0).fit(d.X_train_nominal)
    expl = GradientShapExplainer(ae.score_module(), select_background(d.X_train_nominal, 20, seed=0), nsamples=20, seed=0)
    attr = expl.attribute(d.X_val[:3])
    assert attr.shape == (3, d.d) and np.all(np.isfinite(attr))
    assert np.array_equal(attr, expl.attribute(d.X_val[:3]))
    cmp = compare_attributions(attr, attr)
    assert np.allclose(cmp["spearman"], 1.0)


def test_dev_and_rnd():
    X = np.array([[-2.0, 0.5, 1.0], [0.0, -3.0, 2.0]])
    assert np.array_equal(DevExplainer().attribute(X), np.abs(X))
    r0 = RndExplainer(seed=1, repeat=0).attribute(X, np.array([10, 11]))
    r0b = RndExplainer(seed=1, repeat=0).attribute(X, np.array([10, 11]))
    r1 = RndExplainer(seed=1, repeat=1).attribute(X, np.array([10, 11]))
    assert np.array_equal(r0, r0b) and not np.array_equal(r0, r1)
    for row in rank(r0):
        assert sorted(row.tolist()) == [0, 1, 2]


def test_rankings():
    attr = np.array([[0.1, -5.0, 2.0]])
    assert rank(attr).tolist() == [[2, 0, 1]]
    assert rank_abs(attr).tolist() == [[1, 2, 0]]
    assert top_k(rank(attr), 2).tolist() == [[2, 0]]


def test_checkpoint_resume(tmp_path):
    X = np.arange(10.0).reshape(5, 2)
    ids = np.arange(5)
    calls = []

    class Counting:
        name = "dev"

        def attribute(self, X, row_ids=None):
            calls.append(int(row_ids[0]))
            return np.abs(X)

    path = tmp_path / "ck.npz"
    out = explain_with_checkpoint(Counting(), X, ids, path, every=2)
    assert np.array_equal(out, np.abs(X)) and len(calls) == 5
    with np.load(path) as ck:
        assert ck["done"].all()
    out2 = explain_with_checkpoint(Counting(), X, ids, path, every=2)
    assert np.array_equal(out2, out) and len(calls) == 5  # nothing recomputed
    # partial checkpoint -> only the missing rows are computed
    np.savez(path, row_ids=ids, attr=np.where(np.arange(5)[:, None] < 3, np.abs(X), np.nan), done=np.arange(5) < 3)
    out3 = explain_with_checkpoint(Counting(), X, ids, path, every=2)
    assert np.array_equal(out3, out) and calls[5:] == [3, 4]
