"""Explainers behind one contract: attribute(X) -> (n, d), positive = pushes the anomaly score up.

``build_shap_explainer`` maps the configured SHAP flavour to a detector; ``load_attributions`` reads the
``attributions_clean.npz`` written by scripts/run_attributions.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from opssat_xai.io import resolve
from opssat_xai.seeds import int_seed


def shap_method_for(detector_name: str, cfg: dict[str, Any]) -> str:
    """Configured SHAP flavour (shap_tree / shap_kernel / shap_gradient) for a detector."""
    return str(cfg["explainers"]["shap_for_detector"][detector_name])


def build_shap_explainer(detector: Any, cfg: dict[str, Any], X_train_nominal: np.ndarray, method: str | None = None) -> Any:
    """Instantiate the SHAP explainer for ``detector`` (background built from the standardized train nominals)."""
    from opssat_xai.explainers.base import score_function
    from opssat_xai.explainers.shap_gradient import GradientShapExplainer, select_background
    from opssat_xai.explainers.shap_kernel import KernelShapExplainer, make_background
    from opssat_xai.explainers.shap_tree import TreeShapExplainer

    method = method or shap_method_for(detector.name, cfg)
    ecfg = cfg["explainers"]
    seed = int(cfg["seed"])
    d = int(X_train_nominal.shape[1])
    if method == "shap_tree":
        return TreeShapExplainer(detector)
    if method == "shap_kernel":
        k = ecfg["shap_kernel"]
        background = make_background(X_train_nominal, int(k.get("background_size", 100)))
        return KernelShapExplainer(score_function(detector), background, n_features=d, nsamples=int(k.get("nsamples", 2048)),
                                   l1_reg=str(k.get("l1_reg", "num_features")), seed=int_seed(seed, "shap_kernel", detector.name))
    if method == "shap_gradient":
        g = ecfg["shap_gradient"]
        background = select_background(X_train_nominal, int(g.get("background_size", 100)), seed=seed)
        return GradientShapExplainer(detector.score_module(), background, nsamples=int(g.get("nsamples", 200)),
                                     seed=int_seed(seed, "shap_gradient", detector.name))
    raise ValueError(f"Unknown SHAP method '{method}'")


def save_attributions(path: str | Path, **arrays: Any) -> Path:
    """Write an ``.npz`` of attribution arrays (strings stored as unicode arrays, no pickling)."""
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **{k: np.asarray(v) for k, v in arrays.items()})
    return p


def load_attributions(path: str | Path) -> dict[str, np.ndarray]:
    """Read an ``.npz`` written by :func:`save_attributions` into a dict of arrays."""
    with np.load(resolve(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
