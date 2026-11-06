"""Anomaly detectors behind one contract: fit(X_nominal), score(X) (higher = more anomalous), threshold, predict.

``build_detector`` / ``save_fitted`` / ``load_fitted`` are the only places that map config names to classes
(plain functions, no registry).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opssat_xai.seeds import int_seed

DETECTOR_NAMES: tuple[str, ...] = ("iforest", "ocsvm", "knn", "autoencoder")


def build_detector(name: str, cfg: dict[str, Any], input_dim: int, params: dict[str, Any] | None = None) -> Any:
    """Instantiate a detector from ``cfg['detectors'][name]`` (seeds derived from the global seed)."""
    from opssat_xai.detectors.autoencoder import AutoencoderDetector
    from opssat_xai.detectors.iforest import IsolationForestDetector
    from opssat_xai.detectors.knn import KNNDetector
    from opssat_xai.detectors.ocsvm import OneClassSVMDetector

    seed = int(cfg["seed"])
    d = dict(cfg["detectors"][name])
    d.update(params or {})
    threads = cfg.get("threads", {})
    if name == "iforest":
        return IsolationForestDetector(
            n_estimators=d["n_estimators"], max_samples=d["max_samples"], contamination=d["contamination"],
            max_features=d.get("max_features", 1.0), random_state=int_seed(seed, "iforest"),
            n_jobs=int(threads.get("sklearn_n_jobs", 1)),
        )
    if name == "ocsvm":
        return OneClassSVMDetector(nu=d.get("nu", 0.05), gamma=d.get("gamma", "scale"), kernel=d.get("kernel", "rbf"),
                                   tol=d.get("tol", 1e-4), cache_size=d.get("cache_size", 500))
    if name == "knn":
        return KNNDetector(k=d.get("k", 5))
    if name == "autoencoder":
        return AutoencoderDetector(
            input_dim=input_dim, hidden=tuple(d.get("hidden", (8, 4))), lr=d.get("lr", 1e-3),
            weight_decay=d.get("weight_decay", 1e-5), batch_size=d.get("batch_size", 64),
            max_epochs=d.get("max_epochs", 200), patience=d.get("patience", 20),
            holdout_fraction=d.get("holdout_fraction", 0.1), seed=int_seed(seed, "autoencoder"),
            n_threads=int(threads.get("torch", 1)),
        )
    raise ValueError(f"Unknown detector '{name}' (expected one of {DETECTOR_NAMES})")


def model_filename(name: str, cfg: dict[str, Any]) -> str:
    """File name under results/<run>/detection/ holding the fitted detector."""
    if name == "autoencoder":
        return str(cfg["detectors"]["autoencoder"].get("weights_file", "ae.pt"))
    return f"{name}.joblib"


def save_fitted(detector: Any, path: str | Path) -> Path:
    """Persist a fitted detector (torch.save for the autoencoder, joblib otherwise)."""
    from opssat_xai.detectors.base import save_detector

    if detector.name == "autoencoder":
        return detector.save(path)
    return save_detector(detector, path)


def load_fitted(name: str, path: str | Path) -> Any:
    """Load a detector persisted by :func:`save_fitted`."""
    from opssat_xai.detectors.autoencoder import AutoencoderDetector
    from opssat_xai.detectors.base import load_detector

    if name == "autoencoder":
        return AutoencoderDetector.load(path)
    return load_detector(path)
