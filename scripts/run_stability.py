"""Stage 5: stability of attributions under x + N(0, sigma^2 I) for every explained true positive.

Writes under results/<run>/<det>/: stability.csv (one row per copy per explainer), stability_summary.csv,
checkpoints/shap_stability.npz (KernelSHAP resume), runtime_stability.json; figures/<run>/stability_<det>.png.

Usage: uv run python scripts/run_stability.py --config configs/base.yaml [--detectors ...] [--resume] [--force]
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from opssat_xai.config import build_parser, config_from_args, figures_dir, results_dir
from opssat_xai.contracts import STABILITY_COLUMNS
from opssat_xai.data.preprocess import load_standardized_data
from opssat_xai.detectors import load_fitted, model_filename
from opssat_xai.explainers import build_shap_explainer, load_attributions, shap_method_for
from opssat_xai.explainers.base import explain_with_checkpoint
from opssat_xai.explainers.dev import DevExplainer
from opssat_xai.explainers.rnd import RndExplainer
from opssat_xai.io import finish_manifest, start_manifest, write_csv
from opssat_xai.logging_utils import get_logger
from opssat_xai.plotting.stability import plot_stability_boxes
from opssat_xai.robustness.runtime import Timer, runtime_record, write_runtime_json
from opssat_xai.seeds import rng
from opssat_xai.stability.metrics import per_anomaly_means, retention_summary, stability_metrics
from opssat_xai.stability.perturb import perturb_copies, retained_mask

log = get_logger("opssat_xai.run_stability")

COPY_ID_STRIDE = 10_000  # unique row id per copy: anomaly_id * stride + copy_id (seeds and checkpoints)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser("Attribution stability under Gaussian input noise.")
    parser.add_argument("--detectors", nargs="+", default=None, help="subset of detectors_in_scope")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    detectors = args.detectors or list(cfg["detectors_in_scope"])
    data = load_standardized_data(cfg)
    det_dir = results_dir(cfg, "detection")
    s_cfg = cfg["stability"]
    n_copies, sigma = int(s_cfg["n_copies"]), float(s_cfg["sigma"])
    top_ks = tuple(int(k) for k in s_cfg.get("top_k", (3, 5)))
    written: list[object] = []

    for name in detectors:
        run_dir = results_dir(cfg, name)
        if not (run_dir / "attributions_clean.npz").exists():
            raise FileNotFoundError(f"{run_dir / 'attributions_clean.npz'} missing; run scripts/run_attributions.py first")
        if (run_dir / "stability.csv").exists() and not (args.force or args.resume):
            raise FileExistsError(f"{run_dir / 'stability.csv'} already exists; pass --force")
        manifest = start_manifest(cfg, "scripts/run_stability.py")
        det = load_fitted(name, det_dir / model_filename(name, cfg))
        A = load_attributions(run_dir / "attributions_clean.npz")
        tps = np.flatnonzero(A["is_tp"])
        X = data.X_test[A["test_positions"][tps]]
        log.info("%s: %d true positives x %d copies, sigma=%.3f", name, len(tps), n_copies, sigma)

        # 1) all copies for all anomalies (small: TPs x n_copies x d), retained flags
        copies, meta = [], []
        for j, i in enumerate(tps):
            anomaly_id = int(A["row_ids"][i])
            c = perturb_copies(X[j], n_copies, sigma, rng(int(cfg["seed"]), "stability", anomaly_id))
            kept = retained_mask(det, X[j], c)
            copies.append(c)
            meta += [{"anomaly_id": anomaly_id, "copy_id": k, "retained": bool(kept[k]), "clean_index": int(i)} for k in range(n_copies)]
        copies_arr = np.vstack(copies) if copies else np.empty((0, data.d))
        meta_df = pd.DataFrame(meta, columns=["anomaly_id", "copy_id", "retained", "clean_index"])
        keep = meta_df["retained"].to_numpy(dtype=bool) if len(meta_df) else np.zeros(0, dtype=bool)
        X_keep = copies_arr[keep]
        ids_keep = (meta_df["anomaly_id"].to_numpy() * COPY_ID_STRIDE + meta_df["copy_id"].to_numpy())[keep]
        log.info("%s: retained %d / %d copies", name, int(keep.sum()), len(meta_df))

        # 2) explain the retained copies only
        method = shap_method_for(name, cfg)
        explainer = build_shap_explainer(det, cfg, data.X_train_nominal, method)
        with Timer() as t_shap:
            if method == "shap_kernel":
                shap_c = explain_with_checkpoint(explainer, X_keep, ids_keep, run_dir / "checkpoints" / "shap_stability.npz",
                                                 every=int(s_cfg.get("checkpoint_every", 10)) * max(1, n_copies), log=log.info)
            else:
                shap_c = explainer.attribute(X_keep, ids_keep) if len(X_keep) else np.empty((0, data.d))
        dev_c = DevExplainer().attribute(X_keep, ids_keep) if len(X_keep) else np.empty((0, data.d))
        rnd_c = RndExplainer(int(cfg["seed"]), repeat=0).attribute(X_keep, ids_keep) if len(X_keep) else np.empty((0, data.d))

        # 3) metrics against the clean attribution
        rows: list[dict[str, Any]] = []
        kept_pos = np.flatnonzero(keep)
        pos_of = {int(p): q for q, p in enumerate(kept_pos)}
        for p, m in enumerate(meta_df.itertuples(index=False)):
            i = int(m.clean_index)
            clean = {"shap": A["shap"][i], "dev": A["dev"][i], "rnd": A["rnd"][0, i]}
            for explainer_name, attr_c in (("shap", shap_c), ("dev", dev_c), ("rnd", rnd_c)):
                rec = {"anomaly_id": int(m.anomaly_id), "detector": name, "explainer": explainer_name, "copy_id": int(m.copy_id), "retained": bool(m.retained)}
                if m.retained:
                    rec.update(stability_metrics(clean[explainer_name], attr_c[pos_of[p]], top_ks))
                else:
                    rec.update({f"jaccard{k}": np.nan for k in top_ks})
                    rec.update({"spearman": np.nan, "rel_l2": np.nan})
                rows.append(rec)
        stab = pd.DataFrame(rows, columns=list(STABILITY_COLUMNS))
        written.append(write_csv(run_dir / "stability.csv", stab))
        summary = per_anomaly_means(stab).groupby(["detector", "explainer"], as_index=False)[["jaccard3", "jaccard5", "spearman", "rel_l2"]].mean()
        summary = summary.merge(retention_summary(stab), on="detector", how="left")
        summary["n_copies_per_anomaly"] = n_copies
        summary["sigma"] = sigma
        written.append(write_csv(run_dir / "stability_summary.csv", summary))
        if len(stab) and stab["retained"].any():
            written.append(plot_stability_boxes(stab, name, figures_dir(cfg, f"stability_{name}.png")))
        written.append(write_runtime_json(run_dir / "runtime_stability.json",
                                          [runtime_record(f"{name}_{method}_stability", t_shap.seconds, n_items=int(keep.sum()))],
                                          {"stage": "stability", "detector": name, "n_copies": n_copies, "sigma": sigma}))
        written.append(finish_manifest(manifest, run_dir, written, filename="manifest_stability.json"))
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
