"""Stage 3: SHAP, DEV and RND attributions on the clean explained rows (all test anomalies + test false positives).

Writes under results/<run>/<det>/: attributions_clean.npz, additivity_check.json (TreeSHAP),
gradient_vs_kernel.csv (autoencoder), checkpoints/ (KernelSHAP resume), runtime.json, manifest.json.

Usage: uv run python scripts/run_attributions.py --config configs/base.yaml [--detectors ...] [--max-rows N] [--resume]
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from opssat_xai.config import build_parser, config_from_args, results_dir
from opssat_xai.data.preprocess import load_standardized_data
from opssat_xai.detectors import load_fitted, model_filename
from opssat_xai.explainers import build_shap_explainer, save_attributions, shap_method_for
from opssat_xai.explainers.base import explain_with_checkpoint
from opssat_xai.explainers.dev import DevExplainer
from opssat_xai.explainers.rankings import rank, rank_abs
from opssat_xai.explainers.rnd import RndExplainer
from opssat_xai.explainers.shap_gradient import compare_attributions
from opssat_xai.explainers.shap_tree import check_additivity
from opssat_xai.io import finish_manifest, prepare_run_dir, start_manifest, write_csv, write_json
from opssat_xai.logging_utils import get_logger
from opssat_xai.robustness.runtime import Timer, runtime_record, write_runtime_json

log = get_logger("opssat_xai.run_attributions")


def explained_rows(y_test: np.ndarray, pred: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Positions (within the test set) of the rows to explain: anomalies and/or false positives."""
    e = cfg["explainers"]["explain_rows"]
    y = np.asarray(y_test).astype(int)
    pred = np.asarray(pred).astype(bool)
    mask = np.zeros(len(y), dtype=bool)
    if e.get("test_anomalies", True):
        mask |= y == 1
    if e.get("test_false_positives", True):
        mask |= (y == 0) & pred
    return np.flatnonzero(mask)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser("Compute SHAP / DEV / RND attributions on clean explained rows.")
    parser.add_argument("--detectors", nargs="+", default=None, help="subset of detectors_in_scope")
    parser.add_argument("--max-rows", type=int, default=None, help="explain only the first N rows (pilot runs)")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    detectors = args.detectors or list(cfg["detectors_in_scope"])
    data = load_standardized_data(cfg)
    det_dir = results_dir(cfg, "detection")
    n_repeats = int(cfg["explainers"]["rnd"]["n_repeats"])
    written: list[object] = []

    for name in detectors:
        run_dir = prepare_run_dir(results_dir(cfg, name), force=args.force, resume=args.resume)
        manifest = start_manifest(cfg, "scripts/run_attributions.py")
        runtime: list[dict[str, Any]] = []
        det = load_fitted(name, det_dir / model_filename(name, cfg))
        with np.load(det_dir / f"scores_{name}.npz") as z:
            test_pred, test_scores = z["test_pred"], z["test_scores"]
        pos = explained_rows(data.y_test, test_pred, cfg)
        if args.max_rows is not None:
            pos = pos[: args.max_rows]
        X = data.X_test[pos]
        y = np.asarray(data.y_test)[pos]
        row_ids = data.test_row_ids[pos]
        method = shap_method_for(name, cfg)
        log.info("%s: explaining %d rows (%d anomalies, %d false positives) with %s", name, len(pos), int((y == 1).sum()), int((y == 0).sum()), method)

        explainer = build_shap_explainer(det, cfg, data.X_train_nominal, method)
        with Timer() as t_shap:
            if method == "shap_kernel":
                shap_attr = explain_with_checkpoint(explainer, X, row_ids, run_dir / "checkpoints" / "shap_clean.npz",
                                                    every=int(cfg["explainers"]["shap_kernel"]["checkpoint_every"]), log=log.info)
            else:
                shap_attr = explainer.attribute(X, row_ids)
        runtime.append(runtime_record(f"{name}_{method}_clean", t_shap.seconds, n_items=len(pos)))
        if method == "shap_tree":
            n_check = int(cfg["explainers"]["shap_tree"]["additivity_check_rows"])
            check = check_additivity(det, X[:n_check], tol=float(cfg["explainers"]["shap_tree"]["additivity_tol"]))
            written.append(write_json(run_dir / "additivity_check.json", check))
            log.info("TreeSHAP additivity: %s", check)
        if method == "shap_gradient":
            n_cmp = min(int(cfg["explainers"]["shap_gradient"]["compare_rows"]), len(pos))
            if n_cmp > 0:
                kernel = build_shap_explainer(det, cfg, data.X_train_nominal, "shap_kernel")
                with Timer() as t_cmp:
                    kernel_attr = kernel.attribute(X[:n_cmp], row_ids[:n_cmp])
                runtime.append(runtime_record(f"{name}_shap_kernel_comparison", t_cmp.seconds, n_items=n_cmp))
                cmp = compare_attributions(shap_attr[:n_cmp], kernel_attr, row_ids[:n_cmp])
                cmp["score_minus_background_mean"] = det.score(X[:n_cmp]) - float(np.mean(det.score(explainer.background.numpy())))
                written.append(write_csv(run_dir / "gradient_vs_kernel.csv", cmp))
                log.info("gradient vs kernel SHAP on %d rows: median Spearman %.3f (decide per docs/decisions.md)", n_cmp, float(cmp["spearman"].median()))

        with Timer() as t_dev:
            dev_attr = DevExplainer().attribute(X, row_ids)
        runtime.append(runtime_record(f"{name}_dev_clean", t_dev.seconds, n_items=len(pos)))
        with Timer() as t_rnd:
            rnd_attr = np.stack([RndExplainer(int(cfg["seed"]), repeat=r).attribute(X, row_ids) for r in range(n_repeats)])
        runtime.append(runtime_record(f"{name}_rnd_clean", t_rnd.seconds, n_items=len(pos) * n_repeats))

        written.append(save_attributions(
            run_dir / "attributions_clean.npz",
            row_ids=row_ids, segment_ids=data.test_segment_ids[pos].astype(str), test_positions=pos, y=y,
            pred=test_pred[pos], score=test_scores[pos], is_tp=(y == 1) & test_pred[pos], is_fp=(y == 0) & test_pred[pos],
            channel=data.channel_test[pos].astype(str), feature_names=np.asarray(data.feature_names),
            shap=shap_attr, shap_method=np.asarray(method), dev=dev_attr, rnd=rnd_attr,
            ranking_shap=rank(shap_attr), ranking_shap_abs=rank_abs(shap_attr), ranking_dev=rank(dev_attr),
            ranking_dev_abs=rank_abs(dev_attr), ranking_rnd=np.stack([rank(r) for r in rnd_attr]),
        ))
        written.append(write_runtime_json(run_dir / "runtime_attributions.json", runtime, {"stage": "attributions", "detector": name}))
        written.append(finish_manifest(manifest, run_dir, written))
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
