"""Stage 2: fit every detector in scope on the train nominals, choose thresholds on validation, evaluate on test.

Writes under results/<run>/detection/: <det>.joblib | ae.pt, ocsvm_grid.csv, <det>_val.json, <det>_test_metrics.json,
scores_<det>.npz (validation / test scores and predictions), summary.csv, runtime.json, manifest.json.

Usage: uv run python scripts/run_detection.py --config configs/base.yaml [--detectors iforest ocsvm] [--force]
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from opssat_xai.config import build_parser, config_from_args, results_dir
from opssat_xai.data.preprocess import load_standardized_data
from opssat_xai.detectors import build_detector, model_filename, save_fitted
from opssat_xai.detectors.metrics import at_chance, bootstrap_auc_ci, chance_pr_auc, evaluate_detector
from opssat_xai.detectors.ocsvm import select_hyperparameters
from opssat_xai.detectors.thresholds import fpr_threshold
from opssat_xai.io import finish_manifest, prepare_run_dir, start_manifest, write_csv, write_json
from opssat_xai.logging_utils import get_logger
from opssat_xai.robustness.runtime import Timer, runtime_record, write_runtime_json
from opssat_xai.seeds import int_seed

log = get_logger("opssat_xai.run_detection")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser("Fit detectors on train nominals, threshold on validation, evaluate on test.")
    parser.add_argument("--detectors", nargs="+", default=None, help="subset of detectors_in_scope")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    detectors = args.detectors or list(cfg["detectors_in_scope"])
    run_dir = prepare_run_dir(results_dir(cfg, "detection"), force=args.force, resume=args.resume)
    manifest = start_manifest(cfg, "scripts/run_detection.py", seeds={"root": cfg["seed"], **{f"{n}": int_seed(cfg["seed"], n) for n in detectors}})
    written: list[object] = []
    runtime: list[dict[str, Any]] = []

    data = load_standardized_data(cfg)
    B = int(cfg["detection"]["bootstrap_B"])
    alpha = float(cfg["stats"]["alpha"])
    rule = str(cfg["threshold"]["rule"])
    secondary_fpr = float(cfg["threshold"]["secondary_fpr"])
    chance_val = chance_pr_auc(data.y_val)
    summary_rows: list[dict[str, Any]] = []

    for name in detectors:
        params: dict[str, Any] = {}
        if name == "ocsvm":
            o = cfg["detectors"]["ocsvm"]
            with Timer() as t_grid:
                params, grid = select_hyperparameters(data.X_train_nominal, data.X_val, data.y_val, o["grid"],
                                                      kernel=o.get("kernel", "rbf"), tol=o.get("tol", 1e-4), cache_size=o.get("cache_size", 500))
            runtime.append(runtime_record("ocsvm_grid", t_grid.seconds, n_items=len(grid)))
            written.append(write_csv(run_dir / "ocsvm_grid.csv", grid))
            log.info("ocsvm grid best: %s", params)
        det = build_detector(name, cfg, data.d, params)
        with Timer() as t_fit:
            det.fit(data.X_train_nominal)
        runtime.append(runtime_record(f"{name}_fit", t_fit.seconds, n_items=data.X_train_nominal.shape[0]))
        det.set_threshold_from_validation(data.X_val, data.y_val, rule=rule)
        val_nominal_scores = det.score(data.X_val_nominal)
        val_scores = det.score(data.X_val)
        threshold_fpr = fpr_threshold(val_nominal_scores, secondary_fpr)
        val = bootstrap_auc_ci(data.y_val, val_scores, B=B, seed=int_seed(cfg["seed"], "val_bootstrap", name), alpha=alpha)
        val_info = {
            "detector": name, "params": {**{k: v for k, v in cfg["detectors"][name].items() if k != "grid"}, **params},
            "threshold_rule": rule, "threshold": float(det.threshold),
            f"threshold_fpr{int(secondary_fpr * 100):02d}": threshold_fpr,
            "val_chance_pr_auc": chance_val, "at_chance": at_chance(val["pr_auc_ci_low"], val["pr_auc_ci_high"], chance_val),
            **{f"val_{k}": v for k, v in val.items()},
        }
        if name == "autoencoder":
            val_info["best_epoch"] = det.best_epoch
            written.append(write_csv(run_dir / "ae_training_history.csv", det.history))
        written.append(write_json(run_dir / f"{name}_val.json", val_info))
        with Timer() as t_test:
            test = evaluate_detector(det, data.X_test, data.y_test, B=B, seed=int_seed(cfg["seed"], "test_bootstrap", name), alpha=alpha)
        runtime.append(runtime_record(f"{name}_test_eval", t_test.seconds))
        written.append(write_json(run_dir / f"{name}_test_metrics.json", test))
        test_scores = det.score(data.X_test)
        test_pred = det.predict(data.X_test)
        np.savez(run_dir / f"scores_{name}.npz", val_scores=val_scores, val_nominal_scores=val_nominal_scores,
                 y_val=np.asarray(data.y_val), test_scores=test_scores, test_pred=test_pred,
                 y_test=np.asarray(data.y_test), test_row_ids=data.test_row_ids, threshold=float(det.threshold))
        written.append(run_dir / f"scores_{name}.npz")
        written.append(save_fitted(det, run_dir / model_filename(name, cfg)))
        summary_rows.append({"detector": name, "threshold": float(det.threshold), f"threshold_fpr{int(secondary_fpr * 100):02d}": threshold_fpr,
                             "val_pr_auc": val["pr_auc"], "val_pr_auc_ci_low": val["pr_auc_ci_low"], "val_pr_auc_ci_high": val["pr_auc_ci_high"],
                             "val_chance_pr_auc": chance_val, "at_chance": val_info["at_chance"],
                             **{k: v for k, v in test.items() if k not in ("detector",)}})
        log.info("%s: threshold=%.6g val PR-AUC=%.3f [%.3f, %.3f] at_chance=%s test PR-AUC=%.3f F1=%.3f TP=%d FP=%d",
                 name, det.threshold, val["pr_auc"], val["pr_auc_ci_low"], val["pr_auc_ci_high"], val_info["at_chance"],
                 test["pr_auc"], test["f1"], test["tp"], test["fp"])

    written.append(write_csv(run_dir / "summary.csv", pd.DataFrame(summary_rows)))
    written.append(write_runtime_json(run_dir / "runtime.json", runtime, {"stage": "detection", "run_name": cfg["run_name"]}))
    written.append(finish_manifest(manifest, run_dir, written))
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
