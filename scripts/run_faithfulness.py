"""Stage 4: deletion / insertion curves under R1 and R2 for every explained anomaly, AUDC per anomaly.

Writes under results/<run>/<det>/: faithfulness_<strategy>_<direction>.csv (long format), audc_per_anomaly.csv,
faithfulness_meta.json (normalizer, rank saturation), runtime_faithfulness.json; figures/<run>/faithfulness_<det>_<strategy>.png.

The config key ``faithfulness.plan`` (direction -> strategies) restricts the (strategy, direction) pairs; run_all.py
sets it from ``faithfulness.level1_plan`` for ``--level 1``.

Usage: uv run python scripts/run_faithfulness.py --config configs/base.yaml [--detectors ...] [--force]
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from opssat_xai.config import build_parser, config_from_args, figures_dir, results_dir
from opssat_xai.contracts import FAITHFULNESS_COLUMNS
from opssat_xai.data.preprocess import load_standardized_data
from opssat_xai.detectors import load_fitted, model_filename
from opssat_xai.evaluation.normalize import ScoreNormalizer, rank_saturation_fraction
from opssat_xai.explainers import load_attributions
from opssat_xai.faithfulness.audc import audc_table
from opssat_xai.faithfulness.curves import curves_for_anomaly
from opssat_xai.faithfulness.replacement import make_strategy
from opssat_xai.io import finish_manifest, start_manifest, write_csv, write_json
from opssat_xai.logging_utils import get_logger
from opssat_xai.plotting.curves import plot_faithfulness_curves
from opssat_xai.robustness.runtime import Timer, runtime_record, write_runtime_json
from opssat_xai.seeds import rng

log = get_logger("opssat_xai.run_faithfulness")


def planned_pairs(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """(strategy, direction) pairs to run: ``faithfulness.plan`` if present, else the full product."""
    f = cfg["faithfulness"]
    plan = f.get("plan")
    if plan:
        return [(s, d) for d, strategies in plan.items() for s in strategies]
    return [(s, d) for s in f["strategies"] for d in f["directions"]]


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser("Deletion / insertion faithfulness curves and AUDC.")
    parser.add_argument("--detectors", nargs="+", default=None, help="subset of detectors_in_scope")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    detectors = args.detectors or list(cfg["detectors_in_scope"])
    data = load_standardized_data(cfg)
    det_dir = results_dir(cfg, "detection")
    pairs = planned_pairs(cfg)
    strategies = {s: make_strategy(s, data.X_train_nominal, int(cfg["faithfulness"]["r2_n_draws"])) for s, _ in pairs}
    written: list[object] = []

    for name in detectors:
        run_dir = results_dir(cfg, name)
        if not (run_dir / "attributions_clean.npz").exists():
            raise FileNotFoundError(f"{run_dir / 'attributions_clean.npz'} missing; run scripts/run_attributions.py first")
        existing = [p for p in run_dir.glob("faithfulness_*.csv")]
        if existing and not (args.force or args.resume):
            raise FileExistsError(f"faithfulness files already exist in {run_dir}; pass --force")
        manifest = start_manifest(cfg, "scripts/run_faithfulness.py")
        det = load_fitted(name, det_dir / model_filename(name, cfg))
        with np.load(det_dir / f"scores_{name}.npz") as z:
            normalizer = ScoreNormalizer.fit(z["val_nominal_scores"])
        A = load_attributions(run_dir / "attributions_clean.npz")
        anomalies = np.flatnonzero(A["y"] == 1)
        X = data.X_test[A["test_positions"][anomalies]]
        n_repeats = A["rnd"].shape[0]
        log.info("%s: %d anomalies (%d true positives), pairs=%s", name, len(anomalies), int(A["is_tp"][anomalies].sum()), pairs)

        by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {p: [] for p in pairs}
        with Timer() as t_all:
            for j, i in enumerate(anomalies):
                anomaly_id = int(A["row_ids"][i])
                rankings = [("shap", 0, A["ranking_shap"][i]), ("dev", 0, A["ranking_dev"][i])]
                rankings += [("rnd", r, A["ranking_rnd"][r, i]) for r in range(n_repeats)]
                for strategy_name, direction in pairs:
                    recs = curves_for_anomaly(det.score, X[j], rankings, [strategies[strategy_name]], [direction],
                                              rng(int(cfg["seed"]), "faithfulness", anomaly_id, strategy_name, direction))
                    raw = np.array([r["score_raw"] for r in recs])
                    ranks, zs = normalizer.rank(raw), normalizer.z(raw)
                    for r, sr, sz in zip(recs, ranks, zs):
                        by_pair[(strategy_name, direction)].append({"anomaly_id": anomaly_id, "detector": name, **r,
                                                                    "score_rank": float(sr), "score_z": float(sz)})
        frames = []
        for (strategy_name, direction), recs in by_pair.items():
            df = pd.DataFrame(recs)[list(FAITHFULNESS_COLUMNS)]
            written.append(write_csv(run_dir / f"faithfulness_{strategy_name}_{direction}.csv", df))
            frames.append(df)
        long_df = pd.concat(frames, ignore_index=True)
        audc = audc_table(long_df)
        tp_map = dict(zip(A["row_ids"].astype(int), A["is_tp"].astype(bool)))
        audc["is_tp"] = audc["anomaly_id"].map(tp_map)
        written.append(write_csv(run_dir / "audc_per_anomaly.csv", audc))
        clean_scores = A["score"][anomalies]
        meta = {"normalizer": {"mean": normalizer.mean, "std": normalizer.std, "n_reference": int(normalizer.reference_sorted.size)},
                "rank_saturation_fraction_all_anomalies": rank_saturation_fraction(normalizer, clean_scores),
                "rank_saturation_fraction_true_positives": rank_saturation_fraction(normalizer, clean_scores[A["is_tp"][anomalies]]),
                "saturation_switch_fraction": float(cfg["faithfulness"]["normalization"]["saturation_switch_fraction"]),
                "pairs": pairs, "n_anomalies": int(len(anomalies)), "n_true_positives": int(A["is_tp"][anomalies].sum())}
        written.append(write_json(run_dir / "faithfulness_meta.json", meta))
        for strategy_name in sorted({s for s, _ in pairs}):
            sub = long_df[(long_df["strategy"] == strategy_name) & long_df["anomaly_id"].map(tp_map)]
            if len(sub):
                written.append(plot_faithfulness_curves(sub, name, strategy_name, figures_dir(cfg, f"faithfulness_{name}_{strategy_name}.png"),
                                                        seed=int(cfg["seed"])))
        written.append(write_runtime_json(run_dir / "runtime_faithfulness.json",
                                          [runtime_record(f"{name}_faithfulness", t_all.seconds, n_items=len(anomalies))],
                                          {"stage": "faithfulness", "detector": name}))
        written.append(finish_manifest(manifest, run_dir, written, filename="manifest_faithfulness.json"))
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
