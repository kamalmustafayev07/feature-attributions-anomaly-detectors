"""Stage 6: bootstrap CIs, paired comparisons (SHAP vs DEV, SHAP vs RND), Holm correction and verdicts.

Writes under results/<run>/stats/: faithfulness_summary.csv, stability_summary.csv, final_faithfulness.csv,
final_stability.csv, verdicts.json, manifest.json.

Usage: uv run python scripts/run_stats.py --config configs/base.yaml [--detectors ...] [--force]
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from opssat_xai.config import build_parser, config_from_args, results_dir
from opssat_xai.io import finish_manifest, prepare_run_dir, read_csv, read_json, start_manifest, write_csv, write_json
from opssat_xai.logging_utils import get_logger
from opssat_xai.stability.metrics import per_anomaly_means, retention_summary
from opssat_xai.stats.bootstrap import bootstrap_mean
from opssat_xai.stats.paired import faithfulness_comparisons, stability_comparisons
from opssat_xai.stats.verdicts import add_verdicts, research_question_verdicts

log = get_logger("opssat_xai.run_stats")

AUDC_COLS = ("audc_rank", "audc_z", "audc_raw")
STAB_METRICS = ("jaccard3", "jaccard5", "spearman", "rel_l2")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser("Statistics: CIs, paired tests, effect sizes, Holm, verdicts.")
    parser.add_argument("--detectors", nargs="+", default=None, help="subset of detectors_in_scope")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    detectors = args.detectors or list(cfg["detectors_in_scope"])
    run_dir = prepare_run_dir(results_dir(cfg, "stats"), force=args.force, resume=args.resume)
    manifest = start_manifest(cfg, "scripts/run_stats.py")
    B, alpha, margin, seed = int(cfg["stats"]["bootstrap_B"]), float(cfg["stats"]["alpha"]), float(cfg["stats"]["equivalence_margin"]), int(cfg["seed"])
    written: list[object] = []

    audc_frames: list[pd.DataFrame] = []
    stab_frames: list[pd.DataFrame] = []
    meta: dict[str, Any] = {}
    for name in detectors:
        d = results_dir(cfg, name)
        if (d / "audc_per_anomaly.csv").exists():
            audc_frames.append(read_csv(d / "audc_per_anomaly.csv"))
        if (d / "stability.csv").exists():
            stab_frames.append(read_csv(d / "stability.csv"))
        if (d / "faithfulness_meta.json").exists():
            meta[name] = read_json(d / "faithfulness_meta.json")
    audc = pd.concat(audc_frames, ignore_index=True) if audc_frames else pd.DataFrame()
    stab = pd.concat(stab_frames, ignore_index=True) if stab_frames else pd.DataFrame()

    # Faithfulness summaries (mean + CI) on true positives (primary) and all anomalies (secondary)
    rows: list[dict[str, Any]] = []
    final_f = pd.DataFrame()
    if len(audc):
        for subset, sub in (("true_positives", audc[audc["is_tp"].astype(bool)]), ("all_anomalies", audc)):
            for keys, g in sub.groupby(["detector", "explainer", "strategy", "direction"], sort=True):
                for col in AUDC_COLS:
                    b = bootstrap_mean(g[col].to_numpy(), B=B, seed=seed, alpha=alpha, key=f"{subset}|{keys}|{col}")
                    rows.append({"subset": subset, "detector": keys[0], "explainer": keys[1], "strategy": keys[2], "direction": keys[3],
                                 "metric": col, **b})
        written.append(write_csv(run_dir / "faithfulness_summary.csv", pd.DataFrame(rows)))
        tp = audc[audc["is_tp"].astype(bool)]
        comps = [faithfulness_comparisons(tp, col, B, seed, alpha) for col in ("audc_rank", "audc_z")]
        final_f = add_verdicts(pd.concat([c for c in comps if len(c)], ignore_index=True), margin) if any(len(c) for c in comps) else pd.DataFrame()
        written.append(write_csv(run_dir / "final_faithfulness.csv", final_f))
        secondary = [faithfulness_comparisons(audc, col, B, seed, alpha) for col in ("audc_rank", "audc_z")]
        if any(len(c) for c in secondary):
            written.append(write_csv(run_dir / "final_faithfulness_all_anomalies.csv", add_verdicts(pd.concat(secondary, ignore_index=True), margin)))

    final_s = pd.DataFrame()
    if len(stab):
        per = per_anomaly_means(stab, STAB_METRICS)
        srows: list[dict[str, Any]] = []
        for (detector, explainer), g in per.groupby(["detector", "explainer"], sort=True):
            for metric in STAB_METRICS:
                b = bootstrap_mean(g[metric].to_numpy(), B=B, seed=seed, alpha=alpha, key=f"stab|{detector}|{explainer}|{metric}")
                srows.append({"detector": detector, "explainer": explainer, "metric": metric, **b})
        ssum = pd.DataFrame(srows).merge(retention_summary(stab), on="detector", how="left")
        written.append(write_csv(run_dir / "stability_summary.csv", ssum))
        final_s = add_verdicts(stability_comparisons(per, B, seed, alpha, STAB_METRICS), margin)
        written.append(write_csv(run_dir / "final_stability.csv", final_s))

    primary = "audc_" + str(cfg["faithfulness"]["normalization"].get("primary", "rank"))
    verdicts = research_question_verdicts(final_f, final_s, margin=margin, metric=primary)
    verdicts["rank_saturation"] = {k: {kk: v[kk] for kk in v if kk.startswith("rank_saturation")} for k, v in meta.items()}
    verdicts["rules"] = cfg["stats"].get("verdict_rules")
    written.append(write_json(run_dir / "verdicts.json", verdicts))
    written.append(finish_manifest(manifest, run_dir, written))
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
