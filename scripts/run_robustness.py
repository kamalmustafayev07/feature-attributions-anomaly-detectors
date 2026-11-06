"""Stage 7: correlated feature groups, group-level rankings, disagreement and false-positive subsets, example
figures, runtime table (and the ablation comparison figure when a second run is given).

Writes under results/<run>/robustness/: correlation_groups.json, group_rankings.csv, disagreement.csv,
disagreement_audc.csv, false_positives.csv, runtime/summary.json; figures/<run>/examples_*.png, fp_examples_*.png,
ablation_comparison.png (with --compare-run).

Usage: uv run python scripts/run_robustness.py --config configs/base.yaml [--compare-run ablation_A] [--force]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from opssat_xai.config import build_parser, config_from_args, figures_dir, results_dir
from opssat_xai.data.preprocess import load_standardized_data
from opssat_xai.explainers import load_attributions
from opssat_xai.explainers.rankings import rank
from opssat_xai.io import finish_manifest, prepare_run_dir, read_csv, read_json, resolve, start_manifest, write_csv, write_json
from opssat_xai.logging_utils import get_logger
from opssat_xai.plotting.curves import plot_ablation_comparison
from opssat_xai.plotting.examples import load_segment, plot_example
from opssat_xai.robustness.correlation import correlated_groups, group_attributions, spearman_matrix, top1_agreement
from opssat_xai.robustness.subsets import disagreement_mask, top_features_table
from opssat_xai.stats.bootstrap import bootstrap_mean

log = get_logger("opssat_xai.run_robustness")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser("Robustness analyses: correlation groups, disagreement, false positives, runtime.")
    parser.add_argument("--detectors", nargs="+", default=None, help="subset of detectors_in_scope")
    parser.add_argument("--compare-run", default=None, help="run_name of a second run (e.g. ablation_A) for the comparison figure")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    detectors = args.detectors or list(cfg["detectors_in_scope"])
    run_dir = prepare_run_dir(results_dir(cfg, "robustness"), force=args.force, resume=args.resume)
    manifest = start_manifest(cfg, "scripts/run_robustness.py")
    r_cfg = cfg["robustness"]
    seed, B = int(cfg["seed"]), int(cfg["stats"]["bootstrap_B"])
    data = load_standardized_data(cfg)
    names = data.feature_names
    ablation_set = set(cfg["features"]["ablation_A_features"])
    written: list[object] = []

    corr = spearman_matrix(data.X_train_nominal, names)
    groups = correlated_groups(corr, float(r_cfg["correlation_threshold"]))
    group_of = {f: gi for gi, g in enumerate(groups) for f in g}
    written.append(write_json(run_dir / "correlation_groups.json", {"threshold": r_cfg["correlation_threshold"], "groups": groups,
                                                                     "n_groups": len(groups), "feature_names": names}))

    group_rows, dis_rows, dis_audc_rows, fp_rows = [], [], [], []
    for name in detectors:
        d = results_dir(cfg, name)
        if not (d / "attributions_clean.npz").exists():
            log.warning("%s: no attributions, skipped", name)
            continue
        A = load_attributions(d / "attributions_clean.npz")
        tps = np.flatnonzero(A["is_tp"])
        fps = np.flatnonzero(A["is_fp"])
        shap_tp, dev_tp = A["shap"][tps], A["dev"][tps]
        # group-level rankings
        g_shap, g_dev = group_attributions(shap_tp, groups, names), group_attributions(dev_tp, groups, names)
        top_shap = [names[i] for i in rank(shap_tp)[:, 0]] if len(tps) else []
        top_dev = [names[i] for i in rank(dev_tp)[:, 0]] if len(tps) else []
        ablation_groups = {group_of[f] for f in ablation_set if f in group_of}
        group_rows.append({
            "detector": name, "n_true_positives": int(len(tps)),
            "top1_agreement_feature_level": top1_agreement(shap_tp, dev_tp) if len(tps) else np.nan,
            "top1_agreement_group_level": top1_agreement(g_shap, g_dev) if len(tps) else np.nan,
            "shap_top1_in_length_duration_group": float(np.mean([group_of[f] in ablation_groups for f in top_shap])) if top_shap else np.nan,
            "dev_top1_in_length_duration_group": float(np.mean([group_of[f] in ablation_groups for f in top_dev])) if top_dev else np.nan,
        })
        # disagreement subset
        if len(tps):
            mask = disagreement_mask(shap_tp, dev_tp, int(r_cfg["disagreement_top_k"]), float(r_cfg["disagreement_jaccard_threshold"]))
            ids = A["row_ids"][tps]
            ch = A["channel"][tps]
            for i, m in enumerate(mask):
                dis_rows.append({"detector": name, "anomaly_id": int(ids[i]), "channel": str(ch[i]), "disagree": bool(m),
                                 "shap_top1": top_shap[i], "dev_top1": top_dev[i]})
            if (d / "audc_per_anomaly.csv").exists():
                audc = read_csv(d / "audc_per_anomaly.csv")
                sub = audc[audc["anomaly_id"].isin(ids[mask])]
                for keys, g in sub.groupby(["explainer", "strategy", "direction"], sort=True):
                    b = bootstrap_mean(g["audc_rank"].to_numpy(), B=B, seed=seed, key=f"dis|{name}|{keys}")
                    dis_audc_rows.append({"detector": name, "subset": "disagreement", "explainer": keys[0], "strategy": keys[1], "direction": keys[2], **b})
            for k, i in enumerate(np.flatnonzero(mask)[: int(r_cfg["n_example_figures"])]):
                seg = load_segment(cfg["data"]["segments_file"], A["segment_ids"][tps[i]], cfg["data"]["columns"])
                written.append(plot_example(data.X_test[A["test_positions"][tps[i]]], {"shap": shap_tp[i], "dev": dev_tp[i]}, names,
                                            f"{name} · anomaly {int(ids[i])} ({ch[i]})", figures_dir(cfg, f"examples_{name}_{k}.png"), segment=seg))
        # false positives
        if len(fps):
            t_shap = top_features_table(A["shap"][fps], names, A["row_ids"][fps], 3, "shap_top")
            t_dev = top_features_table(A["dev"][fps], names, A["row_ids"][fps], 3, "dev_top")
            t = t_shap.merge(t_dev, on="row_id")
            t.insert(0, "detector", name)
            t["channel"] = A["channel"][fps]
            t["score"] = A["score"][fps]
            t["shap_top1_in_ablation_A"] = t["shap_top1"].isin(ablation_set)
            t["dev_top1_in_ablation_A"] = t["dev_top1"].isin(ablation_set)
            fp_rows.append(t)
            for k, i in enumerate(fps[: int(r_cfg["n_fp_example_figures"])]):
                seg = load_segment(cfg["data"]["segments_file"], A["segment_ids"][i], cfg["data"]["columns"])
                written.append(plot_example(data.X_test[A["test_positions"][i]], {"shap": A["shap"][i], "dev": A["dev"][i]}, names,
                                            f"{name} · false positive {int(A['row_ids'][i])}", figures_dir(cfg, f"fp_examples_{name}_{k}.png"), segment=seg))

    written.append(write_csv(run_dir / "group_rankings.csv", pd.DataFrame(group_rows)))
    written.append(write_csv(run_dir / "disagreement.csv", pd.DataFrame(dis_rows)))
    written.append(write_csv(run_dir / "disagreement_audc.csv", pd.DataFrame(dis_audc_rows)))
    written.append(write_csv(run_dir / "false_positives.csv", pd.concat(fp_rows, ignore_index=True) if fp_rows else pd.DataFrame()))

    # runtime table: every runtime*.json of this run
    records: list[dict[str, Any]] = []
    env: dict[str, Any] = {}
    for p in sorted(Path(results_dir(cfg)).rglob("runtime*.json")):
        j = read_json(p)
        env = j.get("environment", env)
        for r in j.get("records", []):
            records.append({"file": str(p.relative_to(resolve(cfg["paths"]["results"]))), **j.get("context", {}), **r})
    written.append(write_json(run_dir / "runtime" / "summary.json", {"environment": env, "records": records}))
    written.append(write_csv(run_dir / "runtime" / "summary.csv", pd.DataFrame(records)))

    if args.compare_run:
        frames_a, frames_b = [], []
        for name in detectors:
            pa, pb = results_dir(cfg, name) / "audc_per_anomaly.csv", resolve(Path(cfg["paths"]["results"]) / args.compare_run / name / "audc_per_anomaly.csv")
            if pa.exists() and pb.exists():
                frames_a.append(read_csv(pa))
                frames_b.append(read_csv(pb))
        if frames_a:
            written.append(plot_ablation_comparison(pd.concat(frames_a), pd.concat(frames_b), figures_dir(cfg, "ablation_comparison.png")))
    written.append(finish_manifest(manifest, run_dir, written))
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
