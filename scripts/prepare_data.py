"""Stage 1: validate dataset.csv, build the splits, fit the scaler on the train nominals, write feature stats.

Writes: data/processed/splits.json (identical for every run), data/processed/<run>/scaler.joblib,
feature_stats_train_nominal.csv, correlation_train_nominal.csv, split_counts.json,
results/<run>/prepare_data/manifest.json.

Usage: uv run python scripts/prepare_data.py --config configs/base.yaml [--force]
"""

from __future__ import annotations

import sys

from opssat_xai.config import active_feature_names, build_parser, config_from_args, processed_dir, results_dir
from opssat_xai.data.loader import dataset_arrays, load_dataset, md5sum
from opssat_xai.data.preprocess import feature_stats, fit_scaler, save_scaler
from opssat_xai.data.splits import load_splits, make_splits, save_splits, splits_equal
from opssat_xai.io import finish_manifest, prepare_run_dir, resolve, start_manifest, write_csv, write_json
from opssat_xai.logging_utils import get_logger
from opssat_xai.robustness.correlation import spearman_matrix

log = get_logger("opssat_xai.prepare_data")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser("Validate OPSSAT-AD, build splits, fit the train-nominal scaler.")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    run_dir = prepare_run_dir(results_dir(cfg, "prepare_data"), force=args.force, resume=args.resume)
    out_dir = prepare_run_dir(processed_dir(cfg), force=args.force, resume=args.resume)
    manifest = start_manifest(cfg, "scripts/prepare_data.py")
    written: list[object] = []

    df = load_dataset(cfg)
    manifest["dataset_md5"] = md5sum(cfg["data"]["dataset_file"])
    names = active_feature_names(cfg)
    arrays = dataset_arrays(df, cfg, names)
    log.info("dataset ok: %d rows, %d active features (%d excluded)", len(df), len(names), len(cfg["features"]["names"]) - len(names))

    splits = make_splits(arrays.y, arrays.train, arrays.channel, cfg, seed=int(cfg["seed"]))
    splits_path = resolve(cfg["splits"]["splits_file"])
    if splits_path.exists():
        existing = load_splits(splits_path)
        if not splits_equal(existing, splits):
            if not args.force:
                raise FileExistsError(f"{splits_path} exists with DIFFERENT indices; pass --force to overwrite")
            log.warning("overwriting %s with different indices (--force)", splits_path)
        else:
            log.info("existing %s reproduced bit-identically", splits_path)
    written.append(save_splits(splits, splits_path))
    log.info("splits: %s", splits.counts)

    X_tn = arrays.X[splits.train_nominal]
    scaler = fit_scaler(X_tn, arrays.y[splits.train_nominal])
    written.append(save_scaler(scaler, out_dir / "scaler.joblib"))
    written.append(write_csv(out_dir / "feature_stats_train_nominal.csv", feature_stats(X_tn, names)))
    corr = spearman_matrix(X_tn, names)
    written.append(write_csv(out_dir / "correlation_train_nominal.csv", corr.reset_index().rename(columns={"index": "feature"})))
    written.append(write_json(out_dir / "split_counts.json", {
        "table1": {"train_nominal": int(((arrays.y == 0) & arrays.train).sum()), "train_anomaly": int(((arrays.y == 1) & arrays.train).sum()),
                   "test_nominal": int(((arrays.y == 0) & ~arrays.train).sum()), "test_anomaly": int(((arrays.y == 1) & ~arrays.train).sum())},
        "splits": splits.counts,
        "val_anomaly_rate": float(splits.counts["val_anomaly"] / splits.counts["val"]),
        "feature_names": names,
        "excluded": list(cfg["features"].get("exclude") or []),
    }))
    written.append(finish_manifest(manifest, run_dir, written))
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
