# opssat-xai-faithfulness

Faithfulness and Stability of Feature Attributions for Lightweight Spacecraft Telemetry Anomaly Detectors (OPSSAT-AD).

## Purpose

This repository evaluates whether SHAP attributions for lightweight anomaly detectors (Isolation Forest, One-Class SVM and a small PyTorch autoencoder) trained on the OPSSAT-AD telemetry benchmark are faithful (deletion/insertion curves under two replacement strategies), stable under 5% Gaussian input noise, and better than a model-free deviation ranking (DEV), with a random ranking (RND) as the floor. Every number comes from a CSV in `results/` written by a script with a manifest, thresholds and hyper-parameters are chosen on a validation split only, and the test split is read once. The design is frozen in the execution plan of 2026-11-05; deviations go to `docs/decisions.md`.

## Install (Windows 11, uv)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # once
uv python install 3.12
uv sync --frozen            # creates .venv from the committed uv.lock (CPU-only torch from the PyTorch index)
uv run python -c "import torch, shap, sklearn; print(torch.__version__, torch.cuda.is_available(), shap.__version__, sklearn.__version__)"
uv run pytest -q
```

`torch.cuda.is_available()` must print `False`: the environment is CPU-only by design. Never regenerate `uv.lock` on the second machine; dependency changes go through `uv add` + commit of the new lock.

## Data

Download both files of OPSSAT-AD **version 2** from <https://zenodo.org/records/15108715> into `data/raw/` (they are not redistributed here; license CC BY 4.0):

| File | MD5 |
|---|---|
| `data/raw/dataset.csv` | `5246fdc5e4630a4cecbf7fb6bc8b795e` |
| `data/raw/segments.csv` | `72f109630abb933a386106897a631188` (only used for example figures) |

Check on Windows with `certutil -hashfile data\raw\dataset.csv MD5`, then verify the loader's assertions (MD5, 2,123 rows, 18 feature columns, Table 1 counts 1273/321 train and 416/113 test):

```powershell
uv run python -m opssat_xai.data.loader --verify
```

The column names in `configs/data.yaml` (`segment`, `channel`, `anomaly`, `train`) are assumptions marked `TODO(verify)`; confirm them against the CSV header on first use and record any difference in `docs/decisions.md`.

## Running

Every script takes `--config <yaml>` (repeatable; default `configs/base.yaml`, which includes the other config files), `--set key.path=value` overrides, `--force` (overwrite an existing stage directory) and `--resume` (continue from KernelSHAP checkpoints). Each stage writes `results/<run_name>/<stage>/manifest.json` with the merged config, package versions, Python version, hostname, seeds, git commit and start/end times.

```powershell
uv run python scripts/prepare_data.py --config configs/base.yaml
uv run python scripts/run_detection.py --config configs/base.yaml
uv run python scripts/run_attributions.py --config configs/base.yaml          # --max-rows 10 for the pilot; --resume after a crash
uv run python scripts/run_faithfulness.py --config configs/base.yaml
uv run python scripts/run_stability.py --config configs/base.yaml           # --resume after a crash
uv run python scripts/run_stats.py --config configs/base.yaml
uv run python scripts/run_robustness.py --config configs/base.yaml --compare-run ablation_A
uv run python scripts/make_tables.py --config configs/base.yaml
```

One command for everything (stages run as subprocesses; `--level 1` = deletion under R1 and R2, insertion under R1, 15 stability copies, no robustness stage; `--level 2` = the full essential methodology):

```powershell
uv run python scripts/run_all.py --config configs/base.yaml --level 1
uv run python scripts/run_all.py --config configs/base.yaml --level 2
uv run python scripts/run_all.py --config configs/ablation_A.yaml --level 2   # ablation A: without len, duration, gaps_squared
```

Restrict to some detectors with `--detectors iforest ocsvm`. Compare two result trees (fresh clone, Docker) within tolerances:

```powershell
uv run python scripts/compare_results.py --reference results/main --candidate results_docker/main
```

## Docker (reference environment, CPU-only)

```powershell
docker build -t opssat-xai .
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/results_docker:/app/results opssat-xai
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/results_docker:/app/results opssat-xai uv run pytest -q
```

The image contains no data; `data/` and `results/` are mounted at run time. The default command is `uv run python scripts/run_all.py --level 1`.

## Where results land

```
data/processed/splits.json                  row indices: train-nominal / validation nominals / validation anomalies / test
data/processed/<run>/scaler.joblib          StandardScaler fitted on the train nominals only (+ feature stats, correlations)
results/<run>/detection/                    summary.csv, <det>_val.json, <det>_test_metrics.json, ocsvm_grid.csv, models, scores_<det>.npz
results/<run>/<det>/attributions_clean.npz  SHAP / DEV / RND attributions and rankings of the explained test rows
results/<run>/<det>/faithfulness_{R1,R2}_{deletion,insertion}.csv, audc_per_anomaly.csv, faithfulness_meta.json
results/<run>/<det>/stability.csv, stability_summary.csv
results/<run>/stats/                        faithfulness_summary.csv, stability_summary.csv, final_faithfulness.csv, final_stability.csv, verdicts.json
results/<run>/robustness/                   correlation_groups.json, group_rankings.csv, disagreement*.csv, false_positives.csv, runtime/
results/<run>/tables/                       Markdown and LaTeX exports of the tables above
figures/<run>/                              faithfulness curves, stability box plots, example attributions, ablation comparison
```

`<run>` is `main` for `configs/base.yaml` and `ablation_A` for `configs/ablation_A.yaml`.

## Reproducibility

All randomness derives from the single `seed` in `configs/base.yaml` through `numpy.random.SeedSequence`; every scikit-learn object receives an explicit `random_state`; the autoencoder uses `torch.manual_seed`, `torch.use_deterministic_algorithms(True)` and one thread during training; KernelSHAP seeds numpy with a per-row seed before each call.

Expected agreement across machines (encoded in `scripts/compare_results.py`):

| Output | Expected agreement |
|---|---|
| splits, scaler, Isolation Forest scores, TreeSHAP, DEV, RND, faithfulness curves and AUDC for `iforest` | bit-identical (tolerance 1e-6) |
| One-Class SVM scores | 1e-6 |
| KernelSHAP attributions (`ocsvm`, `knn`) and everything derived from them | ~1e-3 (different BLAS libraries) |
| autoencoder scores | ~1e-5 |
| GradientSHAP attributions (`autoencoder`) and everything derived from them | ~1e-3 |

## Status

| Component | Status |
|---|---|
| Loader with MD5 / row count / column / Table 1 assertions (`src/opssat_xai/data/loader.py`) | implemented; **needs manual verification** of the real CSV column names (`TODO(verify)` in `configs/data.yaml`) |
| Validation split (20% of training nominals stratified by channel + all training anomalies), `splits.json` | implemented, unit-tested on synthetic data |
| StandardScaler on train nominals only, `features.exclude` for ablation A | implemented, unit-tested |
| Isolation Forest, One-Class SVM (validation grid), kNN fallback | implemented, unit-tested (score sign matches sklearn) |
| PyTorch autoencoder (18-8-4-8-18, CPU, deterministic) | implemented, unit-tested (two fits identical, score = per-row MSE) |
| Detection metrics, bootstrap CIs, max-F1 threshold, 5%-FPR secondary threshold, chance-level rule | implemented, unit-tested |
| TreeSHAP + additivity check | implemented, unit-tested; private sklearn import marked `TODO(verify)` |
| KernelSHAP with k-means background, per-row seeds, checkpoint/resume | implemented, unit-tested on a toy model; **needs the Nov 7 timing pilot** on real OC-SVM |
| GradientSHAP for the autoencoder + gradient-vs-kernel comparison | implemented, smoke-tested; which one to use is decided from `gradient_vs_kernel.csv` (`TODO(verify)` in `configs/explainers.yaml`) |
| DEV, RND, signed / absolute rankings | implemented, unit-tested |
| R1 / R2 replacement, batched deletion / insertion curves, AUDC | implemented, unit-tested |
| Rank / z normalization against validation nominals, saturation check | implemented, unit-tested |
| Stability perturbation, class-flip filter, Jaccard / Spearman / relative L2 | implemented, unit-tested |
| Bootstrap, paired comparisons, Wilcoxon, Cohen's d_z, Cliff's delta, Holm, verdict rules, required n | implemented, unit-tested |
| Correlation groups, group-level rankings, disagreement and false-positive subsets, runtime logging | implemented; **not yet run on real data** |
| Scripts `prepare_data` ... `run_robustness`, `run_all`, `compare_results`, `make_tables` | implemented; exercised only on synthetic data (see `docs/reproduction_log.md` for real runs) |
| Example figures with raw segments from `segments.csv` | implemented; **needs manual verification** of the `segments.csv` columns (`TODO(verify)` in `plotting/examples.py`) |
| Dockerfile | written; **build not yet verified** (record the image digest in `docs/reproduction_log.md`) |

No experimental results are reported here by design; see `docs/` once runs exist.
