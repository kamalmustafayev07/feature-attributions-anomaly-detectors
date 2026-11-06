# Figures and caption drafts

Every figure that appears in the paper is listed here with the script that produced it, the results file it reads,
and a caption draft. Do not add a figure that is not referenced in the paper text.

| Figure file | Produced by | Reads | Caption draft |
|---|---|---|---|
| figures/<run>/faithfulness_<detector>_<strategy>.png | scripts/run_faithfulness.py | results/<run>/<detector>/faithfulness_<strategy>_{deletion,insertion}.csv | (to write) |
| figures/<run>/stability_<detector>.png | scripts/run_stability.py | results/<run>/<detector>/stability.csv | (to write) |
| figures/<run>/examples_<detector>_<i>.png | scripts/run_robustness.py | results/<run>/<detector>/attributions_clean.npz | (to write) |
| figures/<run>/ablation_comparison.png | scripts/run_robustness.py | results/main and results/ablation_A audc_per_anomaly.csv | (to write) |
