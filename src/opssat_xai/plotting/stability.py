"""Stability box plots: one panel per metric, one box per explainer, per-anomaly means as the samples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from opssat_xai.io import resolve
from opssat_xai.plotting import EXPLAINER_LABELS, explainer_color, style_axes
from opssat_xai.stability.metrics import per_anomaly_means

METRIC_LABELS = {"jaccard3": "Jaccard@3", "jaccard5": "Jaccard@5", "spearman": "Spearman", "rel_l2": "relative L2"}


def plot_stability_boxes(stability_df: pd.DataFrame, detector: str, out_path: str | Path,
                         metrics: tuple[str, ...] = ("jaccard3", "jaccard5", "spearman", "rel_l2")) -> Path:
    """Box plots of per-anomaly mean stability metrics per explainer for one detector."""
    per = per_anomaly_means(stability_df[stability_df["detector"] == detector], metrics)
    explainers = [e for e in ("shap", "dev", "rnd") if e in set(per["explainer"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        data = [per[per["explainer"] == e][metric].dropna().to_numpy() for e in explainers]
        box = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                         flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
                         medianprops={"color": "#1a1a19", "linewidth": 1.2})
        for patch, e in zip(box["boxes"], explainers):
            patch.set_facecolor(explainer_color(e))
            patch.set_alpha(0.35)
            patch.set_edgecolor(explainer_color(e))
        ax.set_xticks(range(1, len(explainers) + 1))
        ax.set_xticklabels([EXPLAINER_LABELS.get(e, e) for e in explainers])
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        style_axes(ax)
    fig.suptitle(f"{detector}: stability under 5% Gaussian noise (per-anomaly means over retained copies)", fontsize=10)
    fig.tight_layout()
    p = resolve(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
