"""Deletion / insertion curves: mean normalized score per k with a bootstrap band, three explainers on one axis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opssat_xai.io import resolve
from opssat_xai.plotting import EXPLAINER_LABELS, explainer_color, style_axes
from opssat_xai.stats.bootstrap import bootstrap_mean


def mean_curve_with_band(long_df: pd.DataFrame, score_col: str, B: int, seed: int) -> pd.DataFrame:
    """Per (explainer, k): mean over anomalies (repeats averaged first) with a percentile bootstrap band."""
    per_anomaly = long_df.groupby(["explainer", "anomaly_id", "k"], as_index=False)[score_col].mean()
    rows = []
    for (explainer, k), g in per_anomaly.groupby(["explainer", "k"], sort=True):
        b = bootstrap_mean(g[score_col].to_numpy(), B=B, seed=seed, key=f"curve|{explainer}|{k}")
        rows.append({"explainer": explainer, "k": int(k), "mean": b["mean"], "ci_low": b["ci_low"], "ci_high": b["ci_high"], "n": b["n"]})
    return pd.DataFrame(rows)


def plot_faithfulness_curves(
    long_df: pd.DataFrame, detector: str, strategy: str, out_path: str | Path, score_col: str = "score_rank",
    B: int = 500, seed: int = 0, y_range: tuple[float, float] | None = None,
) -> Path:
    """One figure with deletion and insertion panels for one detector and strategy."""
    directions = [d for d in ("deletion", "insertion") if d in set(long_df["direction"])]
    fig, axes = plt.subplots(1, len(directions), figsize=(4.2 * len(directions), 3.4), sharey=True, squeeze=False)
    K = int(long_df["k"].max())
    for ax, direction in zip(axes[0], directions):
        sub = long_df[(long_df["direction"] == direction)]
        curves = mean_curve_with_band(sub, score_col, B, seed)
        for explainer in ("shap", "dev", "rnd"):
            c = curves[curves["explainer"] == explainer].sort_values("k")
            if c.empty:
                continue
            x = c["k"].to_numpy() / K
            color = explainer_color(explainer)
            ax.fill_between(x, c["ci_low"], c["ci_high"], color=color, alpha=0.15, linewidth=0)
            ax.plot(x, c["mean"], color=color, linewidth=2, label=EXPLAINER_LABELS.get(explainer, explainer))
        ax.set_title(f"{detector} · {strategy} · {direction}", fontsize=10)
        ax.set_xlabel("fraction of features replaced (k/d)" if direction == "deletion" else "fraction of features reinstated (k/d)")
        style_axes(ax)
        if y_range:
            ax.set_ylim(*y_range)
    axes[0][0].set_ylabel({"score_rank": "rank-normalized score", "score_z": "z-normalized score"}.get(score_col, score_col))
    axes[0][-1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = resolve(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_ablation_comparison(audc_main: pd.DataFrame, audc_ablation: pd.DataFrame, out_path: str | Path, value_col: str = "audc_rank") -> Path:
    """Paired dots: mean AUDC per explainer in the main run vs ablation A, one panel per (strategy, direction)."""
    keys = ["detector", "strategy", "direction", "explainer"]
    a = audc_main.groupby(keys)[value_col].mean().rename("main")
    b = audc_ablation.groupby(keys)[value_col].mean().rename("ablation_A")
    both = pd.concat([a, b], axis=1).reset_index()
    panels = both[["strategy", "direction"]].drop_duplicates().sort_values(["strategy", "direction"]).to_records(index=False)
    fig, axes = plt.subplots(1, max(1, len(panels)), figsize=(3.6 * max(1, len(panels)), 3.4), squeeze=False)
    for ax, (strategy, direction) in zip(axes[0], panels):
        sub = both[(both["strategy"] == strategy) & (both["direction"] == direction)]
        detectors = sorted(sub["detector"].unique())
        xs = np.arange(len(detectors))
        for j, explainer in enumerate(("shap", "dev", "rnd")):
            s = sub[sub["explainer"] == explainer].set_index("detector").reindex(detectors)
            if s["main"].isna().all():
                continue
            color = explainer_color(explainer)
            off = (j - 1) * 0.18
            ax.scatter(xs + off, s["main"], color=color, s=36, label=f"{EXPLAINER_LABELS[explainer]} main", zorder=3)
            ax.scatter(xs + off, s["ablation_A"], facecolors="none", edgecolors=color, s=36, label=f"{EXPLAINER_LABELS[explainer]} ablation A", zorder=3)
            for xi, m, ab in zip(xs + off, s["main"], s["ablation_A"]):
                ax.plot([xi, xi], [m, ab], color=color, linewidth=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(detectors)
        ax.set_title(f"{strategy} · {direction}", fontsize=10)
        style_axes(ax)
    axes[0][0].set_ylabel(f"mean {value_col}")
    axes[0][-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    p = resolve(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
