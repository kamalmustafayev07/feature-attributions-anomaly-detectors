"""Example attribution bar charts (SHAP vs DEV for one segment), optionally with the raw segment from segments.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opssat_xai.io import resolve
from opssat_xai.plotting import EXPLAINER_LABELS, explainer_color, style_axes


def load_segment(segments_csv: str | Path, segment_id: object, cfg_columns: dict[str, str]) -> pd.DataFrame | None:
    """Rows of segments.csv for one segment id, or ``None`` if the file or id is missing.

    TODO(verify): segments.csv column names (segment id, timestamp, value) are unknown until inspected on 2026-11-06.
    """
    p = resolve(segments_csv)
    if not p.exists():
        return None
    seg_col = cfg_columns.get("segment_id", "segment")
    df = pd.read_csv(p)
    if seg_col not in df.columns:
        return None
    sub = df[df[seg_col] == segment_id]
    return sub if len(sub) else None


def plot_example(
    x_std: np.ndarray, attributions: dict[str, np.ndarray], feature_names: list[str], title: str, out_path: str | Path,
    segment: pd.DataFrame | None = None, value_col: str = "value", time_col: str = "timestamp",
) -> Path:
    """Horizontal bars of attribution per feature for each explainer in ``attributions``; raw segment panel if given."""
    d = len(feature_names)
    n_panels = 1 + (1 if segment is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 0.28 * d + 1.6), squeeze=False)
    ax = axes[0][0]
    order = np.argsort(-np.abs(x_std))
    names = [feature_names[i] for i in order]
    ys = np.arange(d)
    width = 0.8 / max(1, len(attributions))
    for j, (explainer, attr) in enumerate(attributions.items()):
        vals = np.asarray(attr).reshape(-1)[order]
        ax.barh(ys + (j - (len(attributions) - 1) / 2) * width, vals, height=width * 0.9,
                color=explainer_color(explainer), label=EXPLAINER_LABELS.get(explainer, explainer))
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#b5b4ae", linewidth=0.8)
    ax.set_xlabel("attribution (positive = pushes anomaly score up)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    if segment is not None:
        ax2 = axes[0][1]
        xcol = time_col if time_col in segment.columns else None
        if value_col in segment.columns:
            ax2.plot(segment[xcol] if xcol else np.arange(len(segment)), segment[value_col], color="#1a1a19", linewidth=1.2)
        ax2.set_title("raw telemetry segment", fontsize=10)
        style_axes(ax2)
    fig.tight_layout()
    p = resolve(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
