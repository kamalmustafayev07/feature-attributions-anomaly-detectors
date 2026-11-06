"""Matplotlib figures (Agg backend) and table export. Colors follow the explainer, never its rank in a legend."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

# Fixed hue per explainer (validated categorical slots: blue, orange, aqua); gray for anything else.
EXPLAINER_COLORS: dict[str, str] = {"shap": "#2a78d6", "dev": "#eb6834", "rnd": "#1baf7a"}
EXPLAINER_LABELS: dict[str, str] = {"shap": "SHAP", "dev": "DEV", "rnd": "RND"}
NEUTRAL = "#7a7a76"


def explainer_color(name: str) -> str:
    """Color for an explainer name (gray if unknown)."""
    return EXPLAINER_COLORS.get(name, NEUTRAL)


def style_axes(ax: "matplotlib.axes.Axes") -> None:
    """Recessive grid and spines."""
    ax.grid(True, color="#e3e2dd", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
