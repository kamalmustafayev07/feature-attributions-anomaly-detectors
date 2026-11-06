"""Export result tables as Markdown and LaTeX so that no number in the paper is typed by hand."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from opssat_xai.io import resolve


def _fmt(v: object, digits: int) -> str:
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.{digits}f}"
    if v is None:
        return ""
    return str(v)


def df_to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    """GitHub-flavoured Markdown table (no external dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def df_to_latex(df: pd.DataFrame, digits: int = 3, caption: str | None = None, label: str | None = None) -> str:
    """Minimal LaTeX ``tabular`` (booktabs-style rules) without pandas' jinja2 dependency."""
    cols = [str(c).replace("_", "\\_") for c in df.columns]
    lines = ["\\begin{table}[t]", "\\centering", "\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule",
             " & ".join(cols) + " \\\\", "\\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(_fmt(v, digits).replace("_", "\\_") for v in row.tolist()) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, path_stem: str | Path, digits: int = 3, caption: str | None = None, label: str | None = None) -> list[Path]:
    """Write ``<stem>.md`` and ``<stem>.tex``; returns the written paths."""
    stem = resolve(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    md = stem.with_suffix(".md")
    tex = stem.with_suffix(".tex")
    md.write_text(df_to_markdown(df, digits), encoding="utf-8")
    tex.write_text(df_to_latex(df, digits, caption, label), encoding="utf-8")
    return [md, tex]
