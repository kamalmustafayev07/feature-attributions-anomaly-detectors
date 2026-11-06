"""Paired SHAP-vs-DEV and SHAP-vs-RND comparisons per anomaly: mean difference (positive favours SHAP) with a
paired bootstrap CI, Wilcoxon signed-rank p, paired Cohen's d_z, Cliff's delta, and Holm correction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from opssat_xai.stats.bootstrap import bootstrap_mean


def cohen_dz(diff: np.ndarray) -> float:
    """Paired Cohen's d_z = mean(diff) / sd(diff)."""
    d = np.asarray(diff, dtype=np.float64).reshape(-1)
    sd = d.std(ddof=1) if d.size > 1 else float("nan")
    return float(d.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b) over all pairs (positive = ``a`` tends to be larger)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 1)
    b = np.asarray(b, dtype=np.float64).reshape(1, -1)
    return float(np.mean(np.sign(a - b)))


def wilcoxon_p(diff: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value on the paired differences (1.0 if all differences are zero)."""
    d = np.asarray(diff, dtype=np.float64).reshape(-1)
    if np.all(d == 0) or d.size < 2:
        return 1.0
    return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def paired_comparison(
    shap_values: np.ndarray, other_values: np.ndarray, higher_is_better: bool, B: int, seed: int, alpha: float, key: str
) -> dict[str, Any]:
    """Compare SHAP with another explainer on paired per-anomaly values; the difference is oriented so that
    POSITIVE FAVOURS SHAP (``other - shap`` when lower is better, ``shap - other`` when higher is better)."""
    s = np.asarray(shap_values, dtype=np.float64).reshape(-1)
    o = np.asarray(other_values, dtype=np.float64).reshape(-1)
    ok = np.isfinite(s) & np.isfinite(o)
    s, o = s[ok], o[ok]
    diff = (s - o) if higher_is_better else (o - s)
    boot = bootstrap_mean(diff, B=B, seed=seed, alpha=alpha, key=key)
    favourable = (s, o) if higher_is_better else (o, s)
    return {
        "n": int(diff.size),
        "mean_shap": float(s.mean()) if s.size else float("nan"),
        "mean_other": float(o.mean()) if o.size else float("nan"),
        "mean_diff": boot["mean"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "sd_diff": boot["sd"],
        "wilcoxon_p": wilcoxon_p(diff),
        "cohen_dz": cohen_dz(diff),
        "cliffs_delta": cliffs_delta(*favourable),
    }


def holm_correction(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (NaNs are left as NaN and not counted)."""
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    m = valid.size
    if m == 0:
        return adjusted
    order = valid[np.argsort(p[valid], kind="stable")]
    running = 0.0
    for rank_i, idx in enumerate(order):
        running = max(running, (m - rank_i) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def faithfulness_comparisons(
    audc_df: pd.DataFrame, value_col: str, B: int, seed: int, alpha: float, baselines: tuple[str, ...] = ("dev", "rnd")
) -> pd.DataFrame:
    """One row per (detector, strategy, direction, baseline): SHAP vs baseline on per-anomaly AUDC.

    Deletion AUDC: lower is better. Insertion AUDC: higher is better.
    """
    rows: list[dict[str, Any]] = []
    for (detector, strategy, direction), g in audc_df.groupby(["detector", "strategy", "direction"], sort=True):
        wide = g.pivot_table(index="anomaly_id", columns="explainer", values=value_col)
        if "shap" not in wide.columns:
            continue
        higher = direction == "insertion"
        for base in baselines:
            if base not in wide.columns:
                continue
            pair = wide[["shap", base]].dropna()
            res = paired_comparison(pair["shap"].to_numpy(), pair[base].to_numpy(), higher, B, seed, alpha,
                                    key=f"{detector}|{strategy}|{direction}|{value_col}|{base}")
            rows.append({"detector": detector, "metric": value_col, "strategy": strategy, "direction": direction,
                         "comparison": f"shap_vs_{base}", "higher_is_better": higher, **res})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm_correction(table["wilcoxon_p"].to_numpy())
    return table


def stability_comparisons(
    per_anomaly: pd.DataFrame, B: int, seed: int, alpha: float,
    metrics: tuple[str, ...] = ("jaccard3", "jaccard5", "spearman", "rel_l2"), baselines: tuple[str, ...] = ("dev", "rnd"),
) -> pd.DataFrame:
    """One row per (detector, metric, baseline): SHAP vs baseline on per-anomaly stability means."""
    rows: list[dict[str, Any]] = []
    for detector, g in per_anomaly.groupby("detector", sort=True):
        for metric in metrics:
            wide = g.pivot_table(index="anomaly_id", columns="explainer", values=metric)
            if "shap" not in wide.columns:
                continue
            higher = metric != "rel_l2"
            for base in baselines:
                if base not in wide.columns:
                    continue
                pair = wide[["shap", base]].dropna()
                res = paired_comparison(pair["shap"].to_numpy(), pair[base].to_numpy(), higher, B, seed, alpha,
                                        key=f"{detector}|stability|{metric}|{base}")
                rows.append({"detector": detector, "metric": metric, "strategy": "", "direction": "",
                             "comparison": f"shap_vs_{base}", "higher_is_better": higher, **res})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm_correction(table["wilcoxon_p"].to_numpy())
    return table
