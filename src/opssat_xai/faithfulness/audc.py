"""Area under the (normalized) deletion / insertion curve: trapezoidal mean over k/K."""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_KEYS = ["anomaly_id", "detector", "explainer", "strategy", "direction"]


def audc(values: np.ndarray) -> float:
    """``(1/K) * sum_{k=0}^{K-1} (v_k + v_{k+1}) / 2`` for a curve sampled at k = 0..K."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.shape[0] < 2:
        raise ValueError("a curve needs at least two points (k = 0 and k = K)")
    K = v.shape[0] - 1
    return float(np.sum((v[:-1] + v[1:]) / 2.0) / K)


def audc_table(long_df: pd.DataFrame, score_cols: tuple[str, ...] = ("score_rank", "score_z", "score_raw")) -> pd.DataFrame:
    """Per-anomaly AUDC from long-format records; curves are averaged over ``repeat`` before the area is taken."""
    required = set(GROUP_KEYS) | {"repeat", "k"} | set(score_cols)
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"long-format table is missing columns {sorted(missing)}")
    mean_over_repeats = (
        long_df.groupby(GROUP_KEYS + ["k"], sort=True, as_index=False)
        .agg({**{c: "mean" for c in score_cols}, "repeat": "nunique"})
        .rename(columns={"repeat": "n_repeats"})
    )
    rows = []
    for keys, g in mean_over_repeats.groupby(GROUP_KEYS, sort=True):
        g = g.sort_values("k")
        row = dict(zip(GROUP_KEYS, keys))
        row["n_repeats"] = int(g["n_repeats"].iloc[0])
        row["K"] = int(g["k"].max())
        for c in score_cols:
            row["audc_" + c.replace("score_", "")] = audc(g[c].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)
