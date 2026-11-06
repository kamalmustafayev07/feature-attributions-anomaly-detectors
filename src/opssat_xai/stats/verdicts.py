"""Decision rules frozen on 2026-11-05 (execution plan, section 1.2 #10). Nothing here hard-codes a result.

* SHAP is *faithful* (Q1) if it beats RND under BOTH R1 and R2 with the paired CI excluding 0.
* SHAP is *better* than DEV if the mean advantage >= margin (0.05) AND the CI excludes 0.
* *equivalent* if the CI lies entirely inside (-margin, +margin).
* *worse* (symmetric rule) if the mean disadvantage >= margin AND the CI excludes 0.
* otherwise *inconclusive*, reported with the n needed for a +/-margin CI half-width from the observed SD.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class Verdict(str, Enum):
    """Outcome of a paired comparison (difference oriented so that positive favours SHAP)."""

    BETTER = "better"
    WORSE = "worse"
    EQUIVALENT = "equivalent"
    INCONCLUSIVE = "inconclusive"


def comparison_verdict(mean_diff: float, ci_low: float, ci_high: float, margin: float = 0.05) -> Verdict:
    """Apply the frozen rules to one paired comparison."""
    if not all(np.isfinite([mean_diff, ci_low, ci_high])):
        return Verdict.INCONCLUSIVE
    if mean_diff >= margin and ci_low > 0:
        return Verdict.BETTER
    if mean_diff <= -margin and ci_high < 0:
        return Verdict.WORSE
    if ci_low > -margin and ci_high < margin:
        return Verdict.EQUIVALENT
    return Verdict.INCONCLUSIVE


def beats_floor(mean_diff: float, ci_low: float) -> bool:
    """True if the advantage over the floor is positive with a CI excluding 0."""
    return bool(np.isfinite(mean_diff) and np.isfinite(ci_low) and mean_diff > 0 and ci_low > 0)


def faithful_verdict(by_strategy: dict[str, tuple[float, float, float]]) -> str:
    """``'faithful'`` iff SHAP beats RND under BOTH R1 and R2; input maps strategy -> (mean_diff, ci_low, ci_high)."""
    if not {"R1", "R2"} <= set(by_strategy):
        return "inconclusive"
    return "faithful" if all(beats_floor(by_strategy[s][0], by_strategy[s][1]) for s in ("R1", "R2")) else "not_faithful"


def required_n(sd: float, half_width: float = 0.05, z: float = 1.959964) -> int | None:
    """Rows needed for a normal-approximation CI half-width of ``half_width`` given the observed SD of differences."""
    if sd is None or not np.isfinite(sd) or sd <= 0 or half_width <= 0:
        return None
    return int(math.ceil((z * sd / half_width) ** 2))


def add_verdicts(table: pd.DataFrame, margin: float = 0.05) -> pd.DataFrame:
    """Add ``verdict`` and ``required_n`` columns to a comparison table from stats.paired."""
    out = table.copy()
    if not len(out):
        out["verdict"] = []
        out["required_n"] = []
        return out
    out["verdict"] = [comparison_verdict(m, lo, hi, margin).value for m, lo, hi in zip(out["mean_diff"], out["ci_low"], out["ci_high"])]
    out["required_n"] = [required_n(sd, margin) if v == Verdict.INCONCLUSIVE.value else None for sd, v in zip(out["sd_diff"], out["verdict"])]
    return out


def research_question_verdicts(faith: pd.DataFrame, stab: pd.DataFrame, margin: float = 0.05, metric: str = "audc_rank") -> dict[str, Any]:
    """Q1 (faithful vs RND, deletion, both strategies), Q2/Q3 summaries per detector from the comparison tables."""
    out: dict[str, Any] = {"metric": metric, "margin": margin, "detectors": {}}
    for detector in sorted(set(faith.get("detector", pd.Series(dtype=str))) | set(stab.get("detector", pd.Series(dtype=str)))):
        f = faith[(faith["detector"] == detector) & (faith["metric"] == metric)] if len(faith) else faith
        entry: dict[str, Any] = {}
        rnd_del = f[(f["comparison"] == "shap_vs_rnd") & (f["direction"] == "deletion")] if len(f) else f
        by_strategy = {r["strategy"]: (r["mean_diff"], r["ci_low"], r["ci_high"]) for _, r in rnd_del.iterrows()}
        entry["Q1_faithful_vs_rnd_deletion"] = faithful_verdict(by_strategy)
        dev = f[f["comparison"] == "shap_vs_dev"] if len(f) else f
        entry["Q3a_vs_dev_faithfulness"] = {
            f"{r['strategy']}/{r['direction']}": comparison_verdict(r["mean_diff"], r["ci_low"], r["ci_high"], margin).value
            for _, r in dev.iterrows()
        }
        s = stab[stab["detector"] == detector] if len(stab) else stab
        entry["Q2_Q3b_stability_vs_baselines"] = {
            f"{r['metric']}/{r['comparison']}": comparison_verdict(r["mean_diff"], r["ci_low"], r["ci_high"], margin).value
            for _, r in s.iterrows()
        }
        out["detectors"][detector] = entry
    return out
