import pandas as pd

from opssat_xai.stats.verdicts import Verdict, add_verdicts, comparison_verdict, faithful_verdict, required_n, research_question_verdicts


def test_four_outcomes():
    assert comparison_verdict(0.08, 0.02, 0.14) == Verdict.BETTER
    assert comparison_verdict(-0.08, -0.14, -0.02) == Verdict.WORSE
    assert comparison_verdict(0.01, -0.03, 0.04) == Verdict.EQUIVALENT
    assert comparison_verdict(0.03, -0.02, 0.08) == Verdict.INCONCLUSIVE
    assert comparison_verdict(0.08, -0.01, 0.17) == Verdict.INCONCLUSIVE  # large mean but CI includes 0
    assert comparison_verdict(0.02, 0.005, 0.035) == Verdict.EQUIVALENT  # CI excludes 0 but inside the margin
    assert comparison_verdict(float("nan"), 0.0, 0.1) == Verdict.INCONCLUSIVE


def test_faithful_requires_both_strategies():
    assert faithful_verdict({"R1": (0.1, 0.05, 0.15), "R2": (0.2, 0.1, 0.3)}) == "faithful"
    assert faithful_verdict({"R1": (0.1, 0.05, 0.15), "R2": (0.02, -0.01, 0.05)}) == "not_faithful"
    assert faithful_verdict({"R1": (0.1, 0.05, 0.15)}) == "inconclusive"


def test_required_n():
    assert required_n(0.2, 0.05) == 62  # ceil((1.96 * 0.2 / 0.05)^2) = ceil(61.5)
    assert required_n(0.0) is None and required_n(float("nan")) is None


def test_add_verdicts_and_rq():
    table = pd.DataFrame({
        "detector": ["d"] * 3, "metric": ["audc_rank"] * 3, "strategy": ["R1", "R2", "R1"], "direction": ["deletion"] * 3,
        "comparison": ["shap_vs_rnd", "shap_vs_rnd", "shap_vs_dev"],
        "mean_diff": [0.2, 0.2, 0.03], "ci_low": [0.1, 0.1, -0.02], "ci_high": [0.3, 0.3, 0.08], "sd_diff": [0.1, 0.1, 0.2],
    })
    out = add_verdicts(table)
    assert out["verdict"].tolist() == ["better", "better", "inconclusive"]
    assert out["required_n"].iloc[:2].isna().all() and out["required_n"].iloc[2] == 62
    rq = research_question_verdicts(out, pd.DataFrame(), margin=0.05)
    assert rq["detectors"]["d"]["Q1_faithful_vs_rnd_deletion"] == "faithful"
    assert rq["detectors"]["d"]["Q3a_vs_dev_faithfulness"]["R1/deletion"] == "inconclusive"
