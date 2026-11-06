"""Export the result CSVs of a run as Markdown and LaTeX tables (results/<run>/tables/) so no number is typed by hand.

Usage: uv run python scripts/make_tables.py --config configs/base.yaml [--digits 3]
"""

from __future__ import annotations

import sys

from opssat_xai.config import build_parser, config_from_args, results_dir
from opssat_xai.io import read_csv
from opssat_xai.plotting.tables import write_table

TABLES = {
    "detection_summary": ("detection/summary.csv", "Detection quality on the official test split (threshold chosen on validation)."),
    "faithfulness_summary": ("stats/faithfulness_summary.csv", "Mean AUDC per explainer with 95% bootstrap CIs."),
    "final_faithfulness": ("stats/final_faithfulness.csv", "Paired comparisons of SHAP against DEV and RND on per-anomaly AUDC (true positives)."),
    "stability_summary": ("stats/stability_summary.csv", "Stability metrics per explainer with 95% bootstrap CIs and retention rate."),
    "final_stability": ("stats/final_stability.csv", "Paired comparisons of SHAP against DEV and RND on stability metrics."),
    "group_rankings": ("robustness/group_rankings.csv", "Group-level agreement between SHAP and DEV."),
    "false_positives": ("robustness/false_positives.csv", "Top features of test false positives."),
    "runtime": ("robustness/runtime/summary.csv", "Runtime and peak RSS per stage."),
}


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser("Export result tables as Markdown and LaTeX.")
    parser.add_argument("--digits", type=int, default=3)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    out_dir = results_dir(cfg, "tables")
    written = []
    for name, (rel, caption) in TABLES.items():
        src = results_dir(cfg, *rel.split("/"))
        if not src.exists():
            continue
        df = read_csv(src)
        if len(df):
            written += write_table(df, out_dir / name, digits=args.digits, caption=caption, label=f"tab:{name}")
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
