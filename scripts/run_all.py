"""Single entry point: prepare_data -> run_detection -> run_attributions -> run_faithfulness -> run_stability ->
run_stats -> run_robustness for every detector in ``detectors_in_scope``.

``--level 1`` runs deletion under R1 and R2, insertion under R1 only, 15 stability copies and no robustness stage.
Each stage runs as a subprocess with the same interpreter and writes its own manifest; this script writes
results/<run_name>/manifest.json and prints the files written, nothing else.

Usage: uv run python scripts/run_all.py --config configs/base.yaml [--level 1|2] [--force] [--resume]
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from opssat_xai.config import build_parser, config_from_args, figures_dir, processed_dir, results_dir
from opssat_xai.io import finish_manifest, relative_to_root, repo_root, resolve, start_manifest

STAGES = ["prepare_data", "run_detection", "run_attributions", "run_faithfulness", "run_stability", "run_stats", "run_robustness"]


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser("Run the whole pipeline.")
    parser.add_argument("--level", type=int, choices=(1, 2), default=None, help="1 = minimum subset, 2 = full (default: config 'level')")
    parser.add_argument("--detectors", nargs="+", default=None, help="subset of detectors_in_scope")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    level = args.level or int(cfg.get("level", 2))
    overrides = list(args.overrides) + [f"level={level}"]
    if level == 1:
        plan = cfg["faithfulness"]["level1_plan"]
        overrides.append("faithfulness.plan=" + "{" + ", ".join(f"{k}: [{', '.join(v)}]" for k, v in plan.items()) + "}")
        overrides.append("stability.n_copies=15")
    cfg_after = config_from_args(type(args)(**{**vars(args), "overrides": overrides}))
    manifest = start_manifest(cfg_after, "scripts/run_all.py")
    start = time.time()

    common = []
    for c in (args.config or []):
        common += ["--config", c]
    for o in overrides:
        common += ["--set", o]
    if args.force:
        common.append("--force")
    if args.resume:
        common.append("--resume")
    det_args = ["--detectors", *args.detectors] if args.detectors else []
    stages = [s for s in STAGES if not (level == 1 and s == "run_robustness")]
    for stage in stages:
        cmd = [sys.executable, str(repo_root() / "scripts" / f"{stage}.py"), *common]
        if stage not in ("prepare_data",):
            cmd += det_args
        result = subprocess.run(cmd, cwd=repo_root(), stdout=subprocess.DEVNULL)
        if result.returncode != 0:
            print(f"stage {stage} failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode

    run_root = results_dir(cfg_after)
    finish_manifest(manifest, run_root, [])
    written = sorted(
        p for root in (run_root, figures_dir(cfg_after), processed_dir(cfg_after), resolve(cfg_after["splits"]["splits_file"]).parent)
        for p in Path(root).rglob("*") if p.is_file() and p.stat().st_mtime >= start - 1
    )
    for p in written:
        print(relative_to_root(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
