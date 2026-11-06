"""Compare two results directories (fresh-clone / Docker reproduction) within per-file tolerances.

Tolerances: 1e-6 (absolute) for everything, except files under the KernelSHAP / GradientSHAP detectors
(ocsvm, knn, autoencoder) which get 1e-3. Volatile manifest keys (times, hostname, git commit, paths) are ignored.

Usage: uv run python scripts/compare_results.py --reference results/main --candidate results_docker/main [--tol 1e-6] [--tol-shap 1e-3]
Exit code 0 = all files match, 1 = mismatches (listed), 2 = usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from opssat_xai.io import resolve

SHAP_DIRS = ("ocsvm", "knn", "autoencoder")
VOLATILE_KEYS = {"start_time", "end_time", "hostname", "git_commit", "platform", "python_version", "files_written", "file", "environment", "seconds", "seconds_per_item", "peak_rss_bytes", "config"}


def _tol_for(rel: Path, tol: float, tol_shap: float) -> float:
    return tol_shap if any(part in SHAP_DIRS for part in rel.parts) else tol


def _strip(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _json_equal(a: object, b: object, tol: float) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return bool(np.isclose(a, b, rtol=0, atol=tol, equal_nan=True))
    return a == b


def compare_file(ref: Path, cand: Path, tol: float) -> str | None:
    """Return ``None`` if equal within ``tol``, else a short reason."""
    if not cand.exists():
        return "missing in candidate"
    if ref.suffix == ".csv":
        a, b = pd.read_csv(ref), pd.read_csv(cand)
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            return f"shape/columns differ ({a.shape} vs {b.shape})"
        for col in a.columns:
            if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
                if not np.allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float), rtol=0, atol=tol, equal_nan=True):
                    diff = np.nanmax(np.abs(a[col].to_numpy(dtype=float) - b[col].to_numpy(dtype=float)))
                    return f"column '{col}' max abs diff {diff:.3g} > {tol:g}"
            elif not a[col].astype(str).equals(b[col].astype(str)):
                return f"column '{col}' differs"
        return None
    if ref.suffix == ".npz":
        with np.load(ref, allow_pickle=False) as za, np.load(cand, allow_pickle=False) as zb:
            if set(za.files) != set(zb.files):
                return "npz keys differ"
            for k in za.files:
                x, y = za[k], zb[k]
                if x.shape != y.shape:
                    return f"'{k}' shape differs"
                if np.issubdtype(x.dtype, np.number) or x.dtype == bool:
                    if not np.allclose(x.astype(float), y.astype(float), rtol=0, atol=tol, equal_nan=True):
                        return f"'{k}' max abs diff {np.nanmax(np.abs(x.astype(float) - y.astype(float))):.3g} > {tol:g}"
                elif not np.array_equal(x, y):
                    return f"'{k}' differs"
        return None
    if ref.suffix == ".json":
        a = _strip(json.loads(ref.read_text(encoding="utf-8")))
        b = _strip(json.loads(cand.read_text(encoding="utf-8")))
        return None if _json_equal(a, b, tol) else "json content differs (volatile keys ignored)"
    return None  # other files (figures, logs, models) are not compared


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Compare two results directories within tolerances.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--tol-shap", type=float, default=1e-3)
    args = parser.parse_args(argv)
    ref_root, cand_root = resolve(args.reference), resolve(args.candidate)
    if not ref_root.is_dir() or not cand_root.is_dir():
        print("reference and candidate must be directories", file=sys.stderr)
        return 2
    rows = []
    for ref in sorted(p for p in ref_root.rglob("*") if p.is_file() and p.suffix in (".csv", ".npz", ".json")):
        rel = ref.relative_to(ref_root)
        tol = _tol_for(rel, args.tol, args.tol_shap)
        reason = compare_file(ref, cand_root / rel, tol)
        rows.append({"file": rel.as_posix(), "tolerance": tol, "status": "ok" if reason is None else "MISMATCH", "reason": reason or ""})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    n_bad = int((table["status"] == "MISMATCH").sum()) if len(table) else 0
    print(f"\n{len(table)} files compared, {n_bad} mismatches")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
