"""Path resolution (one helper, repository-root relative, Windows-safe), JSON/CSV helpers and manifests."""

from __future__ import annotations

import datetime as dt
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """Repository root (the directory containing pyproject.toml)."""
    return _REPO_ROOT


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` relative to the repository root (absolute paths are returned unchanged)."""
    p = Path(path)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def relative_to_root(path: str | Path) -> str:
    """Repository-relative POSIX string for ``path`` (falls back to the path itself)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def ensure_dir(path: str | Path) -> Path:
    """Create the directory (and parents) if needed and return it."""
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def prepare_run_dir(path: str | Path, force: bool = False, resume: bool = False) -> Path:
    """Create a run directory; refuse to overwrite an existing non-empty one unless ``force`` or ``resume``."""
    p = resolve(path)
    if p.exists() and any(p.iterdir()) and not (force or resume):
        raise FileExistsError(
            f"Run directory {p} already exists and is not empty; pass --force to overwrite or --resume to continue."
        )
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: str | Path, obj: Any) -> Path:
    """Write ``obj`` as indented JSON (numpy types converted)."""
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)
    return p


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with resolve(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str | Path, df: pd.DataFrame) -> Path:
    """Write a DataFrame as CSV without the index, with full float precision."""
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, float_format="%.17g", lineterminator="\n")
    return p


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV written by :func:`write_csv`."""
    return pd.read_csv(resolve(path))


def git_commit_hash() -> str | None:
    """Current git commit hash, or ``None`` when git or the repository is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def package_versions() -> dict[str, str]:
    """Versions of the packages that determine numerical results."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for pkg in ("numpy", "scipy", "pandas", "scikit-learn", "shap", "torch", "matplotlib", "joblib", "psutil"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def utc_now() -> str:
    """ISO-8601 UTC timestamp."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def start_manifest(config: dict[str, Any], script: str, seeds: dict[str, int] | None = None) -> dict[str, Any]:
    """Create the manifest dictionary written to ``results/<run_name>/<stage>/manifest.json``."""
    return {
        "script": script,
        "run_name": config.get("run_name"),
        "start_time": utc_now(),
        "end_time": None,
        "config": config,
        "seeds": seeds or {"root": config.get("seed")},
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "git_commit": git_commit_hash(),
        "package_versions": package_versions(),
        "files_written": [],
    }


def finish_manifest(
    manifest: dict[str, Any], run_dir: str | Path, files: list[str | Path] | None = None, filename: str = "manifest.json"
) -> Path:
    """Stamp the end time, record written files and write the manifest JSON into ``run_dir``."""
    manifest["end_time"] = utc_now()
    manifest["files_written"] = [relative_to_root(f) for f in (files or [])]
    return write_json(resolve(run_dir) / filename, manifest)
