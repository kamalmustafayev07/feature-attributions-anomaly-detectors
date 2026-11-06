"""YAML config loading with ``include`` merging and ``--set key.path=value`` overrides (argparse only)."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from opssat_xai.io import relative_to_root, resolve

DEFAULT_CONFIG = "configs/base.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file; files listed under ``include`` (relative to it) are loaded first and overridden by it."""
    p = resolve(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    includes = raw.pop("include", []) or []
    merged: dict[str, Any] = {}
    for inc in includes:
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = p.parent / inc_path
        merged = deep_merge(merged, load_config(inc_path))
    merged = deep_merge(merged, raw)
    merged["_config_files"] = list(merged.get("_config_files", [])) + [relative_to_root(p)]
    return merged


def set_by_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``cfg[a][b][c] = value`` for ``dotted == "a.b.c"``, creating intermediate dicts."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise KeyError(f"Cannot descend into non-dict key '{k}' of override '{dotted}'")
    node[keys[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.path=value`` overrides; values are parsed as YAML (so ``[a,b]``, ``0.1``, ``true`` work)."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.path=value")
        key, _, raw = item.partition("=")
        set_by_path(out, key.strip(), yaml.safe_load(raw))
    return out


def build_parser(description: str) -> argparse.ArgumentParser:
    """Common CLI: ``--config`` (repeatable, merged in order), ``--set`` overrides, ``--force``, ``--resume``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", action="append", default=None,
                        help=f"YAML config file(s), merged in order (default: {DEFAULT_CONFIG})")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set stability.n_copies=15")
    parser.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    parser.add_argument("--resume", action="store_true", help="continue a run from its checkpoints")
    return parser


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Load and merge the configs named in ``args`` and apply the overrides."""
    paths = args.config or [DEFAULT_CONFIG]
    cfg: dict[str, Any] = {}
    for p in paths:
        cfg = deep_merge(cfg, load_config(p))
    return apply_overrides(cfg, args.overrides)


def active_feature_names(cfg: dict[str, Any]) -> list[str]:
    """Configured feature names minus ``features.exclude`` (order preserved)."""
    names = list(cfg["features"]["names"])
    exclude = set(cfg["features"].get("exclude") or [])
    unknown = exclude - set(names)
    if unknown:
        raise ValueError(f"features.exclude contains unknown feature names: {sorted(unknown)}")
    return [n for n in names if n not in exclude]


def results_dir(cfg: dict[str, Any], *parts: str) -> Path:
    """``results/<run_name>/<parts...>`` resolved against the repository root."""
    return resolve(Path(cfg["paths"]["results"]) / cfg["run_name"] / Path(*parts))


def figures_dir(cfg: dict[str, Any], *parts: str) -> Path:
    """``figures/<run_name>/<parts...>`` resolved against the repository root."""
    return resolve(Path(cfg["paths"]["figures"]) / cfg["run_name"] / Path(*parts))


def processed_dir(cfg: dict[str, Any]) -> Path:
    """``data/processed/<run_name>`` (scaler and per-run processed artifacts; splits.json lives one level up)."""
    return resolve(Path(cfg["paths"]["data_processed"]) / cfg["run_name"])
