"""Minimal logging: one stream handler on the ``opssat_xai`` logger, optional per-run log file."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "opssat_xai") -> logging.Logger:
    """Return a configured logger (INFO to stderr, no duplicate handlers)."""
    logger = logging.getLogger(name)
    root = logging.getLogger("opssat_xai")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return logger


def add_file_handler(run_dir: str | Path, filename: str = "run.log") -> logging.Handler:
    """Also write the ``opssat_xai`` log to ``run_dir/filename``."""
    path = Path(run_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.getLogger("opssat_xai").addHandler(handler)
    return handler
