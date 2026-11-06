"""Wall-clock (time.perf_counter) and peak RSS (psutil peak working set on Windows) logging as JSON."""

from __future__ import annotations

import os
import platform
import socket
import time
from pathlib import Path
from typing import Any

import psutil

from opssat_xai.io import package_versions, write_json


def peak_rss_bytes() -> int:
    """Peak resident set size of this process (``peak_wset`` on Windows, current RSS elsewhere)."""
    info = psutil.Process(os.getpid()).memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def current_rss_bytes() -> int:
    """Current resident set size."""
    return int(psutil.Process(os.getpid()).memory_info().rss)


def environment_info() -> dict[str, Any]:
    """Hostname, CPU, core counts, platform, Python and package versions."""
    return {
        "hostname": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": package_versions(),
    }


class Timer:
    """``with Timer() as t: ...; t.seconds`` using ``time.perf_counter``."""

    def __init__(self) -> None:
        self.start = 0.0
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start


def runtime_record(label: str, seconds: float, n_items: int | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One runtime entry (per-item time is derived when ``n_items`` is given)."""
    rec: dict[str, Any] = {"label": label, "seconds": float(seconds), "peak_rss_bytes": peak_rss_bytes()}
    if n_items:
        rec["n_items"] = int(n_items)
        rec["seconds_per_item"] = float(seconds) / int(n_items)
    if extra:
        rec.update(extra)
    return rec


def write_runtime_json(path: str | Path, records: list[dict[str, Any]], context: dict[str, Any] | None = None) -> Path:
    """Write runtime records together with the environment description."""
    return write_json(path, {"environment": environment_info(), "context": context or {}, "records": records})
