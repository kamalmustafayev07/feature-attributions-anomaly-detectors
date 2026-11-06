"""All randomness derives from the single config seed through numpy SeedSequence.

``int_seed(root, *keys)`` is deterministic in ``(root, keys)``: keys are strings or ints and are hashed
into a spawn key, so e.g. ``int_seed(seed, "rnd", row_id, repeat)`` gives a stable per-(row, repeat) seed.
"""

from __future__ import annotations

import zlib

import numpy as np


def _key_to_int(key: int | str) -> int:
    if isinstance(key, (int, np.integer)):
        return int(key) & 0xFFFFFFFF
    return zlib.crc32(str(key).encode("utf-8")) & 0xFFFFFFFF


def seed_sequence(root_seed: int, *keys: int | str) -> np.random.SeedSequence:
    """Return a SeedSequence for ``(root_seed, keys...)``."""
    return np.random.SeedSequence(int(root_seed), spawn_key=tuple(_key_to_int(k) for k in keys))


def rng(root_seed: int, *keys: int | str) -> np.random.Generator:
    """Return a numpy Generator deterministically derived from the root seed and keys."""
    return np.random.default_rng(seed_sequence(root_seed, *keys))


def int_seed(root_seed: int, *keys: int | str) -> int:
    """Return a 32-bit integer seed (for sklearn ``random_state``, ``np.random.seed``, ``torch.manual_seed``)."""
    return int(seed_sequence(root_seed, *keys).generate_state(1, dtype=np.uint32)[0])


def spawn_seeds(root_seed: int, n: int, *keys: int | str) -> list[int]:
    """Return ``n`` child integer seeds, e.g. one per anomaly."""
    children = seed_sequence(root_seed, *keys).spawn(n)
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]
