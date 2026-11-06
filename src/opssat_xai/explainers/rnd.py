"""RND floor: a uniformly random permutation of the features per row, seeded by (row_id, repeat)."""

from __future__ import annotations

import numpy as np

from opssat_xai.contracts import check_attribution, check_matrix
from opssat_xai.seeds import rng


class RndExplainer:
    """Random ranking; ``attribute`` returns scores whose descending order is a random permutation."""

    name = "rnd"
    method = "rnd"

    def __init__(self, seed: int, repeat: int = 0) -> None:
        self.seed = int(seed)
        self.repeat = int(repeat)

    def attribute(self, X: np.ndarray, row_ids: np.ndarray | None = None) -> np.ndarray:
        """Row ``i`` gets ``rng(seed, 'rnd', row_id, repeat).permutation(d)`` as float scores."""
        X = check_matrix(X)
        n, d = X.shape
        ids = np.arange(n) if row_ids is None else np.asarray(row_ids)
        out = np.empty((n, d), dtype=np.float64)
        for i in range(n):
            out[i] = rng(self.seed, "rnd", int(ids[i]), self.repeat).permutation(d).astype(np.float64)
        return check_attribution(out, X)
