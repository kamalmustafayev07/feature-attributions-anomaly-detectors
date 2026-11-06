"""DEV baseline: attribution_j = |x_j| in standardized space (distance from the nominal mean in nominal SDs)."""

from __future__ import annotations

import numpy as np

from opssat_xai.contracts import check_attribution, check_matrix


class DevExplainer:
    """Model-free deviation ranking."""

    name = "dev"
    method = "dev"

    def attribute(self, X: np.ndarray, row_ids: np.ndarray | None = None) -> np.ndarray:
        """``|X|`` per feature."""
        X = check_matrix(X)
        return check_attribution(np.abs(X), X)
