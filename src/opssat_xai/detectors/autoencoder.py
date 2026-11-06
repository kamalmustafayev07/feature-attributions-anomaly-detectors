"""PyTorch (CPU-only) autoencoder detector: d -> 8 -> 4 -> 8 -> d, score = per-row reconstruction MSE.

Determinism: ``torch.manual_seed``, ``torch.use_deterministic_algorithms(True)`` and ``torch.set_num_threads(1)``
during ``fit``. Never calls ``.cuda()``; tensors stay on the CPU.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from opssat_xai.contracts import check_matrix
from opssat_xai.detectors.base import fit_threshold, predict_from_scores, require_fitted
from opssat_xai.io import resolve


class AutoencoderNet(nn.Module):
    """Symmetric MLP autoencoder with ReLU hidden layers and a linear output."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (8, 4)) -> None:
        super().__init__()
        widths = [int(input_dim), *[int(h) for h in hidden]]
        enc: list[nn.Module] = []
        for a, b in zip(widths[:-1], widths[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        dec: list[nn.Module] = []
        rev = widths[::-1]
        for i, (a, b) in enumerate(zip(rev[:-1], rev[1:])):
            dec.append(nn.Linear(a, b))
            if i < len(rev) - 2:
                dec.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``x``."""
        return self.decoder(self.encoder(x))


class ScoreModule(nn.Module):
    """Wraps a trained net so that ``forward(x)`` returns the scalar anomaly score (per-row MSE), shape ``(n, 1)``."""

    def __init__(self, net: AutoencoderNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-row reconstruction MSE as an ``(n, 1)`` tensor."""
        return ((self.net(x) - x) ** 2).mean(dim=1, keepdim=True)


class AutoencoderDetector:
    """Autoencoder behind the common detector contract (fit / score / threshold / predict)."""

    name = "autoencoder"

    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...] | list[int] = (8, 4),
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        holdout_fraction: float = 0.1,
        seed: int = 0,
        n_threads: int = 1,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.holdout_fraction = float(holdout_fraction)
        self.seed = int(seed)
        self.n_threads = int(n_threads)
        self.net: AutoencoderNet | None = None
        self.threshold: float | None = None
        self.history: pd.DataFrame | None = None
        self.best_epoch: int | None = None

    def fit(self, X_nominal: np.ndarray) -> "AutoencoderDetector":
        """Train on nominal rows only with early stopping on a 10% hold-out of those rows."""
        X = check_matrix(X_nominal, "X_nominal")
        if X.shape[1] != self.input_dim:
            raise ValueError(f"input_dim={self.input_dim} but X has {X.shape[1]} columns")
        prev_threads = torch.get_num_threads()
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(self.n_threads)
        try:
            n = X.shape[0]
            perm = np.random.default_rng(self.seed).permutation(n)
            n_hold = max(1, int(round(self.holdout_fraction * n))) if n > 1 else 0
            hold_idx, train_idx = perm[:n_hold], perm[n_hold:]
            Xt = torch.tensor(X[train_idx], dtype=torch.float32)
            Xh = torch.tensor(X[hold_idx], dtype=torch.float32) if n_hold else Xt
            net = AutoencoderNet(self.input_dim, self.hidden)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            gen = torch.Generator().manual_seed(self.seed)
            best_loss, best_state, best_epoch, bad = float("inf"), copy.deepcopy(net.state_dict()), 0, 0
            rows: list[dict[str, float]] = []
            n_train = Xt.shape[0]
            for epoch in range(self.max_epochs):
                net.train()
                order = torch.randperm(n_train, generator=gen)
                total = 0.0
                for start in range(0, n_train, self.batch_size):
                    xb = Xt[order[start:start + self.batch_size]]
                    loss = ((net(xb) - xb) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += float(loss.item()) * xb.shape[0]
                net.eval()
                with torch.no_grad():
                    hold_loss = float(((net(Xh) - Xh) ** 2).mean().item())
                rows.append({"epoch": epoch, "train_loss": total / n_train, "holdout_loss": hold_loss})
                if hold_loss < best_loss:
                    best_loss, best_state, best_epoch, bad = hold_loss, copy.deepcopy(net.state_dict()), epoch, 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            net.load_state_dict(best_state)
            net.eval()
            self.net, self.history, self.best_epoch = net, pd.DataFrame(rows), int(best_epoch)
        finally:
            torch.set_num_threads(prev_threads)
        return self

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstructions as float64 ``(n, d)`` (computed in float32 on the CPU)."""
        require_fitted(self.net, self.name)
        with torch.no_grad():
            out = self.net(torch.tensor(check_matrix(X), dtype=torch.float32))
        return out.numpy().astype(np.float64)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-row reconstruction MSE: higher = more anomalous."""
        require_fitted(self.net, self.name)
        with torch.no_grad():
            x = torch.tensor(check_matrix(X), dtype=torch.float32)
            mse = ((self.net(x) - x) ** 2).mean(dim=1)
        return mse.numpy().astype(np.float64)

    def set_threshold_from_validation(self, X_val: np.ndarray, y_val: np.ndarray, rule: str = "max_f1") -> float:
        """Choose and store the threshold on the validation set."""
        return fit_threshold(self, X_val, y_val, rule)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``score(X) >= threshold``."""
        return predict_from_scores(self.score(X), self.threshold, self.name)

    def score_module(self) -> ScoreModule:
        """Torch module returning the scalar score, for shap.GradientExplainer."""
        require_fitted(self.net, self.name)
        return ScoreModule(self.net).eval()

    def save(self, path: str | Path) -> Path:
        """Save weights and hyper-parameters with ``torch.save``."""
        require_fitted(self.net, self.name)
        p = resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "state_dict": self.net.state_dict(),
            "input_dim": self.input_dim, "hidden": list(self.hidden), "lr": self.lr,
            "weight_decay": self.weight_decay, "batch_size": self.batch_size, "max_epochs": self.max_epochs,
            "patience": self.patience, "holdout_fraction": self.holdout_fraction, "seed": self.seed,
            "n_threads": self.n_threads, "threshold": self.threshold, "best_epoch": self.best_epoch,
        }
        torch.save(payload, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "AutoencoderDetector":
        """Load a detector saved by :meth:`save` (CPU)."""
        payload = torch.load(resolve(path), map_location="cpu", weights_only=True)
        det = cls(
            input_dim=payload["input_dim"], hidden=payload["hidden"], lr=payload["lr"],
            weight_decay=payload["weight_decay"], batch_size=payload["batch_size"], max_epochs=payload["max_epochs"],
            patience=payload["patience"], holdout_fraction=payload["holdout_fraction"], seed=payload["seed"],
            n_threads=payload["n_threads"],
        )
        net = AutoencoderNet(det.input_dim, det.hidden)
        net.load_state_dict(payload["state_dict"])
        det.net = net.eval()
        det.threshold = payload["threshold"]
        det.best_epoch = payload["best_epoch"]
        return det
