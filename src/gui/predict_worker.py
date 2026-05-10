"""In-process worker for the Predict Next button.

Loads ckpt + NPZ, runs a single greedy forward through ``DuelingDQN``,
returns ``(asof, action, q_values, features_row)`` to the GUI thread.
Honours the locked decision: predict for the next trading day given
features through the latest available close (NPZ test split tail).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PyQt6.QtCore import QObject, pyqtSignal

from src.models.dueling_dqn import DuelingDQN
from src.training.checkpoint import load_online_only
from src.utils.config import load_config
from src.utils.device import pick_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_latest_ckpt(ticker: str, root: Path | None = None) -> Path | None:
    """Find the most-recent ``*_latest.pt`` for a ticker, or ``None``."""
    base = (root or PROJECT_ROOT) / "output" / "models" / ticker
    primary = base / "seed0" / f"{ticker}_seed0_latest.pt"
    if primary.exists():
        return primary
    if not base.exists():
        return None
    candidates = sorted(base.rglob("*_latest.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def resolve_npz(ticker: str, root: Path | None = None) -> Path | None:
    p = (root or PROJECT_ROOT) / "output" / "processed" / f"{ticker}.npz"
    return p if p.exists() else None


class PredictNextWorker(QObject):
    """Single-shot inference worker. Emits ``finished(dict)`` or ``error(str)``."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        ticker: str,
        ckpt_path: str | Path,
        npz_path: str | Path,
        config_path: str | Path,
    ) -> None:
        super().__init__()
        self.ticker = ticker
        self.ckpt_path = Path(ckpt_path)
        self.npz_path = Path(npz_path)
        self.config_path = Path(config_path)

    def run(self) -> None:
        try:
            payload = self._infer()
        except FileNotFoundError as e:
            self.error.emit(str(e))
            return
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"predict failed: {e}")
            return
        self.finished.emit(payload)

    def _infer(self) -> dict:
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.ckpt_path}")
        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ not found: {self.npz_path}")
        cfg = load_config(self.config_path)
        npz = np.load(self.npz_path, allow_pickle=True)
        test = npz["test"]
        window = cfg.data.window
        if test.shape[0] < window:
            raise ValueError(
                f"test split has {test.shape[0]} rows, need >= window={window}",
            )
        obs = test[-window:].astype(np.float32)
        feature_names = list(np.asarray(npz["feature_names"]).tolist())
        last_row = obs[-1]
        features_row = {str(n): float(v) for n, v in zip(feature_names, last_row)}
        asof = str(npz["test_dates"][-1])

        device = pick_device()
        model = DuelingDQN(
            window=window,
            n_features=cfg.data.n_features,
            n_actions=cfg.env.n_actions,
        )
        load_online_only(self.ckpt_path, model, map_location="cpu")
        model.to(device)
        model.eval()
        x = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            q = model(x)
        q_np = q.detach().cpu().numpy().reshape(-1).astype(np.float64)
        action = int(np.argmax(q_np))
        return {
            "ticker": self.ticker,
            "asof": asof,
            "action": action,
            "q_values": q_np,
            "features_row": features_row,
        }
