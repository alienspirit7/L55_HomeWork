"""Checkpoint save/load utilities for the Double DQN trainer.

Save format (Phase 4 will load this):
    online_state_dict, target_state_dict, optimizer_state_dict,
    step, seed, ticker,
    numpy_rng_state, torch_rng_state,
    cfg_dump (config dataclass dumped to dict for traceability),
    normalizer_state (optional; reserved for full reproducibility from raw OHLCV).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim


def save_ckpt(
    path: str | Path,
    online: nn.Module,
    target: nn.Module,
    optimizer: optim.Optimizer,
    step: int,
    *,
    seed: int,
    ticker: str,
    cfg_dump: dict[str, Any],
    normalizer_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "seed": int(seed),
        "ticker": str(ticker),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cfg_dump": cfg_dump,
        "normalizer_state": normalizer_state,
    }
    torch.save(state, path)


def load_ckpt(
    path: str | Path,
    online: nn.Module,
    target: nn.Module,
    optimizer: optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    state = torch.load(str(path), map_location=map_location, weights_only=False)
    online.load_state_dict(state["online_state_dict"])
    target.load_state_dict(state["target_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state
