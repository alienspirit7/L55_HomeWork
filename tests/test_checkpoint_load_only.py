"""Tests for src.training.checkpoint.load_online_only."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn, optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dueling_dqn import DuelingDQN  # noqa: E402
from src.training.checkpoint import (  # noqa: E402
    load_online_only,
    save_ckpt,
)


def _net() -> DuelingDQN:
    return DuelingDQN(window=4, n_features=2, n_actions=3, hidden=8, head_hidden=4)


def test_load_online_only_roundtrip(tmp_path):
    online, target = _net(), _net()
    opt = optim.Adam(online.parameters(), lr=1e-3)
    ckpt = tmp_path / "ckpt.pt"
    save_ckpt(
        ckpt, online, target, opt, step=1234,
        seed=7, ticker="TEST", cfg_dump={"foo": "bar"},
    )

    fresh = _net()
    step, seed = load_online_only(ckpt, fresh)
    assert step == 1234
    assert seed == 7
    # Weights must match.
    for k, v in online.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], v), f"mismatch {k}"


def test_load_online_only_missing_file_raises(tmp_path):
    fresh = _net()
    with pytest.raises(FileNotFoundError):
        load_online_only(tmp_path / "does_not_exist.pt", fresh)


def test_load_online_only_returns_step_and_seed(tmp_path):
    online, target = _net(), _net()
    opt = optim.Adam(online.parameters(), lr=1e-3)
    ckpt = tmp_path / "c.pt"
    save_ckpt(
        ckpt, online, target, opt, step=42,
        seed=0, ticker="T", cfg_dump={},
    )
    step, seed = load_online_only(ckpt, _net())
    assert isinstance(step, int) and step == 42
    assert isinstance(seed, int) and seed == 0
