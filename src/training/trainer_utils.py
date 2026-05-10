"""Helpers for the Double DQN trainer.

Pulled out so ``trainer.py`` stays under the 150-line orchestrator budget.
Pure functions and small utility classes live here; nothing here owns state
across an entire training run.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn, optim


def is_oom(exc: BaseException) -> bool:
    """Detect CUDA / MPS out-of-memory across torch versions and backends."""
    if torch.cuda.is_available() and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    if "out of memory" in msg:
        return True
    # MPS allocations surface various wordings; cover the common ones.
    if "mpsndarray" in msg:
        return True
    if "alloc" in msg and "fail" in msg:
        return True
    return False


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """ε-greedy schedule: linear from ``start`` to ``end`` over ``decay_steps``."""
    if decay_steps <= 0 or step >= decay_steps:
        return end
    frac = step / decay_steps
    return start + (end - start) * frac


def snapshot(
    online: nn.Module, target: nn.Module, optimizer: optim.Optimizer,
) -> dict[str, Any]:
    """Deep copy of online/target/optimizer state — last-known-good snapshot."""
    return {
        "online": copy.deepcopy(online.state_dict()),
        "target": copy.deepcopy(target.state_dict()),
        "opt": copy.deepcopy(optimizer.state_dict()),
    }


def restore(
    snap: dict[str, Any],
    online: nn.Module, target: nn.Module, optimizer: optim.Optimizer,
) -> None:
    online.load_state_dict(snap["online"])
    target.load_state_dict(snap["target"])
    optimizer.load_state_dict(snap["opt"])
