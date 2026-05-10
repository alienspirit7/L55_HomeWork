"""Greedy rollout backtest for a trained Dueling DQN.

Convention (consistent with src/evaluation/benchmark.py):
- Open positions at termination are NOT force-closed; final equity is mark-to-
  market at the last close. This avoids inventing an extra fee that the env
  would not have charged. The benchmark follows the same convention so
  comparisons are fair.

Trade-boundary detection uses the env's existing info dict (``position`` flag
+ ``equity`` snapshot). When ``position`` transitions 0 -> 1 we record the
entry equity and the time index; when it transitions 1 -> 0 we record the
round-trip's PnL (equity_now - entry_equity), notional (entry_equity), and
duration in bars. This matches the env's all-in/all-out semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import compute_all


@dataclass(frozen=True)
class BacktestResult:
    equity: np.ndarray
    returns: np.ndarray
    trade_pnls: np.ndarray
    trade_notionals: np.ndarray
    trade_durations: np.ndarray
    metrics: dict[str, float]
    n_trades: int
    extras: dict[str, Any] = field(default_factory=dict)


def _step_env_with_action(env, action: int):
    return env.step(action)


def backtest(
    model: nn.Module,
    env,
    *,
    device: torch.device | None = None,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run a greedy (argmax-Q) rollout to termination, returning a BacktestResult."""
    model.eval()
    if device is not None:
        model.to(device)
    target_device = device if device is not None else next(model.parameters()).device

    obs, info = env.reset()
    equity_series: list[float] = [float(info["equity"])]
    pnls: list[float] = []
    notionals: list[float] = []
    durations: list[int] = []

    open_entry_equity: float | None = None
    open_entry_step: int | None = None
    prev_position = int(info["position"])

    done = False
    while not done:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=target_device).unsqueeze(0)
            q = model(obs_t)
            action = int(q.argmax(dim=-1).item())
        obs, _reward, terminated, truncated, info = _step_env_with_action(env, action)
        cur_equity = float(info["equity"])
        equity_series.append(cur_equity)
        cur_position = int(info["position"])
        cur_step = int(info["t"])

        if prev_position == 0 and cur_position == 1:
            # Entry just filled: record starting equity (pre-MTM-drift baseline)
            # and entry step. Notional = capital deployed = entry_equity.
            open_entry_equity = cur_equity
            open_entry_step = cur_step
        elif prev_position == 1 and cur_position == 0:
            # Exit just filled: round-trip closed.
            if open_entry_equity is not None and open_entry_step is not None:
                pnls.append(cur_equity - open_entry_equity)
                notionals.append(open_entry_equity)
                durations.append(cur_step - open_entry_step)
            open_entry_equity = None
            open_entry_step = None
        prev_position = cur_position
        done = bool(terminated or truncated)

    equity = np.asarray(equity_series, dtype=np.float64)
    returns = np.diff(equity) / equity[:-1]
    trade_pnls = np.asarray(pnls, dtype=np.float64)
    trade_notionals = np.asarray(notionals, dtype=np.float64)
    trade_durations = np.asarray(durations, dtype=np.float64)

    metrics = compute_all(
        equity, returns, trade_pnls, trade_notionals, trade_durations,
        periods_per_year=periods_per_year,
    )
    return BacktestResult(
        equity=equity,
        returns=returns,
        trade_pnls=trade_pnls,
        trade_notionals=trade_notionals,
        trade_durations=trade_durations,
        metrics=metrics,
        n_trades=len(pnls),
        extras={"position_open_at_end": prev_position == 1},
    )
