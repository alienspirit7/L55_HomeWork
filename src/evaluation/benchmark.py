"""Buy-and-hold benchmark — same env semantics as the model backtest.

Strategy: at the very first allowable step, send a Buy. From then on, Hold.
We do NOT force-close at termination — final equity is mark-to-market at the
last close. This mirrors src/evaluation/backtest.py, so the comparison is
apples-to-apples (both pay one entry fee; neither pays a forced exit fee).
With a flat-price 50-bar env this means final equity = init_cash / (1 + fee_rate).
"""
from __future__ import annotations

import numpy as np

from src.env.trading_env import BUY, HOLD
from src.evaluation.backtest import BacktestResult
from src.evaluation.metrics import compute_all


def buy_and_hold(env, *, periods_per_year: int = 252) -> BacktestResult:
    obs, info = env.reset()
    equity_series: list[float] = [float(info["equity"])]
    pnls: list[float] = []
    notionals: list[float] = []
    durations: list[int] = []

    entry_equity: float | None = None
    entry_step: int | None = None
    prev_position = int(info["position"])
    first_step = True

    done = False
    while not done:
        action = BUY if first_step else HOLD
        first_step = False
        obs, _reward, terminated, truncated, info = env.step(action)
        cur_equity = float(info["equity"])
        equity_series.append(cur_equity)
        cur_position = int(info["position"])
        cur_step = int(info["t"])

        if prev_position == 0 and cur_position == 1:
            entry_equity = cur_equity
            entry_step = cur_step
        elif prev_position == 1 and cur_position == 0:
            if entry_equity is not None and entry_step is not None:
                pnls.append(cur_equity - entry_equity)
                notionals.append(entry_equity)
                durations.append(cur_step - entry_step)
            entry_equity = None
            entry_step = None
        prev_position = cur_position
        done = bool(terminated or truncated)

    # Buy-and-hold by construction completes one round-trip iff position
    # somehow closes (it shouldn't — we only ever send Buy then Hold). We
    # still report n_trades=1 because an entry happened, even if the position
    # is still open at the last bar (consistent with how a real buy-and-hold
    # investor is "in one trade"). The trade arrays remain empty in that case
    # — they only carry CLOSED round-trips.
    n_trades = 1 if entry_equity is not None or len(pnls) > 0 else 0

    equity = np.asarray(equity_series, dtype=np.float64)
    returns = np.diff(equity) / equity[:-1] if equity.size > 1 else np.array([], dtype=np.float64)
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
        n_trades=n_trades,
        extras={"position_open_at_end": prev_position == 1},
    )
