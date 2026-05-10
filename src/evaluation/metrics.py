"""Pure-numpy performance metrics for trading backtests.

All inputs are coerced to float64 numpy arrays at entry. No mutation of inputs.
Conventions:
- Returns are simple per-period returns (not percent).
- Max drawdown is returned as a non-positive number (e.g. -0.23 == 23% drawdown).
- Honest NaN: degenerate inputs return np.nan, not inf or epsilon-clipped values.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "sortino_ratio",
    "calmar_ratio",
    "turnover",
    "average_holding_period",
    "compute_all",
]


def _as_f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def total_return(equity) -> float:
    eq = _as_f64(equity)
    if eq.size < 2 or eq[0] <= 0.0:
        return float("nan")
    return float(eq[-1] / eq[0] - 1.0)


def sharpe_ratio(
    returns,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    r = _as_f64(returns)
    if r.size < 2:
        return float("nan")
    rf_per = risk_free_rate / periods_per_year
    std = float(np.std(r, ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return float("nan")
    excess = float(np.mean(r)) - rf_per
    return float(excess / std * np.sqrt(periods_per_year))


def max_drawdown(equity) -> float:
    """Largest peak-to-trough decline as a non-positive float."""
    eq = _as_f64(equity)
    if eq.size == 0:
        return float("nan")
    running_peak = np.maximum.accumulate(eq)
    # Guard against zero/negative peaks (shouldn't happen for valid equity)
    if np.any(running_peak <= 0.0):
        return float("nan")
    dd = eq / running_peak - 1.0
    return float(dd.min())


def win_rate(trade_pnls) -> float:
    p = _as_f64(trade_pnls)
    if p.size == 0:
        return float("nan")
    return float(np.sum(p > 0.0) / p.size)


def sortino_ratio(
    returns,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    r = _as_f64(returns)
    if r.size < 2:
        return float("nan")
    rf_per = risk_free_rate / periods_per_year
    downside = r[r < rf_per]
    if downside.size < 2:
        return float("nan")
    dstd = float(np.std(downside, ddof=1))
    if dstd == 0.0 or not np.isfinite(dstd):
        return float("nan")
    excess = float(np.mean(r)) - rf_per
    return float(excess / dstd * np.sqrt(periods_per_year))


def calmar_ratio(equity, *, periods_per_year: int = 252) -> float:
    """CAGR / |MaxDD|. Returns nan if MaxDD == 0 or equity invalid."""
    eq = _as_f64(equity)
    if eq.size < 2 or eq[0] <= 0.0 or eq[-1] <= 0.0:
        return float("nan")
    n_periods = eq.size - 1
    cagr = (eq[-1] / eq[0]) ** (periods_per_year / n_periods) - 1.0
    mdd = max_drawdown(eq)
    if not np.isfinite(mdd) or mdd == 0.0:
        return float("nan")
    return float(cagr / abs(mdd))


def turnover(trade_notionals, mean_equity: float) -> float:
    n = _as_f64(trade_notionals)
    me = float(mean_equity)
    if not np.isfinite(me) or me <= 0.0:
        return float("nan")
    if n.size == 0:
        return 0.0
    return float(np.sum(np.abs(n)) / me)


def average_holding_period(trade_durations) -> float:
    d = _as_f64(trade_durations)
    if d.size == 0:
        return float("nan")
    return float(np.mean(d))


def compute_all(
    equity,
    returns,
    trade_pnls,
    trade_notionals,
    trade_durations,
    *,
    periods_per_year: int = 252,
) -> dict[str, float]:
    eq = _as_f64(equity)
    mean_eq = float(eq.mean()) if eq.size > 0 else float("nan")
    return {
        "total_return": total_return(eq),
        "sharpe": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(eq),
        "win_rate": win_rate(trade_pnls),
        "sortino": sortino_ratio(returns, periods_per_year=periods_per_year),
        "calmar": calmar_ratio(eq, periods_per_year=periods_per_year),
        "turnover": turnover(trade_notionals, mean_eq),
        "avg_holding_period": average_holding_period(trade_durations),
    }
