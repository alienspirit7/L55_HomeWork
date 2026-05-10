"""Golden tests for src/evaluation/metrics.py — TDD-first."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    average_holding_period,
    calmar_ratio,
    compute_all,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    turnover,
    win_rate,
)


def test_total_return_basic():
    eq = np.array([100.0, 110.0, 121.0])
    assert total_return(eq) == pytest.approx(0.21, rel=1e-9)


def test_total_return_loss():
    eq = np.array([100.0, 50.0])
    assert total_return(eq) == pytest.approx(-0.5, rel=1e-9)


def test_sharpe_known_series():
    r = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    mean = r.mean()
    std = r.std(ddof=1)
    expected = mean / std * math.sqrt(252)
    assert sharpe_ratio(r, periods_per_year=252, risk_free_rate=0.0) == pytest.approx(
        expected, rel=1e-9
    )


def test_sharpe_zero_std_returns_nan():
    r = np.array([0.01, 0.01, 0.01])
    assert math.isnan(sharpe_ratio(r))


def test_sharpe_short_series_returns_nan():
    r = np.array([0.01])
    assert math.isnan(sharpe_ratio(r))


def test_max_drawdown_simple():
    eq = np.array([100.0, 120.0, 80.0, 130.0, 60.0])
    expected = (60.0 - 130.0) / 130.0
    result = max_drawdown(eq)
    assert result < 0.0
    assert result == pytest.approx(expected, rel=1e-9)


def test_max_drawdown_monotonic_no_dd():
    eq = np.array([100.0, 110.0, 120.0])
    assert max_drawdown(eq) == pytest.approx(0.0, abs=1e-12)


def test_win_rate_basic():
    pnls = np.array([10.0, -5.0, 20.0, 0.0, -3.0])
    assert win_rate(pnls) == pytest.approx(0.4, rel=1e-9)


def test_win_rate_empty_returns_nan():
    assert math.isnan(win_rate(np.array([])))


def test_sortino_only_negative_returns():
    r = np.array([-0.01, -0.02, -0.01])
    mean = r.mean()
    downside = r[r < 0.0]
    dstd = downside.std(ddof=1)
    expected = mean / dstd * math.sqrt(252)
    result = sortino_ratio(r, periods_per_year=252, risk_free_rate=0.0)
    assert math.isfinite(result)
    assert result == pytest.approx(expected, rel=1e-9)


def test_sortino_no_downside_returns_nan():
    r = np.array([0.01, 0.02])
    assert math.isnan(sortino_ratio(r))


def test_calmar_basic():
    # Use periods_per_year=4, n_periods=2 to keep CAGR sane.
    eq = np.array([100.0, 80.0, 110.0])
    n_periods = len(eq) - 1
    cagr = (110.0 / 100.0) ** (4 / n_periods) - 1
    mdd = (80.0 - 100.0) / 100.0  # -0.2
    expected = cagr / abs(mdd)
    assert calmar_ratio(eq, periods_per_year=4) == pytest.approx(expected, rel=1e-6)


def test_calmar_no_drawdown_returns_nan():
    eq = np.array([100.0, 110.0, 120.0])
    assert math.isnan(calmar_ratio(eq))


def test_turnover_basic():
    notionals = np.array([1000.0, 500.0, 1500.0])
    assert turnover(notionals, mean_equity=10000.0) == pytest.approx(0.3, rel=1e-9)


def test_turnover_zero_equity_returns_nan():
    assert math.isnan(turnover(np.array([1.0]), mean_equity=0.0))


def test_average_holding_period_basic():
    durations = np.array([3.0, 5.0, 4.0])
    assert average_holding_period(durations) == pytest.approx(4.0, rel=1e-9)


def test_average_holding_period_empty_returns_nan():
    assert math.isnan(average_holding_period(np.array([])))


def test_compute_all_dict_shape():
    # Need >=2 downside returns to keep sortino finite.
    eq = np.array([100.0, 105.0, 102.0, 108.0, 104.0, 110.0])
    rets = np.diff(eq) / eq[:-1]
    pnls = np.array([5.0, -3.0, 6.0])
    notionals = np.array([100.0, 105.0, 102.0])
    durations = np.array([2.0, 1.0, 3.0])
    out = compute_all(eq, rets, pnls, notionals, durations, periods_per_year=252)
    keys = {
        "total_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "sortino",
        "calmar",
        "turnover",
        "avg_holding_period",
    }
    assert set(out.keys()) == keys
    for k, v in out.items():
        assert math.isfinite(v), f"{k} is not finite: {v}"


def test_metric_inputs_not_mutated():
    eq = np.array([100.0, 105.0, 102.0, 108.0, 104.0, 110.0])
    rets = np.diff(eq) / eq[:-1]
    pnls = np.array([5.0, -3.0, 6.0])
    notionals = np.array([100.0, 105.0, 102.0])
    durations = np.array([2.0, 1.0, 3.0])
    snaps = [arr.copy() for arr in (eq, rets, pnls, notionals, durations)]
    compute_all(eq, rets, pnls, notionals, durations)
    for before, after in zip(snaps, (eq, rets, pnls, notionals, durations)):
        np.testing.assert_array_equal(before, after)
