"""Tests for cross-seed aggregator (mean ± std reporting)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation.aggregate import (
    aggregate_seed_results,
    render_markdown_summary,
)


METRIC_KEYS = (
    "total_return", "sharpe", "sortino", "calmar",
    "max_drawdown", "win_rate", "turnover", "avg_holding_period",
)


def _seed_entry(seed: int, **overrides):
    base = {
        "seed": seed,
        "metrics": {
            "total_return": 0.10,
            "sharpe": 1.0,
            "sortino": 1.5,
            "calmar": 0.8,
            "max_drawdown": -0.20,
            "win_rate": 0.55,
            "turnover": 2.0,
            "avg_holding_period": 5.0,
        },
        "n_trades": 4,
        "final_equity": 11000.0,
    }
    base["metrics"].update(overrides.get("metrics", {}))
    base.update({k: v for k, v in overrides.items() if k != "metrics"})
    return base


def _benchmark():
    return {
        "metrics": {
            "total_return": 0.20,
            "sharpe": 1.2,
            "sortino": 1.8,
            "calmar": 1.0,
            "max_drawdown": -0.30,
            "win_rate": float("nan"),
            "turnover": 0.0,
            "avg_holding_period": float("nan"),
        },
        "n_trades": 1,
        "final_equity": 12000.0,
    }


def test_aggregate_metrics_mean_std():
    per_seed = [
        _seed_entry(0, metrics={"total_return": 0.10, "sharpe": 1.0}),
        _seed_entry(1, metrics={"total_return": 0.20, "sharpe": 2.0}),
        _seed_entry(2, metrics={"total_return": 0.30, "sharpe": 3.0}),
    ]
    agg = aggregate_seed_results(per_seed, _benchmark())
    assert agg["seeds"] == [0, 1, 2]

    tr = agg["model"]["total_return"]
    assert tr["values"] == [0.10, 0.20, 0.30]
    assert math.isclose(tr["mean"], 0.20, abs_tol=1e-9)
    assert math.isclose(tr["std"], float(np.std([0.10, 0.20, 0.30], ddof=1)), abs_tol=1e-9)

    sh = agg["model"]["sharpe"]
    assert math.isclose(sh["mean"], 2.0, abs_tol=1e-9)
    assert math.isclose(sh["std"], float(np.std([1.0, 2.0, 3.0], ddof=1)), abs_tol=1e-9)


def test_aggregate_propagates_nan():
    per_seed = [
        _seed_entry(0, metrics={"sharpe": 1.0}),
        _seed_entry(1, metrics={"sharpe": float("nan")}),
        _seed_entry(2, metrics={"sharpe": 3.0}),
    ]
    agg = aggregate_seed_results(per_seed, _benchmark())
    sh = agg["model"]["sharpe"]
    assert math.isnan(sh["mean"])
    assert math.isnan(sh["std"])
    # Other metrics still aggregated normally.
    assert math.isfinite(agg["model"]["total_return"]["mean"])


def test_render_markdown_includes_all_metrics():
    per_seed = [_seed_entry(s) for s in (0, 1, 2)]
    agg = aggregate_seed_results(per_seed, _benchmark())
    md = render_markdown_summary(agg, ticker="NVDA", seed_count=3)
    for key in METRIC_KEYS:
        assert key in md, f"missing metric {key} in markdown"
    # Benchmark column has at least one numeric value visible.
    assert "+20.00%" in md or "20.00%" in md
    # Header
    assert "# Backtest Summary: NVDA" in md
    # Per-seed table
    assert "Seed" in md
    assert "n_trades" in md


def test_render_markdown_handles_missing_benchmark():
    per_seed = [_seed_entry(s) for s in (0, 1)]
    agg = aggregate_seed_results(per_seed, {})
    md = render_markdown_summary(agg, ticker="FAKE", seed_count=2)
    assert "n/a" in md  # benchmark cells render n/a
    # Doesn't crash with None either.
    agg2 = aggregate_seed_results(per_seed, None)
    md2 = render_markdown_summary(agg2, ticker="FAKE", seed_count=2)
    assert "n/a" in md2


def test_render_markdown_caveat_present():
    per_seed = [_seed_entry(s) for s in (0, 1, 2)]
    agg = aggregate_seed_results(per_seed, _benchmark())
    md = render_markdown_summary(agg, ticker="NVDA", seed_count=3)
    assert "252" in md
    assert "10 bps" in md
    # "annualized" mention
    assert "annual" in md.lower()
