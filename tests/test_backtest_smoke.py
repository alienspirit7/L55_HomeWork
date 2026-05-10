"""Smoke + arithmetic tests for the greedy backtest and buy-and-hold benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.trading_env import TradingEnv  # noqa: E402
from src.evaluation.backtest import BacktestResult, backtest  # noqa: E402
from src.evaluation.benchmark import buy_and_hold  # noqa: E402
from src.models.dueling_dqn import DuelingDQN  # noqa: E402

WINDOW = 30
N_FEATURES = 10


def _sine_env(n_bars: int = 200, *, init_cash: float = 10_000.0, fee_bps: int = 10) -> TradingEnv:
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((n_bars, N_FEATURES)).astype(np.float32)
    t = np.arange(n_bars)
    prices = 100.0 + 5.0 * np.sin(t / 7.0)
    return TradingEnv(feats, prices, prices, window=WINDOW, fee_bps=fee_bps, init_cash=init_cash)


def _flat_env(n_bars: int = 50, *, init_cash: float = 10_000.0, fee_bps: int = 10) -> TradingEnv:
    feats = np.zeros((n_bars, N_FEATURES), dtype=np.float32)
    prices = np.full(n_bars, 100.0, dtype=np.float64)
    return TradingEnv(feats, prices, prices, window=WINDOW, fee_bps=fee_bps, init_cash=init_cash)


def _build_random_model(seed: int = 0) -> DuelingDQN:
    torch.manual_seed(seed)
    return DuelingDQN(window=WINDOW, n_features=N_FEATURES, n_actions=3)


# ------------------------------------------------------------------ tests

def test_backtest_runs_on_random_model():
    env = _sine_env()
    model = _build_random_model()
    result = backtest(model, env)
    assert isinstance(result, BacktestResult)
    expected_len = 200 - WINDOW
    assert result.equity.shape[0] == expected_len
    assert result.equity[0] == pytest.approx(env.init_cash)
    expected_keys = {
        "total_return", "sharpe", "max_drawdown", "win_rate",
        "sortino", "calmar", "turnover", "avg_holding_period",
    }
    assert expected_keys.issubset(set(result.metrics.keys()))


def test_backtest_deterministic_for_fixed_model():
    model = _build_random_model(seed=1)
    eq1 = backtest(model, _sine_env()).equity
    eq2 = backtest(model, _sine_env()).equity
    np.testing.assert_array_equal(eq1, eq2)


def test_buy_and_hold_executes_one_round_trip():
    env = _flat_env(n_bars=50)
    result = buy_and_hold(env)
    assert result.n_trades == 1
    # Flat prices, fee on entry only (no force-close): final equity = init_cash * (1 - fee)
    fee_rate = 10 / 10_000
    expected = 10_000.0 * (1.0 - fee_rate)
    assert result.equity[-1] == pytest.approx(expected, abs=0.5)


def test_buy_and_hold_known_arithmetic():
    fee_rate = 10 / 10_000
    init_cash = 10_000.0
    env = _flat_env(n_bars=50, init_cash=init_cash, fee_bps=10)
    result = buy_and_hold(env)
    # On entry: shares = cash / (price * (1 + fee)), cash = 0.
    # MTM at last close = shares * price = init_cash / (1 + fee).
    expected = init_cash / (1.0 + fee_rate)
    assert result.equity[-1] == pytest.approx(expected, abs=1e-2)


def test_force_close_or_skip_consistency():
    model = _build_random_model(seed=7)
    r1 = backtest(model, _sine_env())
    r2 = backtest(model, _sine_env())
    assert r1.n_trades == r2.n_trades


def test_cli_smoke(tmp_path, monkeypatch):
    # Build a tiny synthetic NPZ matching the prepare_data schema.
    n_bars = 80
    rng = np.random.default_rng(42)
    feats = rng.standard_normal((n_bars, N_FEATURES)).astype(np.float32)
    prices = (100.0 + np.cumsum(rng.standard_normal(n_bars) * 0.1)).astype(np.float32)
    npz_path = tmp_path / "processed" / "FAKE.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"ticker": "FAKE", "n_features": N_FEATURES}
    np.savez(
        npz_path,
        train=feats, val=feats, test=feats,
        train_open=prices, val_open=prices, test_open=prices,
        train_close=prices, val_close=prices, test_close=prices,
        train_dates=np.array(["2020-01-01"] * n_bars, dtype="<U10"),
        val_dates=np.array(["2020-01-01"] * n_bars, dtype="<U10"),
        test_dates=np.array(["2020-01-01"] * n_bars, dtype="<U10"),
        feature_names=np.array([f"f{i}" for i in range(N_FEATURES)]),
        normalizer_state=np.array({}, dtype=object),
        meta=np.array(meta, dtype=object),
    )

    # Build a tiny ckpt with online_state_dict.
    model = _build_random_model(seed=2)
    ckpt_path = tmp_path / "models" / "FAKE_seed3_latest.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"online_state_dict": model.state_dict(), "seed": 3, "ticker": "FAKE"}, ckpt_path)

    out_dir = tmp_path / "backtests"
    config_path = PROJECT_ROOT / "config" / "default.yaml"

    from scripts import backtest as cli  # noqa: WPS433
    rc = cli.main([
        "--model", str(ckpt_path),
        "--ticker", "FAKE",
        "--split", "test",
        "--config", str(config_path),
        "--out", str(out_dir),
        "--npz-dir", str(npz_path.parent),
    ])
    assert rc == 0
    json_path = out_dir / "FAKE_seed3.json"
    png_path = out_dir / "FAKE_seed3_equity.png"
    assert json_path.exists()
    assert png_path.exists()
    assert png_path.stat().st_size > 1024
    payload = json.loads(json_path.read_text())
    assert payload["ticker"] == "FAKE"
    assert payload["seed"] == 3
    assert "model" in payload and "benchmark" in payload
