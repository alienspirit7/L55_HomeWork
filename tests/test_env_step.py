"""Tests for the all-in/all-out trading environment.

Synthetic series only — no NPZ load. Each test verifies one mechanic:
shape/dtype, action/observation spaces, reset state, deterministic
buy-hold-sell PnL, no-pyramiding/no-sell-while-flat, terminal at end,
no look-ahead (fill = next-bar open), fee accounting, agent-feature
injection into the obs, and immutability of the input features array.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym  # noqa: E402

from src.env.trading_env import TradingEnv  # noqa: E402

WINDOW = 5
N_FEAT = 10
HOLD, BUY, SELL = 0, 1, 2
FEE_BPS = 10
FEE_RATE = FEE_BPS / 10_000.0  # 0.001
INIT_CASH = 10_000.0


def _features(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n, N_FEAT)).astype(np.float32)
    # Match what Normalizer.transform writes: agent feature columns zeroed.
    arr[:, -2:] = 0.0
    return arr


def _flat_prices(n: int, opens: float = 100.0, closes: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    return np.full(n, opens, dtype=np.float32), np.full(n, closes, dtype=np.float32)


def _make_env(n: int = 30, opens=None, closes=None, window: int = WINDOW) -> TradingEnv:
    feats = _features(n)
    if opens is None or closes is None:
        opens, closes = _flat_prices(n)
    return TradingEnv(
        feats, opens, closes, window=window, fee_bps=FEE_BPS, init_cash=INIT_CASH,
    )


# 1
def test_obs_shape_and_dtype():
    env = _make_env()
    obs, info = env.reset()
    assert obs.shape == (WINDOW, N_FEAT)
    assert obs.dtype == np.float32
    assert "cash" in info and info["cash"] == pytest.approx(INIT_CASH)


# 2
def test_action_space_and_observation_space():
    env = _make_env()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 3
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (WINDOW, N_FEAT)
    assert env.observation_space.dtype == np.float32


# 3
def test_reset_initial_state():
    env = _make_env()
    obs, info = env.reset()
    assert info["shares"] == 0.0
    assert info["cash"] == pytest.approx(INIT_CASH)
    assert info["position"] == 0
    # last column = unrealized_pnl_pct, second-to-last = position_flag
    assert obs[-1, -2] == pytest.approx(0.0)
    assert obs[-1, -1] == pytest.approx(0.0)


# 4
def test_buy_then_hold_then_sell():
    # 5-bar synthetic: window=2 so we can act early.
    feats = _features(8)
    opens = np.array([10, 20, 25, 30, 40, 50, 60, 70], dtype=np.float32)
    closes = np.array([11, 22, 27, 33, 44, 55, 66, 77], dtype=np.float32)
    env = TradingEnv(feats, opens, closes, window=2, fee_bps=FEE_BPS,
                     init_cash=INIT_CASH)
    env.reset()  # cursor t=2; next bar open is opens[3]=30
    _, _, _, _, info_b = env.step(BUY)  # fills at opens[3]=30
    buy_price = 30.0
    expected_shares = INIT_CASH / (buy_price * (1 + FEE_RATE))
    expected_buy_fee = expected_shares * buy_price * FEE_RATE
    assert info_b["shares"] == pytest.approx(expected_shares, rel=1e-6)
    assert info_b["cash"] == pytest.approx(0.0, abs=1e-6)
    assert info_b["fee_paid_this_step"] == pytest.approx(expected_buy_fee, rel=1e-6)
    # Hold at t=3, then sell at next bar open (opens[5]=50).
    _, _, _, _, _ = env.step(HOLD)
    _, _, _, _, info_s = env.step(SELL)
    sell_price = 50.0
    expected_sell_fee = expected_shares * sell_price * FEE_RATE
    expected_cash = expected_shares * sell_price - expected_sell_fee
    assert info_s["shares"] == pytest.approx(0.0, abs=1e-6)
    assert info_s["cash"] == pytest.approx(expected_cash, rel=1e-6)
    assert info_s["fee_paid_this_step"] == pytest.approx(expected_sell_fee, rel=1e-6)


# 5
def test_buy_while_long_is_noop():
    env = _make_env()
    env.reset()
    _, _, _, _, info1 = env.step(BUY)
    cash_after_buy = info1["cash"]
    shares_after_buy = info1["shares"]
    _, _, _, _, info2 = env.step(BUY)
    assert info2["cash"] == pytest.approx(cash_after_buy)
    assert info2["shares"] == pytest.approx(shares_after_buy)
    assert info2["fee_paid_this_step"] == pytest.approx(0.0)


# 6
def test_sell_while_flat_is_noop():
    env = _make_env()
    env.reset()
    _, reward, _, _, info = env.step(SELL)
    assert info["cash"] == pytest.approx(INIT_CASH)
    assert info["shares"] == pytest.approx(0.0)
    assert info["fee_paid_this_step"] == pytest.approx(0.0)
    # Flat with flat prices ⇒ equity unchanged ⇒ reward 0.
    assert reward == pytest.approx(0.0)


# 7
def test_terminal_at_end():
    # Cursor starts at WINDOW; usable next-bar opens are indices WINDOW+1..n-1.
    # n = WINDOW + 6 ⇒ 5 steps, last one terminal.
    n = WINDOW + 6
    env = _make_env(n=n)
    env.reset()
    terms = []
    for _ in range(5):
        _, _, term, _, _ = env.step(HOLD)
        terms.append(term)
    assert terms[:-1] == [False] * 4
    assert terms[-1] is True


# 8
def test_no_lookahead_buy_executes_at_next_open():
    feats = _features(10)
    # Sharp jump between close[t] and open[t+1] ⇒ different fill price.
    opens = np.array([100, 100, 100, 100, 100, 200, 200, 200, 200, 200], dtype=np.float32)
    closes = np.array([100, 100, 100, 100, 100, 100, 200, 200, 200, 200], dtype=np.float32)
    env = TradingEnv(feats, opens, closes, window=4, fee_bps=FEE_BPS,
                     init_cash=INIT_CASH)
    env.reset()  # t=4; next-bar open is opens[5]=200
    _, _, _, _, info = env.step(BUY)
    # Fill must use opens[5]=200, not closes[4]=100.
    expected_shares = INIT_CASH / (200.0 * (1 + FEE_RATE))
    assert info["shares"] == pytest.approx(expected_shares, rel=1e-6)


# 9
def test_fee_applied_on_entry_and_exit():
    feats = _features(8)
    opens = np.full(8, 100.0, dtype=np.float32)
    closes = np.full(8, 100.0, dtype=np.float32)
    env = TradingEnv(feats, opens, closes, window=2, fee_bps=FEE_BPS,
                     init_cash=INIT_CASH)
    env.reset()
    _, _, _, _, ib = env.step(BUY)
    _, _, _, _, ih = env.step(HOLD)
    _, _, _, _, is_ = env.step(SELL)
    assert ib["fee_paid_this_step"] > 0
    assert ih["fee_paid_this_step"] == pytest.approx(0.0)
    assert is_["fee_paid_this_step"] > 0
    # Round-trip at the same price: final equity = init - 2 fees.
    shares = INIT_CASH / (100.0 * (1 + FEE_RATE))
    expected_total_fees = 2 * shares * 100.0 * FEE_RATE
    assert is_["equity"] == pytest.approx(INIT_CASH - expected_total_fees, rel=1e-6)


# 10
def test_observation_agent_features_filled():
    feats = _features(10)
    opens = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100], dtype=np.float32)
    closes = np.array([100, 100, 100, 100, 100, 100, 110, 121, 121, 121], dtype=np.float32)
    env = TradingEnv(feats, opens, closes, window=4, fee_bps=FEE_BPS,
                     init_cash=INIT_CASH)
    env.reset()  # t=4
    obs, _, _, _, info = env.step(BUY)  # fill at opens[5]=100; cursor advances to 5
    # Last row of obs is row index t=5; closes[5] = 100. avg_entry ≈ 100*(1+fee).
    assert obs[-1, -2] == pytest.approx(1.0)  # position_flag = long
    avg_entry = info["avg_entry"]
    expected_pnl_pct = (closes[5] / avg_entry) - 1.0
    assert obs[-1, -1] == pytest.approx(expected_pnl_pct, rel=1e-5, abs=1e-6)
    # Step a HOLD; closes[6]=110 ⇒ unrealized_pnl_pct should now be larger.
    obs2, _, _, _, _ = env.step(HOLD)
    pnl2 = obs2[-1, -1]
    assert pnl2 > obs[-1, -1]


# 11
def test_input_features_not_mutated():
    feats = _features(20)
    opens = np.full(20, 100.0, dtype=np.float32)
    closes = np.full(20, 100.0, dtype=np.float32)
    snapshot = feats.copy()
    env = TradingEnv(feats, opens, closes, window=4, fee_bps=FEE_BPS,
                     init_cash=INIT_CASH)
    env.reset()
    env.step(BUY)
    env.step(HOLD)
    env.step(SELL)
    np.testing.assert_array_equal(feats, snapshot)
