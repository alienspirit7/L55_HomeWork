"""All-in/all-out trading environment (gymnasium API).

Action space (Discrete(3)): 0=Hold, 1=Buy, 2=Sell.
Execution: action at step t fills at opens[t+1] (next-bar Open) — no
intra-bar look-ahead. Equity is marked-to-market at closes[t+1].
Reward = new_equity - last_equity (absolute USD; fees already subtracted
from equity on Buy/Sell ticks via cash). Per locked decisions: 10 bps
fee on notional, init_cash=10000, no pyramiding, no shorting.

Fractional shares are allowed (academic project; documented in PRD).
The input ``features`` array is never mutated — observations are
constructed in a fresh buffer with the agent-feature columns overwritten.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

HOLD, BUY, SELL = 0, 1, 2


class TradingEnv(gym.Env):
    """Single-asset all-in/all-out RL environment with next-bar-open exec."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        *,
        window: int = 30,
        fee_bps: int = 10,
        init_cash: float = 10000.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        feats = np.ascontiguousarray(features, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[1] < 2:
            raise ValueError("features must be 2-D with at least 2 columns")
        n = feats.shape[0]
        if not (len(opens) == n == len(closes)):
            raise ValueError("features, opens, closes must align in length")
        if n <= window + 1:
            raise ValueError("need N > window + 1 rows")
        self._features = feats  # never written to
        self._opens = np.asarray(opens, dtype=np.float64)
        self._closes = np.asarray(closes, dtype=np.float64)
        self._n = n
        self._n_feat = feats.shape[1]
        self.window = int(window)
        self.fee_rate = float(fee_bps) / 10_000.0
        self.init_cash = float(init_cash)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window, self._n_feat), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)

        self._t = 0
        self._cash = 0.0
        self._shares = 0.0
        self._avg_entry = 0.0
        self._last_equity = 0.0

    # ---- gymnasium API ----------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = self.window  # first obs is rows [0, window)
        self._cash = self.init_cash
        self._shares = 0.0
        self._avg_entry = 0.0
        self._last_equity = self.init_cash
        return self._obs(), self._info(fee_paid=0.0)

    def step(self, action: int):
        if self._t + 1 >= self._n:
            # Defensive: caller should have stopped after terminated=True.
            raise RuntimeError("step called past terminal state")
        fill_price = float(self._opens[self._t + 1])
        fee_paid = 0.0
        if action == BUY and self._shares == 0.0 and self._cash > 0.0:
            shares = self._cash / (fill_price * (1 + self.fee_rate))
            fee_paid = shares * fill_price * self.fee_rate
            self._cash = 0.0
            self._shares = shares
            # avg_entry includes the fee impact (effective cost basis).
            self._avg_entry = fill_price * (1 + self.fee_rate)
        elif action == SELL and self._shares > 0.0:
            gross = self._shares * fill_price
            fee_paid = gross * self.fee_rate
            self._cash = gross - fee_paid
            self._shares = 0.0
            self._avg_entry = 0.0
        # Hold, buy-while-long, sell-while-flat: no-op.

        self._t += 1
        mtm_close = float(self._closes[self._t])
        new_equity = self._cash + self._shares * mtm_close
        reward = new_equity - self._last_equity
        self._last_equity = new_equity
        terminated = self._t + 1 >= self._n
        return self._obs(), float(reward), terminated, False, self._info(fee_paid)

    # ---- helpers ----------------------------------------------------------
    def _obs(self) -> np.ndarray:
        # Last `window` bars including the current cursor row t inclusive.
        # After reset (t=window) this is rows [1, window+1); after step it
        # slides forward by one. agent features overwrite the last row.
        start = self._t - self.window + 1
        end = self._t + 1
        out = self._features[start:end].copy()  # never mutate the input array
        pos_flag = 1.0 if self._shares > 0.0 else 0.0
        if self._shares > 0.0 and self._avg_entry > 0.0:
            close_now = float(self._closes[self._t])
            unrealized = (close_now / self._avg_entry) - 1.0
        else:
            unrealized = 0.0
        out[-1, -2] = pos_flag
        out[-1, -1] = unrealized
        return out

    def _info(self, fee_paid: float) -> dict[str, Any]:
        equity = self._cash + self._shares * float(self._closes[min(self._t, self._n - 1)])
        return {
            "cash": self._cash,
            "shares": self._shares,
            "equity": equity,
            "position": 1 if self._shares > 0.0 else 0,
            "fee_paid_this_step": fee_paid,
            "t": self._t,
            "avg_entry": self._avg_entry,
        }
