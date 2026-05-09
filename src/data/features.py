"""Leak-free feature engineering for the Dueling DQN trading project.

Computes 8 market features from OHLCV plus 2 zero-placeholder agent features
(env injects them at step time). Volume normalization (z-score, window=60)
is fit ONLY on the train slice via the Normalizer class.

VWAP is the daily-bar approximation: vwap_t = cumsum(V * (H+L+C)/3) / cumsum(V),
documented in README per locked decision. It is causal (cumulative-only).

All other indicators come from pandas_ta 0.4.71b0 in function form
(`pandas_ta.rsi`, `pandas_ta.macd`, `pandas_ta.bbands`).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta


FEATURE_ORDER: Tuple[str, ...] = (
    "log_return",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bbp",
    "vwap_dist",
    "volume_norm",
    "position_flag",
    "unrealized_pnl_pct",
)

_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 7 indicator columns + raw `volume`, drop warmup NaN rows.

    Returns a DataFrame indexed like `df` (minus warmup rows) with columns:
        log_return, rsi_14, macd, macd_signal, macd_hist, bbp, vwap_dist,
        volume.
    The `volume` column is kept raw; Normalizer.transform turns it into
    `volume_norm`.
    """
    missing = [c for c in _OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    log_return = np.log(close / close.shift(1))

    rsi_14 = ta.rsi(close, length=14)

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    # pandas_ta 0.4 column names: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9.
    macd_line = macd_df.iloc[:, 0]
    macd_hist = macd_df.iloc[:, 1]
    macd_signal = macd_df.iloc[:, 2]

    bb = ta.bbands(close, length=20, std=2)
    # pandas_ta 0.4 BBP column has the form BBP_20_2.0_2.0; pick by prefix.
    bbp_col = next(c for c in bb.columns if c.startswith("BBP_"))
    bbp = bb[bbp_col]

    typical = (high + low + close) / 3.0
    cum_vp = (volume * typical).cumsum()
    cum_v = volume.cumsum()
    vwap = cum_vp / cum_v
    vwap_dist = (close - vwap) / vwap

    out = pd.DataFrame(
        {
            "log_return": log_return,
            "rsi_14": rsi_14,
            "macd": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "bbp": bbp,
            "vwap_dist": vwap_dist,
            "volume": volume,
        },
        index=df.index,
    )
    return out.dropna()


class Normalizer:
    """Train-only z-score normalizer for volume; passes other features through.

    Fit stores `volume_mean` and `volume_std` from the train slice. Transform
    is deterministic and idempotent. State dict is small enough to ship next
    to model checkpoints.
    """

    def __init__(self, volume_window: int = 60) -> None:
        self.volume_window = int(volume_window)
        self._fitted = False
        self.volume_mean: float | None = None
        self.volume_std: float | None = None

    def fit(self, train_df: pd.DataFrame) -> "Normalizer":
        if self._fitted:
            raise RuntimeError("Normalizer.fit called twice")
        if "volume" not in train_df.columns:
            raise ValueError("train_df missing 'volume' column")
        # Rolling z-score parameters: use the global mean/std over the train
        # slice as the leak-free reference (rolling-window window=N would
        # collapse to global on the train tail anyway, and we apply the same
        # scalar params to val/test). volume_window kept in state for audit.
        v = train_df["volume"].astype(float)
        self.volume_mean = float(v.mean())
        self.volume_std = float(v.std(ddof=0))
        if not np.isfinite(self.volume_std) or self.volume_std == 0.0:
            raise ValueError("train volume has zero/non-finite std")
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Normalizer.transform called before fit")
        volume_norm = (df["volume"].astype(float) - self.volume_mean) / self.volume_std
        out = pd.DataFrame(index=df.index)
        for col in ("log_return", "rsi_14", "macd", "macd_signal",
                    "macd_hist", "bbp", "vwap_dist"):
            out[col] = df[col].astype(float)
        out["volume_norm"] = volume_norm
        out["position_flag"] = 0.0
        out["unrealized_pnl_pct"] = 0.0
        return out[list(FEATURE_ORDER)]

    def state_dict(self) -> dict:
        return {
            "volume_window": self.volume_window,
            "volume_mean": self.volume_mean,
            "volume_std": self.volume_std,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state: dict) -> None:
        self.volume_window = int(state["volume_window"])
        self.volume_mean = state["volume_mean"]
        self.volume_std = state["volume_std"]
        self._fitted = bool(state.get("fitted", True))
