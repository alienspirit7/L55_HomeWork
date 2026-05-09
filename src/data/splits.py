"""Chronological 70/15/15 temporal split for time-series data.

No shuffling; preserves DataFrame index. Splits computed by floor(N*ratio).
"""
from __future__ import annotations

import math

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) chronological slices of `df`.

    Splits by row count: n_train = floor(N*train), n_val = floor(N*val),
    n_test = N - n_train - n_val. Index order is preserved (no shuffle).
    Raises ValueError if ratios do not sum to 1.0 (within 1e-6).
    """
    if not math.isclose(train + val + test, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"split ratios must sum to 1.0, got {train + val + test:.6f}"
        )
    n = len(df)
    n_train = int(n * train)
    n_val = int(n * val)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]
    return train_df, val_df, test_df
