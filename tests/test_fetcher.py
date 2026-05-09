"""Tests for the 3-tier data fetcher.

Tier 1: parquet cache (ticker-only key, TTL 24h).
Tier 2: live yfinance behind ApiGatekeeper.
Tier 3: CSV fallback in input/.
"""
from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data import fetcher as fetcher_mod
from src.data.fetcher import DataUnavailable, fetch


OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class FakeGatekeeper:
    """Minimal stand-in: counts acquire() calls."""

    def __init__(self) -> None:
        self.acquire_calls = 0

    @contextmanager
    def acquire(self, timeout=None):
        self.acquire_calls += 1
        yield


def _frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(dates)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Open": rng.uniform(100, 200, n),
            "High": rng.uniform(200, 300, n),
            "Low": rng.uniform(50, 100, n),
            "Close": rng.uniform(100, 200, n),
            "Volume": rng.integers(1_000_000, 10_000_000, n),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def _write_parquet(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _write_csv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="Date")


@pytest.fixture
def dirs(tmp_path):
    cache = tmp_path / "cache"
    fallback = tmp_path / "fallback"
    cache.mkdir()
    fallback.mkdir()
    return cache, fallback


def test_tier1_parquet_hit(dirs):
    cache, fallback = dirs
    full = _frame(pd.date_range("2020-01-01", "2024-12-31", freq="D"))
    _write_parquet(cache / "AAPL.parquet", full)
    gk = FakeGatekeeper()
    out = fetch(
        "AAPL", "2021-01-01", "2021-12-31", gk,
        cache_dir=cache, fallback_dir=fallback, ttl_hours=24,
    )
    assert gk.acquire_calls == 0
    assert out.index.min() >= pd.Timestamp("2021-01-01")
    assert out.index.max() <= pd.Timestamp("2021-12-31")
    assert list(out.columns) == OHLCV


def test_tier1_stale_triggers_refetch(dirs):
    cache, fallback = dirs
    old = _frame(pd.date_range("2020-01-01", "2024-12-31", freq="D"))
    pq = cache / "AAPL.parquet"
    _write_parquet(pq, old)
    # Make mtime older than TTL.
    very_old = pq.stat().st_mtime - 48 * 3600
    os.utime(pq, (very_old, very_old))
    fresh = _frame(pd.date_range("2021-01-01", "2021-12-31", freq="D"))
    gk = FakeGatekeeper()
    with patch.object(fetcher_mod.yf, "download", return_value=fresh.copy()) as mock_dl:
        out = fetch(
            "AAPL", "2021-01-01", "2021-12-31", gk,
            cache_dir=cache, fallback_dir=fallback, ttl_hours=24,
        )
    assert gk.acquire_calls == 1
    assert mock_dl.called
    new_mtime = pq.stat().st_mtime
    assert new_mtime > very_old + 1000
    assert list(out.columns) == OHLCV


def test_tier2_writes_parquet_after_fetch(dirs):
    cache, fallback = dirs
    fresh = _frame(pd.date_range("2020-01-01", "2021-12-31", freq="D"))
    gk = FakeGatekeeper()
    with patch.object(fetcher_mod.yf, "download", return_value=fresh.copy()) as mock_dl:
        out1 = fetch(
            "MSFT", "2020-01-01", "2021-12-31", gk,
            cache_dir=cache, fallback_dir=fallback, ttl_hours=24,
        )
    assert (cache / "MSFT.parquet").exists()
    assert mock_dl.call_count == 1
    # Second call: inner sub-range — should hit Tier 1 only.
    with patch.object(fetcher_mod.yf, "download", side_effect=AssertionError("no fetch")):
        out2 = fetch(
            "MSFT", "2020-06-01", "2020-12-31", gk,
            cache_dir=cache, fallback_dir=fallback, ttl_hours=24,
        )
    assert out2.index.min() >= pd.Timestamp("2020-06-01")
    assert out2.index.max() <= pd.Timestamp("2020-12-31")
    assert list(out1.columns) == OHLCV


def test_tier2_merges_with_existing_parquet(dirs):
    cache, fallback = dirs
    existing = _frame(pd.date_range("2020-01-01", "2020-12-31", freq="D"))
    pq = cache / "GOOG.parquet"
    _write_parquet(pq, existing)
    # Force stale so the fetcher refetches and merges.
    very_old = pq.stat().st_mtime - 48 * 3600
    os.utime(pq, (very_old, very_old))
    new = _frame(pd.date_range("2021-01-01", "2021-12-31", freq="D"))
    gk = FakeGatekeeper()
    with patch.object(fetcher_mod.yf, "download", return_value=new.copy()):
        out = fetch(
            "GOOG", "2020-01-01", "2021-12-31", gk,
            cache_dir=cache, fallback_dir=fallback, ttl_hours=24,
        )
    merged = pd.read_parquet(pq)
    assert merged.index.is_unique
    assert merged.index.min() == pd.Timestamp("2020-01-01")
    assert merged.index.max() == pd.Timestamp("2021-12-31")
    assert out.index.min() == pd.Timestamp("2020-01-01")
    assert out.index.max() == pd.Timestamp("2021-12-31")


def test_tier3_csv_fallback_on_yfinance_failure(dirs):
    cache, fallback = dirs
    csv_df = _frame(pd.date_range("2020-01-01", "2021-12-31", freq="D"))
    _write_csv(fallback / "NVDA.csv", csv_df)
    gk = FakeGatekeeper()
    with patch.object(fetcher_mod.yf, "download", side_effect=RuntimeError("network")):
        out = fetch(
            "NVDA", "2020-06-01", "2021-06-30", gk,
            cache_dir=cache, fallback_dir=fallback, ttl_hours=24,
        )
    assert list(out.columns) == OHLCV
    assert out.index.min() >= pd.Timestamp("2020-06-01")
    assert out.index.max() <= pd.Timestamp("2021-06-30")
    assert isinstance(out.index, pd.DatetimeIndex)


def test_tier3_missing_csv_raises_data_unavailable(dirs):
    cache, fallback = dirs
    gk = FakeGatekeeper()
    with patch.object(fetcher_mod.yf, "download", side_effect=RuntimeError("network")):
        with pytest.raises(DataUnavailable):
            fetch(
                "ZZZZ", "2020-01-01", "2020-12-31", gk,
                cache_dir=cache, fallback_dir=fallback, ttl_hours=24,
            )


def test_partial_yfinance_then_csv_fills_gap(dirs):
    cache, fallback = dirs
    csv_df = _frame(pd.date_range("2021-01-01", "2021-12-31", freq="D"))
    _write_csv(fallback / "META.csv", csv_df)
    partial = _frame(pd.date_range("2021-01-01", "2021-06-30", freq="D"))
    gk = FakeGatekeeper()
    with patch.object(fetcher_mod.yf, "download", return_value=partial.copy()):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = fetch(
                "META", "2021-01-01", "2021-12-31", gk,
                cache_dir=cache, fallback_dir=fallback, ttl_hours=24,
            )
    assert out.index.min() == pd.Timestamp("2021-01-01")
    assert out.index.max() == pd.Timestamp("2021-12-31")
    assert out.index.is_unique
    # Tier 2 (partial) values for overlap dates must be preferred over CSV.
    overlap_date = pd.Timestamp("2021-03-15")
    assert np.isclose(out.loc[overlap_date, "Open"], partial.loc[overlap_date, "Open"])
    # Tier 3 fills the tail.
    tail_date = pd.Timestamp("2021-09-15")
    assert np.isclose(out.loc[tail_date, "Open"], csv_df.loc[tail_date, "Open"])
    assert any("partial" in str(w.message).lower() or "gap" in str(w.message).lower() for w in caught)


def test_sanitize_called_before_paths(dirs):
    cache, fallback = dirs
    gk = FakeGatekeeper()
    with patch.object(fetcher_mod.yf, "download", side_effect=AssertionError("must not call yf")):
        with pytest.raises(ValueError):
            fetch(
                "../etc/passwd", "2020-01-01", "2020-12-31", gk,
                cache_dir=cache, fallback_dir=fallback, ttl_hours=24,
            )
    assert gk.acquire_calls == 0


def test_returns_columns_and_index(dirs):
    cache, fallback = dirs
    full = _frame(pd.date_range("2020-01-01", "2021-12-31", freq="D"))
    _write_parquet(cache / "AAPL.parquet", full)
    gk = FakeGatekeeper()
    out = fetch(
        "AAPL", "2020-06-01", "2021-06-30", gk,
        cache_dir=cache, fallback_dir=fallback, ttl_hours=24,
    )
    assert isinstance(out.index, pd.DatetimeIndex)
    assert list(out.columns) == OHLCV
