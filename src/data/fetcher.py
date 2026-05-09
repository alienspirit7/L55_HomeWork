"""3-tier OHLCV fetcher with parquet cache, TTL, and CSV fallback.

Tier 1: parquet at ``cache_dir/{ticker}.parquet`` if fresh and covering range.
Tier 2: live yfinance call gated by ``ApiGatekeeper``; merged into parquet.
Tier 3: CSV at ``fallback_dir/{ticker}.csv`` if Tier 2 fails or has gaps.

Cache key is ticker only (full available range is stored; sliced in memory).
TTL is compared against parquet mtime via ``os.path.getmtime``.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf

from src.data.gatekeeper import sanitize_ticker

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataUnavailable(RuntimeError):
    """Raised when no tier could supply data for the requested range."""


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], name="Date"))


def _normalize(df: pd.DataFrame | None) -> pd.DataFrame:
    """Coerce to ``DatetimeIndex`` + capitalized OHLCV columns; drop dupes."""
    if df is None or len(df) == 0:
        return _empty()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns={c: c.capitalize() for c in out.columns if c.capitalize() in OHLCV})
    missing = [c for c in OHLCV if c not in out.columns]
    if missing:
        raise ValueError(f"frame missing OHLCV columns: {missing}")
    out = out[OHLCV]
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = "Date"
    return out[~out.index.duplicated(keep="last")].sort_index()


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df if df.empty else df.loc[(df.index >= start) & (df.index <= end)]


def _covers(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return (not df.empty) and df.index.min() <= start and df.index.max() >= end


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union of date ranges; on duplicate dates keep ``new``."""
    if old.empty:
        return new
    if new.empty:
        return old
    combined = pd.concat([old, new])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def fetch(
    ticker: str,
    start,
    end,
    gatekeeper,
    *,
    cache_dir,
    fallback_dir,
    ttl_hours: float,
    time_fn: Callable[[], float] | None = None,
) -> pd.DataFrame:
    """Return OHLCV DataFrame for ``ticker`` covering ``[start, end]`` inclusive."""
    safe = sanitize_ticker(ticker)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    cache_dir, fallback_dir = Path(cache_dir), Path(fallback_dir)
    now_s = (time_fn or time.time)()
    pq_path = cache_dir / f"{safe}.parquet"
    csv_path = fallback_dir / f"{safe}.csv"

    cached = _normalize(pd.read_parquet(pq_path)) if pq_path.exists() else _empty()
    fresh = pq_path.exists() and (now_s - os.path.getmtime(pq_path)) < ttl_hours * 3600.0
    if fresh and _covers(cached, start_ts, end_ts):
        return _slice(cached, start_ts, end_ts)

    # Tier 2: live yfinance behind the gatekeeper.
    tier2 = _empty()
    tier2_failed = False
    try:
        with gatekeeper.acquire():
            raw = yf.download(
                safe, start=start_ts.strftime("%Y-%m-%d"),
                end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False, auto_adjust=False,
            )
        tier2 = _normalize(raw)
        if not tier2.empty:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = _merge(cached, tier2)
            cached.to_parquet(pq_path)
    except Exception:
        tier2_failed = True

    requested = _slice(cached, start_ts, end_ts)
    if not tier2_failed and not tier2.empty and _covers(cached, start_ts, end_ts):
        return requested

    # Tier 3: CSV fallback (Tier 2 failed entirely, or has gaps in the window).
    if not csv_path.exists():
        if tier2_failed and cached.empty:
            raise DataUnavailable(
                f"no data for {safe}: yfinance failed and {csv_path} is missing"
            )
        if not tier2_failed:
            warnings.warn(
                f"partial yfinance data for {safe}; no CSV fallback at {csv_path}",
                UserWarning, stacklevel=2,
            )
        return requested

    csv_df = _slice(_normalize(pd.read_csv(csv_path, index_col=0, parse_dates=True)), start_ts, end_ts)
    if tier2_failed and cached.empty:
        return csv_df
    # Gap-fill: prefer Tier 2 (cached) where it exists, fill remaining dates from CSV.
    csv_only = csv_df.loc[~csv_df.index.isin(requested.index)]
    if not csv_only.empty:
        warnings.warn(
            f"partial yfinance data for {safe}; filling gap from CSV ({len(csv_only)} rows)",
            UserWarning, stacklevel=2,
        )
    return pd.concat([requested, csv_only]).sort_index()
