"""One-off: download daily OHLCV CSVs to input/ for offline reproducibility.

This is the ONLY script allowed to write under input/. After this runs,
input/ is treated as read-only by the rest of the project (per CLAUDE.md).

Direct yfinance call under ApiGatekeeper — fetcher.py's Tier 3 fallback
relies on these CSVs, so we cannot use fetcher.py here.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.gatekeeper import ApiGatekeeper, sanitize_ticker  # noqa: E402

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
MIN_ROWS = 1500


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns MultiIndex columns even for one ticker."""
    if isinstance(df.columns, pd.MultiIndex):
        # Keep first level (e.g. 'Open', 'High', ...) — drop ticker level.
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _validate(df: pd.DataFrame, ticker: str) -> None:
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: yfinance returned empty frame")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: missing columns {missing}; got {list(df.columns)}")
    if len(df) < MIN_ROWS:
        raise RuntimeError(f"{ticker}: only {len(df)} rows, need >= {MIN_ROWS}")


def _download_one(ticker: str, start: str, end: str, gatekeeper: ApiGatekeeper) -> pd.DataFrame:
    end_inclusive = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    with gatekeeper.acquire():
        df = yf.download(
            ticker,
            start=start,
            end=end_inclusive,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    df = _flatten_columns(df)
    df = df[[c for c in REQUIRED_COLS if c in df.columns]]
    df.index.name = "Date"
    _validate(df, ticker)
    return df


def _load_config(path: Path) -> tuple[list[str], str, str]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    tickers = [sanitize_ticker(t) for t in cfg["tickers"]]
    rng = cfg["default_range"]
    return tickers, str(rng["start"]), str(rng["end"])


def main() -> int:
    p = argparse.ArgumentParser(description="Download sample OHLCV CSVs to input/.")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "tickers.yaml")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "input")
    p.add_argument("--force", action="store_true", help="overwrite existing CSVs")
    args = p.parse_args()

    tickers, start, end = _load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    gatekeeper = ApiGatekeeper(
        per_minute=10, per_hour=100, max_concurrent=2, burst=5, burst_window_sec=10
    )

    for ticker in tickers:
        target = args.out / f"{ticker}.csv"
        if target.exists() and target.stat().st_size > 0 and not args.force:
            print(f"{ticker}: skip (exists) -> {target}")
            continue
        df = _download_one(ticker, start, end, gatekeeper)
        df.to_csv(target, index=True)
        first = df.index[0].strftime("%Y-%m-%d")
        last = df.index[-1].strftime("%Y-%m-%d")
        print(f"{ticker}: {len(df)} rows, {first} -> {last}, wrote {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
