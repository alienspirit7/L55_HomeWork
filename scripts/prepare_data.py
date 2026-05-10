"""Prepare a single ticker: fetch -> features -> 70/15/15 split -> NPZ.

Writes ``{out}/{TICKER}.npz`` with keys: train/val/test (float32, 10 cols),
*_dates (ISO ``<U10``), feature_names, normalizer_state, meta.

Exit codes: 0 ok, 2 data unavailable, 3 history too short, 4 bad ticker, 1 other.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import FEATURE_ORDER, Normalizer, compute_market_features  # noqa: E402
from src.data.fetcher import DataUnavailable, fetch  # noqa: E402
from src.data.gatekeeper import ApiGatekeeper, sanitize_ticker  # noqa: E402
from src.data.splits import temporal_split  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_OUT = PROJECT_ROOT / "output" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "raw"
FALLBACK_DIR = PROJECT_ROOT / "input"
HISTORY_MARGIN = 30  # rows above ``window`` required for non-empty splits


def _build_gatekeeper(cfg) -> ApiGatekeeper:
    g = cfg.gatekeeper
    return ApiGatekeeper(
        per_minute=g.rate_limit_per_min,
        per_hour=g.rate_limit_per_hour,
        max_concurrent=g.rate_limit_concurrent,
        burst=g.rate_limit_burst,
        burst_window_sec=g.rate_limit_burst_window_sec,
    )


def _iso(idx) -> np.ndarray:
    return np.asarray([d.strftime("%Y-%m-%d") for d in idx], dtype="<U10")


def _f32(df) -> np.ndarray:
    return df[list(FEATURE_ORDER)].to_numpy(dtype=np.float32, copy=True)


def _short(msg: str) -> int:
    print(f"error: history < window+horizon ({msg})", file=sys.stderr)
    return 3


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Prepare ticker dataset for training.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--start", required=True, help="ISO date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="ISO date YYYY-MM-DD")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        ticker = sanitize_ticker(args.ticker)
    except ValueError as e:
        print(f"error: bad ticker: {e}", file=sys.stderr)
        return 4
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as e:
        print(f"error: cannot load config {args.config}: {e}", file=sys.stderr)
        return 1

    try:
        ohlcv = fetch(
            ticker, args.start, args.end, _build_gatekeeper(cfg),
            cache_dir=CACHE_DIR, fallback_dir=FALLBACK_DIR,
            ttl_hours=cfg.data.cache_ttl_hours,
        )
    except DataUnavailable as e:
        print(f"error: data unavailable for {ticker}: {e}", file=sys.stderr)
        return 2

    min_rows = cfg.data.window + HISTORY_MARGIN
    if len(ohlcv) < min_rows:
        return _short(f"have {len(ohlcv)} rows, need >= {min_rows}")
    features = compute_market_features(ohlcv)
    if len(features) < min_rows:
        return _short(f"post-warmup {len(features)} rows, need >= {min_rows}")

    train_df, val_df, test_df = temporal_split(
        features, cfg.data.split_train, cfg.data.split_val, cfg.data.split_test,
    )
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        return _short("a split is empty")

    norm = Normalizer(volume_window=cfg.data.volume_norm_window).fit(train_df)
    train_t, val_t, test_t = norm.transform(train_df), norm.transform(val_df), norm.transform(test_df)
    # Aligned raw OHLC prices for env execution (next-bar Open) and MTM (Close).
    raw_aligned = ohlcv.loc[features.index]
    train_open = raw_aligned.loc[train_t.index, "Open"].to_numpy(dtype=np.float32)
    val_open = raw_aligned.loc[val_t.index, "Open"].to_numpy(dtype=np.float32)
    test_open = raw_aligned.loc[test_t.index, "Open"].to_numpy(dtype=np.float32)
    train_close = raw_aligned.loc[train_t.index, "Close"].to_numpy(dtype=np.float32)
    val_close = raw_aligned.loc[val_t.index, "Close"].to_numpy(dtype=np.float32)
    test_close = raw_aligned.loc[test_t.index, "Close"].to_numpy(dtype=np.float32)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{ticker}.npz"
    meta = {
        "ticker": ticker,
        "start": str(args.start),
        "end": str(args.end),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(args.config),
        "n_features": int(cfg.data.n_features),
    }
    np.savez(
        npz_path,
        train=_f32(train_t), val=_f32(val_t), test=_f32(test_t),
        train_dates=_iso(train_t.index), val_dates=_iso(val_t.index), test_dates=_iso(test_t.index),
        train_open=train_open, val_open=val_open, test_open=test_open,
        train_close=train_close, val_close=val_close, test_close=test_close,
        feature_names=np.array(list(FEATURE_ORDER)),
        normalizer_state=np.array(norm.state_dict(), dtype=object),
        meta=np.array(meta, dtype=object),
    )
    print(
        f"{ticker}: {len(features)} rows -> "
        f"train {len(train_t)}, val {len(val_t)}, test {len(test_t)} -> {npz_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
