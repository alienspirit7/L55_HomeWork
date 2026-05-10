"""CLI: greedy backtest + buy-and-hold benchmark for one (ticker, seed).

Usage:
    python scripts/backtest.py --model PATH/to/ckpt.pt --ticker NVDA \
        [--split test] [--config config/default.yaml] [--out output/backtests]

Writes:
    {out}/{ticker}_seed{N}.json  -- full metrics (model + benchmark + meta)
    {out}/{ticker}_seed{N}_equity.png  -- two equity curves overlaid
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.trading_env import TradingEnv  # noqa: E402
from src.evaluation.backtest import backtest  # noqa: E402
from src.evaluation.benchmark import buy_and_hold  # noqa: E402
from src.models.dueling_dqn import DuelingDQN  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.device import pick_device  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_OUT = PROJECT_ROOT / "output" / "backtests"
DEFAULT_NPZ_DIR = PROJECT_ROOT / "output" / "processed"


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Run greedy backtest + buy-and-hold benchmark.")
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--ticker", required=True)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    return p.parse_args(argv)


def _seed_from_ckpt(ckpt: dict, model_path: Path) -> int:
    if isinstance(ckpt, dict) and "seed" in ckpt:
        return int(ckpt["seed"])
    m = re.search(r"seed(\d+)", model_path.name)
    return int(m.group(1)) if m else 0


def _build_env(npz, split: str, cfg) -> TradingEnv:
    feats = npz[split]
    opens = npz[f"{split}_open"]
    closes = npz[f"{split}_close"]
    return TradingEnv(
        feats, opens, closes,
        window=cfg.data.window,
        fee_bps=cfg.env.fee_bps,
        init_cash=cfg.env.init_cash,
    )


def _plot(out_path: Path, model_eq, bench_eq, ticker: str, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model_eq, label="Model (greedy)", linewidth=1.5)
    ax.plot(bench_eq, label="Buy-and-Hold", linewidth=1.5, alpha=0.8)
    ax.set_title(f"{ticker} seed{seed} — equity curves")
    ax.set_xlabel("Step")
    ax.set_ylabel("Equity (USD)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _summary_line(ticker: str, seed: int, split: str, m_metrics, b_metrics) -> str:
    return (
        f"{ticker} seed{seed} {split}: "
        f"model_total_return={m_metrics['total_return']*100:+.2f}%, "
        f"sharpe={m_metrics['sharpe']:.2f}, "
        f"max_dd={m_metrics['max_drawdown']*100:.1f}% | "
        f"benchmark_total_return={b_metrics['total_return']*100:+.2f}%, "
        f"sharpe={b_metrics['sharpe']:.2f}, "
        f"max_dd={b_metrics['max_drawdown']*100:.1f}%"
    )


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    npz_path = Path(args.npz_dir) / f"{args.ticker}.npz"
    if not npz_path.exists():
        print(f"error: NPZ not found: {npz_path}", file=sys.stderr)
        return 3
    npz = np.load(npz_path, allow_pickle=True)

    device = pick_device()
    ckpt = torch.load(str(args.model), map_location=device, weights_only=False)
    seed = _seed_from_ckpt(ckpt, args.model)
    state = ckpt["online_state_dict"] if isinstance(ckpt, dict) and "online_state_dict" in ckpt else ckpt

    model = DuelingDQN(window=cfg.data.window, n_features=cfg.data.n_features, n_actions=cfg.env.n_actions)
    model.load_state_dict(state)
    model.to(device)

    sharpe_n = cfg.eval.sharpe_annualization
    model_res = backtest(model, _build_env(npz, args.split, cfg), device=device, periods_per_year=sharpe_n)
    bench_res = buy_and_hold(_build_env(npz, args.split, cfg), periods_per_year=sharpe_n)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.ticker}_seed{seed}.json"
    png_path = out_dir / f"{args.ticker}_seed{seed}_equity.png"
    payload = {
        "ticker": args.ticker,
        "seed": seed,
        "split": args.split,
        "model": {
            "metrics": model_res.metrics,
            "n_trades": model_res.n_trades,
            "final_equity": float(model_res.equity[-1]),
        },
        "benchmark": {
            "metrics": bench_res.metrics,
            "n_trades": bench_res.n_trades,
            "final_equity": float(bench_res.equity[-1]),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=float))
    _plot(png_path, model_res.equity, bench_res.equity, args.ticker, seed)
    print(_summary_line(args.ticker, seed, args.split, model_res.metrics, bench_res.metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
