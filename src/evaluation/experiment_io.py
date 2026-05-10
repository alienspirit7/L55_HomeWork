"""Helpers for run_experiment.py — backtest dispatch + artifact writing.

Kept separate so scripts/run_experiment.py stays under the 150-line ceiling.
This module imports torch + matplotlib (Agg backend) lazily inside the
functions that need them; pure aggregator stays torch-free.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _result_to_dict(res) -> dict:
    return {
        "metrics": res.metrics,
        "n_trades": res.n_trades,
        "final_equity": float(res.equity[-1]),
        "equity": res.equity.tolist(),
    }


def backtest_seed(
    *, ticker: str, seed: int, ckpt_path: str, cfg, npz_dir: Path,
    split: str = "test",
) -> dict[str, Any]:
    """Run greedy backtest + buy-and-hold for one seed; return JSON-shaped dict."""
    import numpy as np
    import torch
    from src.env.trading_env import TradingEnv
    from src.evaluation.backtest import backtest
    from src.evaluation.benchmark import buy_and_hold
    from src.models.dueling_dqn import DuelingDQN
    from src.utils.device import pick_device

    npz = np.load(Path(npz_dir) / f"{ticker}.npz", allow_pickle=True)
    feats, opens, closes = npz[split], npz[f"{split}_open"], npz[f"{split}_close"]

    def _env():
        return TradingEnv(feats, opens, closes,
                          window=cfg.data.window, fee_bps=cfg.env.fee_bps,
                          init_cash=cfg.env.init_cash)

    device = pick_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("online_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = DuelingDQN(window=cfg.data.window, n_features=cfg.data.n_features,
                       n_actions=cfg.env.n_actions)
    model.load_state_dict(state)
    model.to(device)

    n = cfg.eval.sharpe_annualization
    return {
        "seed": int(seed), "split": split,
        "model": _result_to_dict(backtest(model, _env(), device=device, periods_per_year=n)),
        "benchmark": _result_to_dict(buy_and_hold(_env(), periods_per_year=n)),
    }


def to_aggregator_entry(seed_payload: dict) -> dict:
    """Convert a backtest_seed payload into the per_seed entry shape."""
    m = seed_payload["model"]
    return {
        "seed": int(seed_payload["seed"]),
        "metrics": dict(m["metrics"]),
        "n_trades": int(m["n_trades"]),
        "final_equity": float(m["final_equity"]),
    }


def save_aggregate_json(path: Path, agg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(agg, indent=2, default=float))


def save_summary_md(path: Path, md: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md)


def write_full_artifacts(
    *,
    ticker: str,
    payloads: list[dict],
    agg_entries: list[dict],
    analysis_dir: Path,
    run_meta: dict,
) -> dict[str, Path]:
    """Aggregate, render Markdown, write JSON + MD + equity PNG.

    Returns a dict {kind: path} for printing. Caller decides whether to print
    the markdown to stdout.
    """
    from src.evaluation.aggregate import (
        aggregate_seed_results, render_markdown_summary,
    )

    benchmark = {
        "metrics": payloads[0]["benchmark"]["metrics"],
        "n_trades": payloads[0]["benchmark"]["n_trades"],
        "final_equity": payloads[0]["benchmark"]["final_equity"],
    }
    agg = aggregate_seed_results(agg_entries, benchmark)
    json_path = analysis_dir / f"{ticker}_aggregate.json"
    md_path = analysis_dir / f"{ticker}_summary.md"
    png_path = analysis_dir / f"{ticker}_equity_all_seeds.png"
    md = render_markdown_summary(
        agg, ticker=ticker, seed_count=len(agg_entries), run_meta=run_meta,
    )
    save_aggregate_json(json_path, agg)
    save_summary_md(md_path, md)
    save_equity_plot(
        png_path, per_seed_payloads=payloads,
        benchmark_equity=payloads[0]["benchmark"].get("equity"),
        ticker=ticker,
    )
    return {"json": json_path, "md": md_path, "png": png_path, "_md_text": md}


def save_equity_plot(
    path: Path,
    *,
    per_seed_payloads: list[dict],
    benchmark_equity: list[float] | None,
    ticker: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in per_seed_payloads:
        eq = p["model"].get("equity")
        if eq:
            ax.plot(eq, linewidth=1.2, label=f"seed {p['seed']}")
    if benchmark_equity:
        ax.plot(benchmark_equity, linewidth=1.5, color="black",
                linestyle="--", label="Buy-and-Hold")
    ax.set_title(f"{ticker} — equity curves across seeds")
    ax.set_xlabel("Step")
    ax.set_ylabel("Equity (USD)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
