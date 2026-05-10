"""Cross-seed aggregator + Markdown summary renderer.

Pure: numpy + python stdlib only. No torch, no matplotlib here. The equity
plotting helper lives in scripts/run_experiment.py to keep this importable
in headless analysis contexts.

Per-seed entry shape (matches scripts/backtest.py JSON `model` sub-dict plus
seed integer):
    {"seed": int, "metrics": {<metric>: float, ...}, "n_trades": int,
     "final_equity": float}

Aggregation rules:
- mean = np.mean(values), std = np.std(values, ddof=1).
- NaN propagation is honest: if ANY seed reports NaN for a metric, mean and
  std for that metric are also NaN. We do not silently drop NaN seeds.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

__all__ = ["aggregate_seed_results", "render_markdown_summary"]

METRIC_ORDER = (
    "total_return", "sharpe", "sortino", "calmar",
    "max_drawdown", "win_rate", "turnover", "avg_holding_period",
)
PCT_METRICS = {"total_return", "max_drawdown", "win_rate"}
RATIO_METRICS = {"sharpe", "sortino", "calmar"}


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if np.any(np.isnan(arr)):
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return mean, std


def aggregate_seed_results(
    per_seed: Iterable[dict],
    benchmark: dict | None,
) -> dict[str, Any]:
    """Aggregate per-seed metrics into mean/std/values triples.

    Args:
      per_seed: iterable of {"seed", "metrics", "n_trades", "final_equity"}.
      benchmark: single dict (deterministic) or None / empty.

    Returns:
      {"seeds": [...], "model": {metric: {mean, std, values}}, "benchmark": {...}}
    """
    entries = list(per_seed)
    seeds = [int(e["seed"]) for e in entries]
    model: dict[str, dict[str, Any]] = {}
    for key in METRIC_ORDER:
        values = [float(e["metrics"].get(key, float("nan"))) for e in entries]
        mean, std = _mean_std(values)
        model[key] = {"mean": mean, "std": std, "values": values}
    # Per-seed transparency rows (n_trades, final_equity).
    n_trades = [int(e.get("n_trades", 0)) for e in entries]
    final_equity = [float(e.get("final_equity", float("nan"))) for e in entries]
    model["_n_trades"] = {"values": n_trades}
    model["_final_equity"] = {"values": final_equity}
    return {
        "seeds": seeds,
        "model": model,
        "benchmark": benchmark if benchmark else {},
    }


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fmt_metric(key: str, value) -> str:
    if _isnan(value):
        return "n/a"
    if key in PCT_METRICS:
        return f"{value*100:+.2f}%"
    if key in RATIO_METRICS:
        return f"{value:.3f}"
    if key == "avg_holding_period":
        return f"{value:.2f} bars"
    return f"{value:.2f}"


def _fmt_mean_std(key: str, mean, std) -> str:
    return "n/a" if _isnan(mean) else f"{_fmt_metric(key, mean)} ± {_fmt_metric(key, std)}"


def _delta(key: str, mm, bv) -> str:
    return "n/a" if _isnan(mm) or _isnan(bv) else _fmt_metric(key, mm - bv)


def render_markdown_summary(
    agg: dict, ticker: str, seed_count: int, *, run_meta: dict | None = None,
) -> str:
    """Render a Markdown summary table from an aggregate dict."""
    bench_metrics = (agg.get("benchmark") or {}).get("metrics", {})
    has_bench = bool(bench_metrics)
    out = [
        f"# Backtest Summary: {ticker}", "",
        f"_{seed_count} seeds, test split, costs included._", "",
        "| Metric | Model (mean ± std) | Benchmark | Δ (model − benchmark) |",
        "|---|---|---|---|",
    ]
    for key in METRIC_ORDER:
        m = agg["model"][key]
        bv = bench_metrics.get(key, float("nan")) if has_bench else float("nan")
        bench_cell = _fmt_metric(key, bv) if has_bench else "n/a"
        out.append(
            f"| {key} | {_fmt_mean_std(key, m['mean'], m['std'])} | "
            f"{bench_cell} | {_delta(key, m['mean'], bv)} |"
        )
    out += ["", "## Per-seed values", "",
            "| Seed | total_return | sharpe | max_dd | n_trades |",
            "|---|---|---|---|---|"]
    for i, seed in enumerate(agg["seeds"]):
        tr = agg["model"]["total_return"]["values"][i]
        sh = agg["model"]["sharpe"]["values"][i]
        dd = agg["model"]["max_drawdown"]["values"][i]
        nt = agg["model"]["_n_trades"]["values"][i]
        out.append(
            f"| {seed} | {_fmt_metric('total_return', tr)} | "
            f"{_fmt_metric('sharpe', sh)} | "
            f"{_fmt_metric('max_drawdown', dd)} | {nt} |"
        )
    out += ["", "_Note: Sharpe annualized at √252; costs of 10 bps per side "
            "included in both model and benchmark._"]
    if run_meta:
        out += ["", "## Run metadata", ""]
        out += [f"- **{k}**: {v}" for k, v in run_meta.items()]
    out.append("")
    return "\n".join(out)
