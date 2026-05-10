"""Run an N-seed training experiment for one ticker, then aggregate.

Trains seeds sequentially (TrainingDiverged is recorded, loop continues).
Writes manifest to {output_dir}/runs/{ticker}/manifest.json. After training,
unless --no-backtest, runs backtest on test split for each surviving seed
and writes aggregate JSON, Markdown summary, and equity PNG into
{output_dir}/analysis/. Exit non-zero on divergence or 0 valid seeds.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.experiment_io import (  # noqa: E402
    backtest_seed, to_aggregator_entry, write_full_artifacts,
)
from src.training.runner import train_one_seed  # noqa: E402
from src.training.trainer import TrainingDiverged  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_NPZ = PROJECT_ROOT / "output" / "processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "output"


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Run multi-seed training experiment.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--backtest", dest="backtest", action="store_true", default=True)
    p.add_argument("--no-backtest", dest="backtest", action="store_false")
    return p.parse_args(argv)


def _run_one(ticker, seed, cfg, *, npz_dir, log_root, ckpt_root, steps):
    started = time.time()
    try:
        res = train_one_seed(
            ticker=ticker, seed=seed, cfg=cfg,
            npz_dir=npz_dir, log_root=log_root, ckpt_root=ckpt_root,
            override_steps=steps,
        )
        res.update(status="ok", error=None)
    except TrainingDiverged as e:
        res = {
            "ticker": ticker, "seed": int(seed), "final_step": None,
            "ckpt_path": None,
            "log_dir": str(Path(log_root) / ticker / f"seed{seed}"),
            "status": "diverged", "error": str(e),
        }
    res["wall_time_sec"] = round(time.time() - started, 2)
    return res


def _backtest_all(args, cfg, entries):
    payloads, agg_entries = [], []
    for e in entries:
        if e["status"] != "ok" or not e.get("ckpt_path"):
            continue
        try:
            p = backtest_seed(
                ticker=args.ticker, seed=int(e["seed"]),
                ckpt_path=e["ckpt_path"], cfg=cfg, npz_dir=args.npz_dir,
            )
            payloads.append(p)
            agg_entries.append(to_aggregator_entry(p))
            e["backtest_status"] = "ok"
        except Exception as ex:  # noqa: BLE001 — defensive
            e["backtest_status"] = "backtest_failed"
            e["backtest_error"] = str(ex)
            print(f"[seed {e['seed']}] backtest failed: {ex}", file=sys.stderr)
    return payloads, agg_entries


def _save_manifest(path: Path, ticker: str, args, entries) -> None:
    path.write_text(json.dumps({
        "ticker": ticker, "config": str(args.config),
        "steps_override": args.steps, "seeds": entries,
    }, indent=2))


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as e:
        print(f"error: cannot load config {args.config}: {e}", file=sys.stderr)
        return 1
    seeds = args.seeds if args.seeds is not None else list(cfg.eval.seeds)
    log_root = args.output_dir / "runs"
    ckpt_root = args.output_dir / "models"
    manifest_dir = log_root / args.ticker
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"

    entries = []
    for s in seeds:
        print(f"[seed {s}] starting...", flush=True)
        res = _run_one(
            args.ticker, int(s), cfg,
            npz_dir=args.npz_dir, log_root=log_root, ckpt_root=ckpt_root,
            steps=args.steps,
        )
        entries.append(res)
        print(f"[seed {s}] {res['status']} in {res['wall_time_sec']:.2f}s", flush=True)
        _save_manifest(manifest_path, args.ticker, args, entries)

    for e in entries:
        print(f"  seed={e['seed']:>3} status={e['status']:<16} "
              f"step={e.get('final_step')} wall={e.get('wall_time_sec', 0):.1f}s")
    print(f"manifest: {manifest_path}")
    diverged = [e for e in entries if e["status"] == "diverged"]

    if args.backtest:
        payloads, agg_entries = _backtest_all(args, cfg, entries)
        if not agg_entries:
            print("error: 0 valid seeds for backtest aggregation", file=sys.stderr)
            return 4
        _save_manifest(manifest_path, args.ticker, args, entries)
        paths = write_full_artifacts(
            ticker=args.ticker, payloads=payloads, agg_entries=agg_entries,
            analysis_dir=args.output_dir / "analysis",
            run_meta={
                "training_steps": args.steps if args.steps is not None else cfg.train.train_steps,
                "config": str(args.config),
                "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "manifest": str(manifest_path),
            },
        )
        print(paths.pop("_md_text"))
        for kind, p in paths.items():
            print(f"wrote ({kind}): {p}")
    return 0 if not diverged else 2


if __name__ == "__main__":
    sys.exit(main())
