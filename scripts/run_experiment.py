"""Run a 3-seed (or N-seed) training experiment for one ticker.

Sequential per-seed runs; failures (TrainingDiverged) are recorded but the
loop continues. Writes a manifest at:
    {output_dir}/runs/{ticker}/manifest.json

Exit code is non-zero if any seed diverged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.runner import train_one_seed  # noqa: E402
from src.training.trainer import TrainingDiverged  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_NPZ = PROJECT_ROOT / "output" / "processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "output"


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Run multi-seed training experiment.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Override cfg.eval.seeds list.")
    p.add_argument("--steps", type=int, default=None,
                   help="Override cfg.train.train_steps (smoke runs).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help="Root for runs/ and models/ subdirs.")
    return p.parse_args(argv)


def _run_one(ticker, seed, cfg, *, npz_dir, log_root, ckpt_root, steps):
    started = time.time()
    try:
        res = train_one_seed(
            ticker=ticker, seed=seed, cfg=cfg,
            npz_dir=npz_dir, log_root=log_root, ckpt_root=ckpt_root,
            override_steps=steps,
        )
        res["status"] = "ok"
        res["error"] = None
    except TrainingDiverged as e:
        res = {
            "ticker": ticker, "seed": int(seed),
            "final_step": None, "ckpt_path": None,
            "log_dir": str(Path(log_root) / ticker / f"seed{seed}"),
            "status": "diverged", "error": str(e),
        }
    res["wall_time_sec"] = round(time.time() - started, 2)
    return res


def _print_summary(entries):
    print("\nSummary:")
    print(f"  {'seed':>6}  {'status':>10}  {'final_step':>12}  {'wall_s':>8}  ckpt")
    for e in entries:
        ckpt = e.get("ckpt_path") or "-"
        fs = e.get("final_step")
        print(
            f"  {e['seed']:>6}  {e['status']:>10}  "
            f"{(str(fs) if fs is not None else '-'):>12}  "
            f"{e.get('wall_time_sec', 0):>8.2f}  {ckpt}"
        )


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
        # Write manifest after every seed for crash-safety.
        manifest = {
            "ticker": args.ticker,
            "config": str(args.config),
            "steps_override": args.steps,
            "seeds": entries,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    _print_summary(entries)
    print(f"\nmanifest: {manifest_path}")
    diverged = [e for e in entries if e["status"] == "diverged"]
    return 0 if not diverged else 2


if __name__ == "__main__":
    sys.exit(main())
