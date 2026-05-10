"""Train a single seed for one ticker.

Usage:
    python scripts/train.py --ticker NVDA --seed 0 [--steps N] [--config PATH]

Prints the resulting artifact dict as a one-line JSON. Exits non-zero if
training diverges.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.runner import train_one_seed  # noqa: E402
from src.training.trainer import TrainingDiverged  # noqa: E402
from src.utils.config import load_config  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_NPZ = PROJECT_ROOT / "output" / "processed"
DEFAULT_LOG = PROJECT_ROOT / "output" / "runs"
DEFAULT_CKPT = PROJECT_ROOT / "output" / "models"


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Train a single seed.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=None,
                   help="Override cfg.train.train_steps (smoke runs).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--log-root", type=Path, default=DEFAULT_LOG)
    p.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as e:
        print(f"error: cannot load config {args.config}: {e}", file=sys.stderr)
        return 1
    try:
        result = train_one_seed(
            ticker=args.ticker, seed=args.seed, cfg=cfg,
            npz_dir=args.npz_dir, log_root=args.log_root, ckpt_root=args.ckpt_root,
            override_steps=args.steps,
        )
    except TrainingDiverged as e:
        print(f"error: training diverged: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
