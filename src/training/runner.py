"""Single-seed training runner — pure library function.

`train_one_seed` loads a prepared NPZ for a ticker, builds the env, network,
target net, replay buffer, and Trainer, then runs training. Returns a dict
describing the produced artifacts. CLI wrappers live in `scripts/`.

The runner does NOT mutate `cfg` (frozen). To override `train_steps` for a
smoke run, pass `override_steps`.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import optim

from src.env.trading_env import TradingEnv
from src.models.dueling_dqn import DuelingDQN
from src.training.checkpoint import save_ckpt
from src.training.replay_buffer import ReplayBuffer
from src.training.trainer import Trainer, TrainingDiverged
from src.utils.device import pick_device
from src.utils.seeding import seed_everything


def _load_npz(npz_dir: Path, ticker: str) -> dict[str, Any]:
    path = Path(npz_dir) / f"{ticker}.npz"
    if not path.exists():
        raise FileNotFoundError(f"prepared NPZ not found: {path}")
    with np.load(path, allow_pickle=True) as z:
        return {
            "train": z["train"],
            "train_open": z["train_open"],
            "train_close": z["train_close"],
        }


def _maybe_override_steps(cfg, override_steps: int | None):
    if override_steps is None:
        return cfg
    new_train = dataclasses.replace(cfg.train, train_steps=int(override_steps))
    return dataclasses.replace(cfg, train=new_train)


def train_one_seed(
    ticker: str,
    seed: int,
    cfg: Any,
    *,
    npz_dir: str | Path,
    log_root: str | Path,
    ckpt_root: str | Path,
    override_steps: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train a single seed; return artifact paths.

    Returns: {"ticker", "seed", "final_step", "ckpt_path", "log_dir"}.
    Raises TrainingDiverged with ticker/seed embedded if 3 NaN events fire.
    """
    cfg = _maybe_override_steps(cfg, override_steps)
    npz_dir = Path(npz_dir)
    log_dir = Path(log_root) / ticker / f"seed{seed}"
    ckpt_dir = Path(ckpt_root) / ticker / f"seed{seed}"

    arrays = _load_npz(npz_dir, ticker)
    dev = device if device is not None else pick_device()

    # Seed BEFORE constructing network/buffer so weight init + RNG are seeded.
    seed_everything(seed)

    env = TradingEnv(
        features=arrays["train"],
        opens=arrays["train_open"],
        closes=arrays["train_close"],
        window=int(cfg.data.window),
        fee_bps=int(cfg.env.fee_bps),
        init_cash=float(cfg.env.init_cash),
        seed=seed,
    )

    online = DuelingDQN(
        window=int(cfg.data.window),
        n_features=int(cfg.data.n_features),
        n_actions=int(cfg.env.n_actions),
    ).to(dev)
    target = DuelingDQN(
        window=int(cfg.data.window),
        n_features=int(cfg.data.n_features),
        n_actions=int(cfg.env.n_actions),
    ).to(dev)
    target.load_state_dict(online.state_dict())

    replay = ReplayBuffer(
        capacity=int(cfg.train.buffer),
        obs_shape=(int(cfg.data.window), int(cfg.data.n_features)),
        device=dev,
        seed=seed,
    )
    optimizer = optim.Adam(online.parameters(), lr=float(cfg.train.lr))

    trainer = Trainer(
        env=env, online_net=online, target_net=target, replay=replay,
        optimizer=optimizer, cfg=cfg, device=dev,
        ticker=ticker, seed=seed, log_dir=log_dir, ckpt_dir=ckpt_dir,
    )
    try:
        trainer.run()
    except TrainingDiverged as exc:
        raise TrainingDiverged(
            f"diverged ticker={ticker} seed={seed}: {exc}"
        ) from exc

    final_step = int(cfg.train.train_steps)
    latest = ckpt_dir / f"{ticker}_seed{seed}_latest.pt"
    if not latest.exists():
        # Eval cadence may not land on the final step (or on a short smoke run).
        # Always persist a final ckpt so backtest has something to load.
        cfg_dump = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else {}
        final_path = ckpt_dir / f"{ticker}_seed{seed}_step{final_step}.pt"
        save_ckpt(
            final_path, online, target, optimizer,
            step=final_step, seed=seed, ticker=ticker, cfg_dump=cfg_dump,
        )
        torch.save(
            torch.load(final_path, map_location="cpu", weights_only=False), latest,
        )
    return {
        "ticker": ticker,
        "seed": int(seed),
        "final_step": final_step,
        "ckpt_path": str(latest),
        "log_dir": str(log_dir),
    }
