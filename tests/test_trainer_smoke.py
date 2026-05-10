"""Smoke tests for the Double DQN Trainer."""
from __future__ import annotations

import copy
import dataclasses
import glob
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn, optim

from src.env.trading_env import TradingEnv
from src.models.dueling_dqn import DuelingDQN
from src.training.replay_buffer import ReplayBuffer
from src.training.trainer import Trainer, TrainingDiverged
from src.utils.config import load_config


def _make_env(window: int = 30, n_features: int = 10, n_bars: int = 600):
    rng = np.random.default_rng(0)
    t = np.arange(n_bars)
    # Sine-wave price series with some noise.
    base = 100.0 + 5.0 * np.sin(t / 8.0) + 0.1 * rng.standard_normal(n_bars)
    opens = base
    closes = base + 0.05 * rng.standard_normal(n_bars)
    feats = rng.standard_normal((n_bars, n_features)).astype(np.float32) * 0.3
    return TradingEnv(
        features=feats, opens=opens, closes=closes,
        window=window, fee_bps=10, init_cash=10000.0, seed=0,
    )


def _override_train_cfg(cfg, **overrides):
    new_train = dataclasses.replace(cfg.train, **overrides)
    return dataclasses.replace(cfg, train=new_train)


def _make_components(seed: int, device: torch.device, cfg, log_dir, ckpt_dir):
    # Seed BEFORE creating the network so weight init is reproducible.
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = _make_env()
    online = DuelingDQN(window=30, n_features=10, n_actions=3).to(device)
    target = DuelingDQN(window=30, n_features=10, n_actions=3).to(device)
    target.load_state_dict(online.state_dict())
    replay = ReplayBuffer(
        capacity=cfg.train.buffer, obs_shape=(30, 10), device=device, seed=seed,
    )
    opt = optim.Adam(online.parameters(), lr=cfg.train.lr)
    return Trainer(
        env=env, online_net=online, target_net=target, replay=replay,
        optimizer=opt, cfg=cfg, device=device,
        ticker="TEST", seed=seed, log_dir=log_dir, ckpt_dir=ckpt_dir,
    )


def test_smoke_short_training_loop(tmp_path: Path) -> None:
    cfg = load_config("config/default.yaml")
    cfg = _override_train_cfg(
        cfg, train_steps=2000, batch=32, buffer=5000,
        target_sync_steps=200, eps_decay_steps=500, eval_every=1000,
    )
    device = torch.device("cpu")
    trainer = _make_components(
        seed=0, device=device, cfg=cfg,
        log_dir=tmp_path / "tb", ckpt_dir=tmp_path / "ck",
    )
    trainer.run()

    assert len(trainer.replay) > 200
    # TB event file exists.
    events = list(Path(tmp_path / "tb").rglob("events.out.tfevents.*"))
    assert events, "no TB event file written"
    # Loss decreased.
    losses = trainer.loss_history
    assert len(losses) > 50
    quartile = len(losses) // 4
    first = sum(losses[:quartile]) / quartile
    last = sum(losses[-quartile:]) / quartile
    assert last < first, f"loss did not decrease: first={first:.5f} last={last:.5f}"
    # Save the numbers for the report — printed by pytest -s.
    print(f"\nloss_first_25%={first:.6f} loss_last_25%={last:.6f}")


def test_target_net_synced_at_correct_steps(tmp_path: Path) -> None:
    cfg = load_config("config/default.yaml")
    cfg = _override_train_cfg(
        cfg, train_steps=210, batch=16, buffer=1000,
        target_sync_steps=100, eps_decay_steps=200, eval_every=10_000,
    )
    device = torch.device("cpu")
    trainer = _make_components(
        seed=0, device=device, cfg=cfg,
        log_dir=tmp_path / "tb", ckpt_dir=tmp_path / "ck",
    )
    # Force quick warmup by lowering it.
    trainer.warmup_steps = 16
    trainer.run()
    # Target should NOT bytewise equal online after divergence following last sync.
    o_state = trainer.online_net.state_dict()
    t_state = trainer.target_net.state_dict()
    same = all(torch.equal(o_state[k], t_state[k]) for k in o_state)
    # After 210 env steps and target_sync_steps=100, the last sync was at grad_step 200,
    # then ~10 grad steps occurred → params should diverge.
    assert not same, "target net unexpectedly identical to online"


def test_nan_guard_triggers_rollback(tmp_path: Path) -> None:
    cfg = load_config("config/default.yaml")
    cfg = _override_train_cfg(
        cfg, train_steps=300, batch=16, buffer=1000,
        target_sync_steps=10_000, eps_decay_steps=200, eval_every=10_000,
    )
    device = torch.device("cpu")
    trainer = _make_components(
        seed=0, device=device, cfg=cfg,
        log_dir=tmp_path / "tb", ckpt_dir=tmp_path / "ck",
    )
    trainer.warmup_steps = 16
    initial_lr = trainer.optimizer.param_groups[0]["lr"]

    nan_calls = {"n": 0}
    real_loss = trainer._compute_loss

    def patched_loss(batch):
        loss, mean_q = real_loss(batch)
        # Force NaN for the first 1 call only -> verifies lr halving.
        if nan_calls["n"] < 1:
            nan_calls["n"] += 1
            nan_loss = loss * float("nan")
            return nan_loss, mean_q
        return loss, mean_q

    trainer._compute_loss = patched_loss
    trainer.run()
    assert nan_calls["n"] == 1
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr / 2)

    # Now force 3 NaN events total → should raise.
    cfg2 = _override_train_cfg(cfg, train_steps=300)
    trainer2 = _make_components(
        seed=0, device=device, cfg=cfg2,
        log_dir=tmp_path / "tb2", ckpt_dir=tmp_path / "ck2",
    )
    trainer2.warmup_steps = 16
    real2 = trainer2._compute_loss
    counter = {"n": 0}

    def always_nan(batch):
        loss, mean_q = real2(batch)
        counter["n"] += 1
        return loss * float("nan"), mean_q

    trainer2._compute_loss = always_nan
    with pytest.raises(TrainingDiverged):
        trainer2.run()


def test_checkpoint_saved_at_eval_cadence(tmp_path: Path) -> None:
    cfg = load_config("config/default.yaml")
    cfg = _override_train_cfg(
        cfg, train_steps=400, batch=16, buffer=1000,
        target_sync_steps=10_000, eps_decay_steps=200, eval_every=200,
    )
    device = torch.device("cpu")
    trainer = _make_components(
        seed=0, device=device, cfg=cfg,
        log_dir=tmp_path / "tb", ckpt_dir=tmp_path / "ck",
    )
    trainer.warmup_steps = 16
    trainer.run()
    ckpts = glob.glob(str(tmp_path / "ck" / "TEST_seed0_step*.pt"))
    assert ckpts, "no step-checkpoint saved"
    state = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    for k in (
        "online_state_dict", "target_state_dict", "optimizer_state_dict",
        "step", "seed", "ticker", "numpy_rng_state", "torch_rng_state", "cfg_dump",
    ):
        assert k in state, f"missing checkpoint key: {k}"


def test_seeded_run_reproducible(tmp_path: Path) -> None:
    cfg = load_config("config/default.yaml")
    cfg = _override_train_cfg(
        cfg, train_steps=100, batch=16, buffer=500,
        target_sync_steps=10_000, eps_decay_steps=200, eval_every=10_000,
    )
    device = torch.device("cpu")
    trainer1 = _make_components(
        seed=42, device=device, cfg=cfg,
        log_dir=tmp_path / "tb1", ckpt_dir=tmp_path / "ck1",
    )
    trainer1.warmup_steps = 9999  # disable optimization → just env stepping
    trainer1.run()
    actions1 = list(trainer1.action_history[:100])

    trainer2 = _make_components(
        seed=42, device=device, cfg=cfg,
        log_dir=tmp_path / "tb2", ckpt_dir=tmp_path / "ck2",
    )
    trainer2.warmup_steps = 9999
    trainer2.run()
    actions2 = list(trainer2.action_history[:100])
    assert actions1 == actions2
