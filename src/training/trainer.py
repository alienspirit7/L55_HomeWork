"""Double DQN trainer — single seed, single env, TensorBoard logging.

Per PLAN.md: Double-DQN target, hard target sync, linear ε, Huber(δ=1.0),
grad-norm clip, NaN guard with LR halving + in-memory rollback, OOM guard
halving the batch once. See trainer_utils for pure helpers.
"""
from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from src.training.checkpoint import save_ckpt
from src.training.dqn_loss import double_dqn_loss
from src.training.trainer_utils import is_oom, linear_epsilon, restore, snapshot


class TrainingDiverged(RuntimeError):
    """Raised after 3 NaN-loss events; caller decides how to recover."""


class Trainer:
    def __init__(
        self,
        env: Any, online_net: nn.Module, target_net: nn.Module, replay: Any,
        optimizer: optim.Optimizer, cfg: Any, device: torch.device,
        *,
        ticker: str, seed: int, log_dir: str | Path, ckpt_dir: str | Path,
    ) -> None:
        self.env, self.online_net, self.target_net = env, online_net, target_net
        self.replay, self.optimizer, self.cfg, self.device = replay, optimizer, cfg, device
        self.ticker, self.seed = ticker, seed
        self.log_dir, self.ckpt_dir = Path(log_dir), Path(ckpt_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.warmup_steps = max(int(cfg.train.batch), 1000)
        self.batch_size = int(cfg.train.batch)
        self.nan_count = 0
        self.oom_halved = False
        self.action_history: list[int] = []
        self.loss_history: list[float] = []
        self._healthy = snapshot(online_net, target_net, optimizer)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    def _compute_loss(self, batch):
        return double_dqn_loss(
            self.online_net, self.target_net, batch,
            gamma=self.cfg.model.gamma, huber_delta=self.cfg.model.huber_delta,
        )

    def _select_action(self, obs_np: np.ndarray, eps: float, n_actions: int) -> int:
        if random.random() < eps:
            return random.randrange(n_actions)
        with torch.no_grad():
            t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
            return int(self.online_net(t).argmax(dim=1).item())

    def _opt_step(self, batch):
        try:
            loss, mean_q = self._compute_loss(batch)
        except RuntimeError as exc:
            if is_oom(exc) and not self.oom_halved:
                self.oom_halved = True
                self.batch_size = max(1, self.batch_size // 2)
                return None
            raise
        if not torch.isfinite(loss):
            self.nan_count += 1
            restore(self._healthy, self.online_net, self.target_net, self.optimizer)
            for g in self.optimizer.param_groups:
                g["lr"] *= 0.5
            if self.nan_count >= 3:
                raise TrainingDiverged(f"3 NaN events; aborting (last lr={g['lr']})")
            return None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        return float(loss.detach().item()), mean_q

    def _eval_episode(self) -> float:
        obs, _ = self.env.reset()
        equity = float(self.cfg.env.init_cash)
        done = False
        while not done:
            with torch.no_grad():
                t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                a = int(self.online_net(t).argmax(dim=1).item())
            obs, _, term, trunc, info = self.env.step(a)
            equity = float(info.get("equity", equity))
            done = bool(term or trunc)
        return equity

    def _save_step_ckpt(self, step: int) -> None:
        cfg_dump = dataclasses.asdict(self.cfg) if dataclasses.is_dataclass(self.cfg) else {}
        path = self.ckpt_dir / f"{self.ticker}_seed{self.seed}_step{step}.pt"
        save_ckpt(
            path, self.online_net, self.target_net, self.optimizer,
            step=step, seed=self.seed, ticker=self.ticker, cfg_dump=cfg_dump,
        )
        # Cheap "latest" copy (Phase 4 backtest can load this).
        latest = self.ckpt_dir / f"{self.ticker}_seed{self.seed}_latest.pt"
        torch.save(torch.load(path, map_location="cpu", weights_only=False), latest)

    def run(self) -> None:
        cfg = self.cfg
        n_actions = int(cfg.env.n_actions)
        obs, _ = self.env.reset()
        episode_return = 0.0
        for step in range(int(cfg.train.train_steps)):
            eps = linear_epsilon(step, cfg.train.eps_start, cfg.train.eps_end, cfg.train.eps_decay_steps)
            action = self._select_action(obs, eps, n_actions)
            self.action_history.append(action)
            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = bool(term or trunc)
            self.replay.add(obs, action, float(reward), next_obs, done)
            episode_return += float(reward)
            obs = next_obs
            if step % 100 == 0:
                self.writer.add_scalar("epsilon", eps, step)
            if len(self.replay) >= max(self.batch_size, self.warmup_steps):
                batch = self.replay.sample(self.batch_size, device=self.device)
                result = self._opt_step(batch)
                if result is not None:
                    loss_v, mean_q = result
                    self.loss_history.append(loss_v)
                    self.writer.add_scalar("loss", loss_v, step)
                    self.writer.add_scalar("mean_q", mean_q, step)
                    self._healthy = snapshot(self.online_net, self.target_net, self.optimizer)
                if (step + 1) % int(cfg.train.target_sync_steps) == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())
            if done:
                self.writer.add_scalar("episode/return", episode_return, step)
                episode_return = 0.0
                obs, _ = self.env.reset()
            if (step + 1) % int(cfg.train.eval_every) == 0:
                self._save_step_ckpt(step + 1)
                eq = self._eval_episode()
                self.writer.add_scalar("eval/equity", eq, step + 1)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], step + 1)
                obs, _ = self.env.reset()
        self.writer.flush()
        self.writer.close()
