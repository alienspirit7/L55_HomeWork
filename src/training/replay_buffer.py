"""Uniform experience replay buffer (FIFO eviction, preallocated numpy arrays).

Storage lives on CPU host memory; tensors are moved to the requested device
only at sample time. Sampling uses a per-instance numpy Generator seeded at
construction so reproducibility does not depend on global RNG state.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.obs_buf = np.zeros((capacity, *self.obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *self.obs_shape), dtype=np.float32)
        self.act_buf = np.zeros((capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=bool)

        self._cursor = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._cursor
        # np.copyto performs an explicit copy into preallocated storage,
        # so caller-side mutation cannot leak into the buffer.
        np.copyto(self.obs_buf[i], np.asarray(obs, dtype=np.float32))
        np.copyto(self.next_obs_buf[i], np.asarray(next_obs, dtype=np.float32))
        self.act_buf[i] = int(action)
        self.rew_buf[i] = float(reward)
        self.done_buf[i] = bool(done)
        self._cursor = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        obs_arr: np.ndarray,
        act_arr: np.ndarray,
        rew_arr: np.ndarray,
        next_obs_arr: np.ndarray,
        done_arr: np.ndarray,
    ) -> None:
        n = len(act_arr)
        for k in range(n):
            self.add(obs_arr[k], int(act_arr[k]), float(rew_arr[k]), next_obs_arr[k], bool(done_arr[k]))

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        if self._size < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} from buffer of size {self._size}"
            )
        idx = self._rng.integers(0, self._size, size=batch_size)
        target_device = torch.device(device) if device is not None else self.device

        obs = torch.from_numpy(self.obs_buf[idx].copy())
        next_obs = torch.from_numpy(self.next_obs_buf[idx].copy())
        act = torch.from_numpy(self.act_buf[idx].copy())
        rew = torch.from_numpy(self.rew_buf[idx].copy())
        done = torch.from_numpy(self.done_buf[idx].copy())

        if target_device.type != "cpu":
            obs = obs.to(target_device)
            next_obs = next_obs.to(target_device)
            act = act.to(target_device)
            rew = rew.to(target_device)
            done = done.to(target_device)

        return {"obs": obs, "act": act, "rew": rew, "next_obs": next_obs, "done": done}

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "obs_buf": self.obs_buf.copy(),
            "next_obs_buf": self.next_obs_buf.copy(),
            "act_buf": self.act_buf.copy(),
            "rew_buf": self.rew_buf.copy(),
            "done_buf": self.done_buf.copy(),
            "cursor": self._cursor,
            "size": self._size,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["capacity"] != self.capacity or tuple(state["obs_shape"]) != self.obs_shape:
            raise ValueError("state_dict shape/capacity mismatch")
        self.obs_buf = state["obs_buf"].copy()
        self.next_obs_buf = state["next_obs_buf"].copy()
        self.act_buf = state["act_buf"].copy()
        self.rew_buf = state["rew_buf"].copy()
        self.done_buf = state["done_buf"].copy()
        self._cursor = int(state["cursor"])
        self._size = int(state["size"])
        self._rng.bit_generator.state = state["rng_state"]
