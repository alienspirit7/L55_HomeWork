"""Dueling DQN network.

Architecture (locked in PLAN.md Task 3.1):
    Input: (batch, window, n_features) float32
    Trunk:  flatten -> Linear(hidden) -> ReLU -> Linear(hidden) -> ReLU
    V head: Linear(hidden -> head_hidden) -> ReLU -> Linear(head_hidden -> 1)
    A head: Linear(hidden -> head_hidden) -> ReLU -> Linear(head_hidden -> n_actions)
    Aggregation: Q = V + (A - A.mean(dim=-1, keepdim=True))

The mean-centered aggregation removes the identifiability ambiguity between V
and A — Wang et al. 2016, "Dueling Network Architectures for Deep RL".
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DuelingDQN(nn.Module):
    """Dueling-DQN value network for the trading env.

    Hyperparameters are constructor arguments (defaults match config/default.yaml);
    the forward pass derives shapes from the input tensor and self.input_dim,
    so 30/10/3 are never hardcoded inside forward.
    """

    def __init__(
        self,
        window: int = 30,
        n_features: int = 10,
        n_actions: int = 3,
        hidden: int = 256,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        if window <= 0 or n_features <= 0 or n_actions <= 0:
            raise ValueError("window, n_features, n_actions must be positive")
        if hidden <= 0 or head_hidden <= 0:
            raise ValueError("hidden, head_hidden must be positive")

        self.window = window
        self.n_features = n_features
        self.n_actions = n_actions
        self.input_dim = window * n_features

        # Shared trunk.
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=False),
        )

        # State-value head: V(s) -> scalar.
        self.value_head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(head_hidden, 1),
        )

        # Advantage head: A(s, a) -> vector of size n_actions.
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(head_hidden, n_actions),
        )

    # ------------------------------------------------------------------ utils

    def _features(self, x: Tensor) -> Tensor:
        """Flatten observation and run the shared trunk.

        Accepts either (batch, window, n_features) or pre-flattened
        (batch, window * n_features). Returns (batch, hidden).
        """
        if x.dim() == 3:
            if x.shape[-2:] != (self.window, self.n_features):
                raise ValueError(
                    f"expected input (..., {self.window}, {self.n_features}), "
                    f"got {tuple(x.shape)}",
                )
            x = x.reshape(x.size(0), self.input_dim)
        elif x.dim() == 2:
            if x.size(-1) != self.input_dim:
                raise ValueError(
                    f"expected flat input dim {self.input_dim}, got {x.size(-1)}",
                )
        else:
            raise ValueError(f"expected 2D or 3D tensor, got {x.dim()}D")
        return self.trunk(x)

    def value_advantage(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return raw V (batch, 1) and A (batch, n_actions) — useful for debug/TB."""
        h = self._features(x)
        v = self.value_head(h)
        a = self.advantage_head(h)
        return v, a

    # ------------------------------------------------------------------- API

    def forward(self, x: Tensor) -> Tensor:
        """Q(s, ·) = V(s) + (A(s, ·) - mean_a A(s, a)).

        Shape: (batch, n_actions). Safe for batch=1.
        """
        v, a = self.value_advantage(x)
        a_centered = a - a.mean(dim=-1, keepdim=True)
        return v + a_centered
