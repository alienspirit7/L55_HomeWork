"""Double DQN loss — pure function, easy to unit test.

The Double DQN target decouples action selection (online net) from action
evaluation (target net), which mitigates the maximization bias of vanilla DQN
(van Hasselt et al. 2016). The loss is Huber (smooth L1) at the chosen action:

    a*  = argmax_a Q_online(s', a)
    y   = r + γ · Q_target(s', a*) · (1 − done)
    L   = SmoothL1(Q_online(s, a), y)

The function returns ``(loss_tensor, mean_q_scalar)``. ``mean_q`` is the mean
of Q_online(s, a) for the actions actually taken — useful for TB monitoring of
value drift independent of the loss curve.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def double_dqn_loss(
    online_net: nn.Module,
    target_net: nn.Module,
    batch: dict[str, Tensor],
    gamma: float,
    huber_delta: float,
) -> tuple[Tensor, float]:
    """Compute Double DQN loss for a single SGD step.

    Parameters
    ----------
    online_net, target_net : nn.Module
        Both must accept the obs tensor and return ``(B, n_actions)`` Q-values.
    batch : dict
        Keys: ``obs``, ``act``, ``rew``, ``next_obs``, ``done`` already on the
        same device as the networks.
    gamma : float
        Discount factor (use γ from config, locked at 0.99).
    huber_delta : float
        ``beta`` for ``smooth_l1_loss`` — the transition point between the
        quadratic and linear regions. Locked at 1.0 in this project.
    """
    obs = batch["obs"]
    act = batch["act"]
    rew = batch["rew"]
    next_obs = batch["next_obs"]
    done = batch["done"]

    # Q(s, a) — value of action actually taken; gradient flows here.
    q_all = online_net(obs)
    q_sa = q_all.gather(1, act.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Online net picks the argmax action at s' (Double DQN's "select").
        online_next = online_net(next_obs)
        a_star = online_next.argmax(dim=1, keepdim=True)
        # Target net evaluates that action (Double DQN's "evaluate").
        target_next = target_net(next_obs)
        q_target_s_next = target_next.gather(1, a_star).squeeze(1)
        # Mask out the bootstrap term on terminal transitions.
        not_done = (~done).to(q_target_s_next.dtype)
        y = rew + gamma * q_target_s_next * not_done

    loss = F.smooth_l1_loss(q_sa, y, beta=huber_delta)
    mean_q = float(q_sa.detach().mean().item())
    return loss, mean_q
