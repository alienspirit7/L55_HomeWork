"""Unit tests for the pure-function Double DQN loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.training.dqn_loss import double_dqn_loss


class _StubNet(nn.Module):
    """Returns a fixed (B, n_actions) tensor regardless of input.

    The fixed table is parameterized to allow deterministic unit testing of
    the Double-DQN loss formula without depending on the real Dueling head.
    """

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        # Wrap as a Parameter so .parameters() is non-empty if needed.
        self.table = nn.Parameter(table.clone(), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.table.unsqueeze(0).expand(b, -1).clone()


def _batch(s, a, r, sn, d, device="cpu"):
    return {
        "obs": torch.as_tensor(s, dtype=torch.float32, device=device),
        "act": torch.as_tensor(a, dtype=torch.int64, device=device),
        "rew": torch.as_tensor(r, dtype=torch.float32, device=device),
        "next_obs": torch.as_tensor(sn, dtype=torch.float32, device=device),
        "done": torch.as_tensor(d, dtype=torch.bool, device=device),
    }


def test_terminal_target_equals_reward() -> None:
    online = _StubNet(torch.tensor([1.0, 2.0, 3.0]))
    target = _StubNet(torch.tensor([10.0, 20.0, 30.0]))
    batch = _batch(
        s=[[0.0]], a=[1], r=[0.5], sn=[[0.0]], d=[True],
    )
    # q_sa = 2.0; y = 0.5 (done so target term is masked).
    loss, mean_q = double_dqn_loss(online, target, batch, gamma=0.99, huber_delta=1.0)
    expected = F.smooth_l1_loss(torch.tensor([2.0]), torch.tensor([0.5]), beta=1.0)
    assert torch.isclose(loss, expected)
    assert abs(mean_q - 2.0) < 1e-6


def test_non_terminal_target_uses_online_argmax_and_target_value() -> None:
    # Online: argmax over s' at index 2.
    online = _StubNet(torch.tensor([0.1, 0.2, 5.0]))
    # Target at index 2 = 7.0.
    target = _StubNet(torch.tensor([100.0, 100.0, 7.0]))
    batch = _batch(
        s=[[0.0]], a=[0], r=[1.0], sn=[[0.0]], d=[False],
    )
    # q_sa = online[0] = 0.1; y = 1.0 + 0.99 * 7.0 = 7.93.
    loss, mean_q = double_dqn_loss(online, target, batch, gamma=0.99, huber_delta=1.0)
    expected_y = 1.0 + 0.99 * 7.0
    expected_loss = F.smooth_l1_loss(
        torch.tensor([0.1]), torch.tensor([expected_y]), beta=1.0,
    )
    assert torch.isclose(loss, expected_loss, atol=1e-5)
    assert abs(mean_q - 0.1) < 1e-5


def test_loss_is_huber_at_action_taken() -> None:
    online = _StubNet(torch.tensor([4.0, -2.0, 1.5]))
    target = _StubNet(torch.tensor([0.0, 0.0, 0.0]))
    batch = _batch(s=[[0.0]], a=[2], r=[0.0], sn=[[0.0]], d=[True])
    # q_sa = online[2] = 1.5; y = 0.0; smooth_l1(1.5, 0.0, beta=1.0) = 1.5 - 0.5 = 1.0.
    loss, _ = double_dqn_loss(online, target, batch, gamma=0.99, huber_delta=1.0)
    assert torch.isclose(loss, torch.tensor(1.0))


def test_loss_finite_for_random_inputs() -> None:
    torch.manual_seed(0)
    online = _StubNet(torch.randn(3))
    target = _StubNet(torch.randn(3))
    for _ in range(100):
        b = 32
        batch = _batch(
            s=torch.randn(b, 4),
            a=torch.randint(0, 3, (b,)),
            r=torch.randn(b),
            sn=torch.randn(b, 4),
            d=torch.randint(0, 2, (b,)).bool(),
        )
        loss, mean_q = double_dqn_loss(online, target, batch, gamma=0.99, huber_delta=1.0)
        assert torch.isfinite(loss)
        assert torch.isfinite(torch.tensor(mean_q))


def test_mean_q_returned() -> None:
    online = _StubNet(torch.tensor([1.0, 5.0, 3.0]))
    target = _StubNet(torch.tensor([0.0, 0.0, 0.0]))
    batch = _batch(
        s=[[0.0], [0.0], [0.0]],
        a=[0, 1, 2],
        r=[0.0, 0.0, 0.0],
        sn=[[0.0], [0.0], [0.0]],
        d=[True, True, True],
    )
    # q_sa = [1, 5, 3] -> mean = 3.0.
    _, mean_q = double_dqn_loss(online, target, batch, gamma=0.99, huber_delta=1.0)
    assert abs(mean_q - 3.0) < 1e-6
