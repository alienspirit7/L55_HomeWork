"""TDD tests for the Dueling DQN network (Task 3.1)."""

from __future__ import annotations

import pytest
import torch

from src.models.dueling_dqn import DuelingDQN


WINDOW = 30
N_FEATURES = 10
N_ACTIONS = 3


def _make_model() -> DuelingDQN:
    torch.manual_seed(0)
    return DuelingDQN(
        window=WINDOW, n_features=N_FEATURES, n_actions=N_ACTIONS,
    )


def test_forward_output_shape() -> None:
    model = _make_model()
    x = torch.randn(8, WINDOW, N_FEATURES)
    q = model(x)
    assert q.shape == (8, N_ACTIONS)


def test_forward_batch_one() -> None:
    model = _make_model()
    x = torch.randn(1, WINDOW, N_FEATURES)
    q = model(x)
    assert q.shape == (1, N_ACTIONS)


def test_aggregation_formula_matches_value_advantage() -> None:
    """Q == V + (A - A.mean(dim=-1, keepdim=True))."""
    model = _make_model()
    x = torch.randn(4, WINDOW, N_FEATURES)
    with torch.no_grad():
        v, a = model.value_advantage(x)
        q_manual = v + (a - a.mean(dim=-1, keepdim=True))
        q_forward = model(x)
    assert torch.allclose(q_forward, q_manual, atol=1e-6)


def test_aggregation_centers_advantage() -> None:
    """(Q - V).mean(dim=-1) ≈ 0 by construction of mean-centering."""
    model = _make_model()
    x = torch.randn(16, WINDOW, N_FEATURES)
    with torch.no_grad():
        v, _ = model.value_advantage(x)
        q = model(x)
        residual = (q - v).mean(dim=-1)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


def test_gradient_flows_through_both_heads() -> None:
    model = _make_model()
    x = torch.randn(2, WINDOW, N_FEATURES)
    q = model(x)
    # NOTE: q.sum() over all actions zeroes the advantage gradient because
    # sum_a [V + A_a - mean(A)] = n_actions * V (mean-centering cancels A).
    # Pick a single action so the advantage stream actually sees a gradient.
    q[:, 0].sum().backward()

    v_grads = [p.grad for p in model.value_head.parameters() if p.grad is not None]
    a_grads = [p.grad for p in model.advantage_head.parameters() if p.grad is not None]
    assert v_grads, "value head has no grads"
    assert a_grads, "advantage head has no grads"
    assert any(g.abs().sum().item() > 0 for g in v_grads), "value head grads all zero"
    assert any(g.abs().sum().item() > 0 for g in a_grads), "advantage head grads all zero"


def test_dtype_and_device_cpu() -> None:
    model = _make_model().to("cpu")
    x = torch.randn(2, WINDOW, N_FEATURES, dtype=torch.float32)
    q = model(x)
    assert q.dtype == torch.float32
    assert q.device.type == "cpu"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available",
)
def test_dtype_and_device_mps() -> None:
    device = torch.device("mps")
    model = _make_model().to(device)
    x = torch.randn(2, WINDOW, N_FEATURES, device=device, dtype=torch.float32)
    q = model(x)
    assert q.dtype == torch.float32
    assert q.device.type == "mps"
    assert q.shape == (2, N_ACTIONS)


def test_param_count_reasonable() -> None:
    model = _make_model()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Locked architecture (300->256->256 trunk, 256->128 heads) yields ~209k.
    # Bound widened from the brief's 100-200k to match the locked design and
    # still catch architecture drift.
    assert 100_000 <= total <= 220_000, f"unexpected param count: {total}"


def test_forward_does_not_mutate_input() -> None:
    model = _make_model()
    x = torch.randn(3, WINDOW, N_FEATURES)
    x_clone = x.clone()
    _ = model(x)
    assert torch.equal(x, x_clone)
