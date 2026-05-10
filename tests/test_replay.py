import numpy as np
import pytest
import torch

from src.training.replay_buffer import ReplayBuffer


OBS_SHAPE = (30, 10)


def _mk_obs(value: float) -> np.ndarray:
    return np.full(OBS_SHAPE, value, dtype=np.float32)


def _fill(buf: ReplayBuffer, n: int, base_reward: float = 0.0) -> None:
    for i in range(n):
        obs = _mk_obs(float(i))
        next_obs = _mk_obs(float(i) + 0.5)
        buf.add(obs, i % 3, base_reward + float(i + 1), next_obs, bool(i % 2))


def test_capacity_fifo_eviction():
    buf = ReplayBuffer(capacity=3, obs_shape=OBS_SHAPE, seed=0)
    for i, r in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        buf.add(_mk_obs(float(i)), 0, r, _mk_obs(float(i)), False)
    assert len(buf) == 3
    rewards = set(buf.rew_buf.tolist())
    assert rewards == {3.0, 4.0, 5.0}
    batch = buf.sample(3)
    sampled = set(batch["rew"].cpu().numpy().tolist())
    assert sampled.issubset({3.0, 4.0, 5.0})


def test_sample_shape_and_dtype():
    buf = ReplayBuffer(capacity=100, obs_shape=OBS_SHAPE, seed=1)
    _fill(buf, 50)
    batch = buf.sample(8)
    assert batch["obs"].shape == (8, 30, 10)
    assert batch["next_obs"].shape == (8, 30, 10)
    assert batch["act"].shape == (8,)
    assert batch["rew"].shape == (8,)
    assert batch["done"].shape == (8,)
    assert batch["obs"].dtype == torch.float32
    assert batch["next_obs"].dtype == torch.float32
    assert batch["act"].dtype == torch.int64
    assert batch["rew"].dtype == torch.float32
    assert batch["done"].dtype == torch.bool


def test_sample_raises_when_underfilled():
    buf = ReplayBuffer(capacity=100, obs_shape=OBS_SHAPE, seed=2)
    _fill(buf, 5)
    with pytest.raises((ValueError, AssertionError)):
        buf.sample(8)


def test_sample_device_cpu():
    buf = ReplayBuffer(capacity=100, obs_shape=OBS_SHAPE, device="cpu", seed=3)
    _fill(buf, 20)
    batch = buf.sample(4)
    for k in ("obs", "act", "rew", "next_obs", "done"):
        assert batch[k].device.type == "cpu"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_sample_device_mps():
    buf = ReplayBuffer(capacity=100, obs_shape=OBS_SHAPE, device="mps", seed=4)
    _fill(buf, 20)
    batch = buf.sample(4)
    for k in ("obs", "act", "rew", "next_obs", "done"):
        assert batch[k].device.type == "mps"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_sample_device_override():
    buf = ReplayBuffer(capacity=100, obs_shape=OBS_SHAPE, device="cpu", seed=5)
    _fill(buf, 20)
    batch = buf.sample(4, device="mps")
    for k in ("obs", "act", "rew", "next_obs", "done"):
        assert batch[k].device.type == "mps"


def test_seeded_sample_is_reproducible():
    buf_a = ReplayBuffer(capacity=100, obs_shape=OBS_SHAPE, seed=42)
    buf_b = ReplayBuffer(capacity=100, obs_shape=OBS_SHAPE, seed=42)
    _fill(buf_a, 50)
    _fill(buf_b, 50)
    for _ in range(3):
        a = buf_a.sample(8)["act"].cpu().numpy()
        b = buf_b.sample(8)["act"].cpu().numpy()
        np.testing.assert_array_equal(a, b)


def test_sample_does_not_mutate_storage():
    buf = ReplayBuffer(capacity=100, obs_shape=OBS_SHAPE, seed=6)
    _fill(buf, 50)
    snap_obs = buf.obs_buf.copy()
    snap_next = buf.next_obs_buf.copy()
    snap_act = buf.act_buf.copy()
    snap_rew = buf.rew_buf.copy()
    snap_done = buf.done_buf.copy()
    for _ in range(20):
        buf.sample(16)
    np.testing.assert_array_equal(snap_obs, buf.obs_buf)
    np.testing.assert_array_equal(snap_next, buf.next_obs_buf)
    np.testing.assert_array_equal(snap_act, buf.act_buf)
    np.testing.assert_array_equal(snap_rew, buf.rew_buf)
    np.testing.assert_array_equal(snap_done, buf.done_buf)


def test_add_does_not_share_memory():
    buf = ReplayBuffer(capacity=10, obs_shape=OBS_SHAPE, seed=7)
    obs = _mk_obs(1.0)
    next_obs = _mk_obs(2.0)
    buf.add(obs, 0, 1.0, next_obs, False)
    obs[0, 0] = 999.0
    next_obs[0, 0] = 999.0
    assert buf.obs_buf[0, 0, 0] == 1.0
    assert buf.next_obs_buf[0, 0, 0] == 2.0


def test_state_dict_roundtrip():
    buf = ReplayBuffer(capacity=50, obs_shape=OBS_SHAPE, seed=11)
    _fill(buf, 30)
    expected = buf.sample(8)["act"].cpu().numpy()

    buf2 = ReplayBuffer(capacity=50, obs_shape=OBS_SHAPE, seed=11)
    _fill(buf2, 30)
    state = buf2.state_dict()

    buf3 = ReplayBuffer(capacity=50, obs_shape=OBS_SHAPE, seed=999)
    buf3.load_state_dict(state)
    got = buf3.sample(8)["act"].cpu().numpy()
    np.testing.assert_array_equal(expected, got)
