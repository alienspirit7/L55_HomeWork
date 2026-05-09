import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.config import load_config
from src.utils.device import device_label, pick_device
from src.utils.seeding import seed_everything


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO_ROOT / "config" / "default.yaml"


def test_seed_reproducibility():
    seed_everything(42)
    a = torch.randn(3)
    seed_everything(42)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_seed_numpy_and_random():
    seed_everything(7)
    np_a = np.random.rand(3)
    py_a = random.random()
    seed_everything(7)
    np_b = np.random.rand(3)
    py_b = random.random()
    assert np.array_equal(np_a, np_b)
    assert py_a == py_b


def test_pick_device_returns_torch_device():
    d = pick_device()
    assert isinstance(d, torch.device)
    if torch.cuda.is_available():
        assert d.type == "cuda"
    elif torch.backends.mps.is_available():
        assert d.type == "mps"
    else:
        assert d.type == "cpu"


def test_device_label():
    assert device_label(torch.device("cpu")) == "CPU"
    assert device_label(torch.device("mps")) == "MPS"
    assert device_label(torch.device("cuda")) == "CUDA"


def test_load_config_default():
    cfg = load_config(DEFAULT_CFG)
    assert cfg.model.gamma == 0.99
    assert cfg.env.fee_bps == 10
    assert cfg.eval.seeds == [0, 1, 2]
    assert cfg.data.window == 30
    assert cfg.train.batch == 64
    assert cfg.gatekeeper.rate_limit_per_min == 10


def test_load_config_rejects_unknown_top_level(tmp_path):
    bogus = tmp_path / "bad.yaml"
    bogus.write_text(
        "data:\n  window: 30\n  n_features: 10\n  normalization_split: train\n"
        "  volume_norm_window: 60\n  split_train: 0.7\n  split_val: 0.15\n"
        "  split_test: 0.15\n  cache_ttl_hours: 24\n"
        "rogue_section:\n  foo: bar\n"
    )
    with pytest.raises(ValueError, match="unknown"):
        load_config(bogus)
