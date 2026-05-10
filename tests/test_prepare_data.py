"""Smoke tests for scripts/prepare_data.py CLI.

Forces offline (Tier 3 CSV fallback) by monkeypatching ``yf.download`` to
raise. Uses the locked-in input/NVDA.csv (1761 rows) from Task 1.3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import prepare_data  # noqa: E402
from src.data import fetcher as fetcher_mod  # noqa: E402
from src.data.features import FEATURE_ORDER  # noqa: E402


def _force_offline(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("forced offline")

    monkeypatch.setattr(fetcher_mod.yf, "download", _boom)


def test_prepare_data_offline_smoke(monkeypatch, tmp_path):
    _force_offline(monkeypatch)
    rc = prepare_data.main([
        "--ticker", "NVDA",
        "--start", "2018-01-02",
        "--end", "2024-12-31",
        "--out", str(tmp_path),
    ])
    assert rc == 0
    npz_path = tmp_path / "NVDA.npz"
    assert npz_path.exists(), f"missing NPZ at {npz_path}"

    data = np.load(npz_path, allow_pickle=True)
    expected_keys = {
        "train", "val", "test",
        "train_dates", "val_dates", "test_dates",
        "feature_names", "normalizer_state", "meta",
    }
    assert expected_keys.issubset(set(data.files)), f"missing keys: {expected_keys - set(data.files)}"

    feat_names = list(data["feature_names"])
    assert feat_names == list(FEATURE_ORDER)

    train = data["train"]
    val = data["val"]
    test = data["test"]
    assert train.dtype == np.float32
    assert train.shape[1] == 10
    assert val.shape[1] == 10
    assert test.shape[1] == 10
    assert train.shape[0] > 0 and val.shape[0] > 0 and test.shape[0] > 0
    # Sanity: dates align with rows.
    assert len(data["train_dates"]) == train.shape[0]
    assert len(data["val_dates"]) == val.shape[0]
    assert len(data["test_dates"]) == test.shape[0]


def test_history_too_short(monkeypatch, capsys):
    _force_offline(monkeypatch)
    out_dir = Path(__file__).resolve().parent / "_tmp_out"
    rc = prepare_data.main([
        "--ticker", "NVDA",
        "--start", "2024-12-01",
        "--end", "2024-12-15",
        "--out", str(out_dir),
    ])
    assert rc == 3
    captured = capsys.readouterr()
    assert "history < window+horizon" in captured.err


def test_bad_ticker_rejected(monkeypatch, tmp_path):
    _force_offline(monkeypatch)
    rc = prepare_data.main([
        "--ticker", "../etc/passwd",
        "--start", "2018-01-02",
        "--end", "2024-12-31",
        "--out", str(tmp_path),
    ])
    assert rc != 0
