"""Tests for the multi-seed training runner and orchestrator script."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.training.runner import train_one_seed
from src.training.trainer import TrainingDiverged
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_EXP = PROJECT_ROOT / "scripts" / "run_experiment.py"


def _write_fake_npz(path: Path, *, n_total: int = 300, n_features: int = 10) -> None:
    rng = np.random.default_rng(0)
    t = np.arange(n_total)
    base = 100.0 + 5.0 * np.sin(t / 8.0) + 0.05 * rng.standard_normal(n_total)
    feats = rng.standard_normal((n_total, n_features)).astype(np.float32) * 0.3
    n_train, n_val, n_test = 200, 50, 50
    train = feats[:n_train]
    val = feats[n_train : n_train + n_val]
    test = feats[n_train + n_val :]
    train_open = base[:n_train].astype(np.float32)
    val_open = base[n_train : n_train + n_val].astype(np.float32)
    test_open = base[n_train + n_val :].astype(np.float32)
    train_close = (base[:n_train] + 0.02).astype(np.float32)
    val_close = (base[n_train : n_train + n_val] + 0.02).astype(np.float32)
    test_close = (base[n_train + n_val :] + 0.02).astype(np.float32)
    feature_names = np.array([f"f{i}" for i in range(n_features)])
    norm_state = np.array({"volume_mean": 0.0, "volume_std": 1.0}, dtype=object)
    meta = np.array({"ticker": "FAKE", "n_features": n_features}, dtype=object)
    np.savez(
        path,
        train=train, val=val, test=test,
        train_open=train_open, val_open=val_open, test_open=test_open,
        train_close=train_close, val_close=val_close, test_close=test_close,
        feature_names=feature_names,
        normalizer_state=norm_state,
        meta=meta,
    )


def _tiny_cfg():
    cfg = load_config(PROJECT_ROOT / "config" / "default.yaml")
    new_train = dataclasses.replace(
        cfg.train,
        train_steps=50, batch=8, buffer=200,
        target_sync_steps=20, eps_decay_steps=20, eval_every=25,
    )
    return dataclasses.replace(cfg, train=new_train)


def test_train_one_seed_smoke(tmp_path: Path) -> None:
    npz_dir = tmp_path / "processed"
    npz_dir.mkdir()
    _write_fake_npz(npz_dir / "FAKE.npz")
    cfg = _tiny_cfg()
    log_root = tmp_path / "runs"
    ckpt_root = tmp_path / "models"
    result = train_one_seed(
        ticker="FAKE", seed=0, cfg=cfg,
        npz_dir=npz_dir, log_root=log_root, ckpt_root=ckpt_root,
        device=torch.device("cpu"),
    )
    assert result["ticker"] == "FAKE"
    assert result["seed"] == 0
    assert result["final_step"] == 50
    ckpt_path = Path(result["ckpt_path"])
    assert ckpt_path.exists()
    log_dir = Path(result["log_dir"])
    assert log_dir.exists()
    # TB event file written.
    events = list(log_dir.rglob("events.out.tfevents.*"))
    assert events


def test_train_one_seed_seeded_reproducible(tmp_path: Path) -> None:
    npz_dir = tmp_path / "processed"
    npz_dir.mkdir()
    _write_fake_npz(npz_dir / "FAKE.npz")
    cfg = _tiny_cfg()

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    res_a = train_one_seed(
        ticker="FAKE", seed=7, cfg=cfg,
        npz_dir=npz_dir, log_root=out_a / "runs", ckpt_root=out_a / "models",
        device=torch.device("cpu"),
    )
    res_b = train_one_seed(
        ticker="FAKE", seed=7, cfg=cfg,
        npz_dir=npz_dir, log_root=out_b / "runs", ckpt_root=out_b / "models",
        device=torch.device("cpu"),
    )
    state_a = torch.load(res_a["ckpt_path"], map_location="cpu", weights_only=False)
    state_b = torch.load(res_b["ckpt_path"], map_location="cpu", weights_only=False)
    sd_a = state_a["online_state_dict"]
    sd_b = state_b["online_state_dict"]
    first_key = next(iter(sd_a))
    ta, tb = sd_a[first_key].float(), sd_b[first_key].float()
    assert abs(float(ta.mean()) - float(tb.mean())) < 1e-6
    assert abs(float(ta.std()) - float(tb.std())) < 1e-6


def test_train_one_seed_handles_short_history_npz(tmp_path: Path) -> None:
    npz_dir = tmp_path / "processed"
    npz_dir.mkdir()
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((10, 10)).astype(np.float32)
    np.savez(
        npz_dir / "TINY.npz",
        train=feats, val=feats, test=feats,
        train_open=np.arange(10, dtype=np.float32),
        val_open=np.arange(10, dtype=np.float32),
        test_open=np.arange(10, dtype=np.float32),
        train_close=np.arange(10, dtype=np.float32),
        val_close=np.arange(10, dtype=np.float32),
        test_close=np.arange(10, dtype=np.float32),
        feature_names=np.array([f"f{i}" for i in range(10)]),
        normalizer_state=np.array({}, dtype=object),
        meta=np.array({}, dtype=object),
    )
    cfg = _tiny_cfg()
    with pytest.raises((ValueError, AssertionError)):
        train_one_seed(
            ticker="TINY", seed=0, cfg=cfg,
            npz_dir=npz_dir, log_root=tmp_path / "runs", ckpt_root=tmp_path / "models",
            device=torch.device("cpu"),
        )


def test_run_experiment_continues_after_divergence(tmp_path: Path, monkeypatch) -> None:
    npz_dir = tmp_path / "processed"
    npz_dir.mkdir()
    _write_fake_npz(npz_dir / "FAKE.npz")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Build a small wrapper script that monkeypatches train_one_seed.
    helper = tmp_path / "run_with_divergence.py"
    helper.write_text(
        f"""
import sys, json
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from src.training import runner
from src.training.trainer import TrainingDiverged

real = runner.train_one_seed
def patched(ticker, seed, cfg, *, npz_dir, log_root, ckpt_root, override_steps=None, device=None):
    if seed == 1:
        raise TrainingDiverged(f"forced divergence ticker={{ticker}} seed={{seed}}")
    return real(ticker, seed, cfg, npz_dir=npz_dir, log_root=log_root, ckpt_root=ckpt_root,
                override_steps=override_steps, device=device)
runner.train_one_seed = patched

# Re-import the script's main with patched runner.
import importlib.util
spec = importlib.util.spec_from_file_location("run_experiment_mod", {str(RUN_EXP)!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.train_one_seed = patched  # also patch the local reference

rc = mod.main([
    "--ticker", "FAKE",
    "--seeds", "0", "1", "2",
    "--config", {str(PROJECT_ROOT / 'config' / 'default.yaml')!r},
    "--npz-dir", {str(npz_dir)!r},
    "--output-dir", {str(output_dir)!r},
    "--steps", "50",
])
sys.exit(rc)
"""
    )
    proc = subprocess.run(
        [sys.executable, str(helper)],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    assert proc.returncode != 0, f"expected non-zero exit, got {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    manifest_path = output_dir / "runs" / "FAKE" / "manifest.json"
    assert manifest_path.exists(), proc.stderr
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["seeds"]
    assert len(entries) == 3
    statuses = [e["status"] for e in entries]
    assert statuses == ["ok", "diverged", "ok"], statuses
