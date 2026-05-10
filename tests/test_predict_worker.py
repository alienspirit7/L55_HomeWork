"""Tests for src.gui.predict_worker.PredictNextWorker."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless Qt: must precede QApplication imports.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui.predict_worker import PredictNextWorker  # noqa: E402

CKPT = PROJECT_ROOT / "output" / "models" / "NVDA" / "seed0" / "NVDA_seed0_latest.pt"
NPZ = PROJECT_ROOT / "output" / "processed" / "NVDA.npz"
CFG = PROJECT_ROOT / "config" / "default.yaml"


@pytest.mark.skipif(not CKPT.exists() or not NPZ.exists(),
                    reason="Real NVDA ckpt / NPZ not present")
def test_predict_worker_runs_with_real_ckpt(qtbot):
    w = PredictNextWorker("NVDA", str(CKPT), str(NPZ), str(CFG))
    captured = {}
    err = {}
    w.finished.connect(lambda p: captured.update(p))
    w.error.connect(lambda e: err.setdefault("msg", e))
    w.run()
    assert "msg" not in err, err
    assert captured.get("ticker") == "NVDA"
    assert captured.get("action") in (0, 1, 2)
    q = captured.get("q_values")
    assert isinstance(q, np.ndarray) and q.shape == (3,)
    assert isinstance(captured.get("features_row"), dict)
    assert isinstance(captured.get("asof"), str)


def test_predict_worker_missing_ckpt(tmp_path, qtbot):
    bad_ckpt = tmp_path / "nope.pt"
    npz = tmp_path / "x.npz"
    npz.write_bytes(b"\0")  # exists, but irrelevant — ckpt check is first
    w = PredictNextWorker("X", str(bad_ckpt), str(npz), str(CFG))
    err = {}
    ok = {}
    w.error.connect(lambda e: err.setdefault("msg", e))
    w.finished.connect(lambda p: ok.setdefault("p", p))
    w.run()
    assert "msg" in err
    assert "checkpoint not found" in err["msg"]
    assert "p" not in ok


def test_predict_worker_missing_npz(tmp_path, qtbot):
    # Use real ckpt if available, else a placeholder file (still passes the
    # exists() check before raising on NPZ).
    ckpt = CKPT if CKPT.exists() else (tmp_path / "ckpt.pt")
    if not ckpt.exists():
        ckpt.write_bytes(b"\0")
    bad_npz = tmp_path / "missing.npz"
    w = PredictNextWorker("X", str(ckpt), str(bad_npz), str(CFG))
    err = {}
    w.error.connect(lambda e: err.setdefault("msg", e))
    w.run()
    assert "msg" in err
    assert "NPZ not found" in err["msg"]
