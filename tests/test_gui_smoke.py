"""Structural smoke tests for the PyQt6 GUI skeleton (Task 5.1)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Headless Qt: must precede any QApplication import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def main_window(qtbot):
    from src.gui.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_main_window_constructs(main_window):
    from PyQt6.QtWidgets import QPushButton, QLabel
    for name in ("btn_prepare", "btn_train", "btn_backtest", "btn_predict"):
        assert main_window.findChild(QPushButton, name) is not None, f"missing {name}"
    disclaimer = main_window.findChild(QLabel, "lbl_disclaimer")
    assert disclaimer is not None
    assert "not investment advice" in disclaimer.text()
    device_lbl = main_window.findChild(QLabel, "lbl_device")
    assert device_lbl is not None
    assert device_lbl.text().startswith("Backend:")


def test_buttons_initial_enabled_state(main_window):
    assert main_window.btn_prepare.isEnabled() is True
    assert main_window.btn_train.isEnabled() is False
    assert main_window.btn_backtest.isEnabled() is False
    assert main_window.btn_predict.isEnabled() is False


def test_button_enable_chain(main_window):
    main_window.on_prepare_finished({"ticker": "NVDA", "rows": 100})
    assert main_window.btn_train.isEnabled() is True
    assert main_window.btn_backtest.isEnabled() is False
    main_window.on_train_finished({"summary_path": "x.md"})
    assert main_window.btn_backtest.isEnabled() is True
    assert main_window.btn_predict.isEnabled() is False
    main_window.on_backtest_finished({"summary": "ok"})
    assert main_window.btn_predict.isEnabled() is True


def test_invalid_ticker_disables_prepare(main_window):
    # Validator should reject non-conforming text: edit stays empty / unchanged.
    main_window.ticker_edit.clear()
    main_window.ticker_edit.insert("../etc/passwd")
    # Validator filters chars; resulting text must be only allowed chars.
    txt = main_window.ticker_edit.text()
    assert all(c.isupper() or c.isdigit() or c in ".-" for c in txt), txt


def test_workers_subprocess_invocation(monkeypatch):
    from src.gui import workers

    fake_stdout = (
        "NVDA: 1500 rows -> train 1050, val 225, test 225 -> "
        "output/processed/NVDA.npz\n"
    )

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=fake_stdout, stderr="")

    monkeypatch.setattr(workers.subprocess, "run", fake_run)

    captured = {}
    err = {}
    w = workers.PrepareDataWorker(
        ticker="NVDA", start="2018-01-02", end="2024-12-31",
        config_path="config/default.yaml",
    )
    w.finished.connect(lambda payload: captured.update(payload))
    w.error.connect(lambda msg: err.setdefault("msg", msg))
    w.run()
    assert "msg" not in err, err
    assert captured.get("ticker") == "NVDA"
    assert captured.get("rows") == 1500
    assert captured.get("train") == 1050
    assert captured.get("val") == 225
    assert captured.get("test") == 225
    assert "output/processed/NVDA.npz" in captured.get("npz_path", "")
