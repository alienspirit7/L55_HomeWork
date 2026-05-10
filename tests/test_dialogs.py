"""Tests for src.gui.dialogs.BacktestResultsDialog."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui.dialogs import BacktestResultsDialog  # noqa: E402

PAYLOAD = {
    "ticker": "TEST",
    "seed": 0,
    "split": "test",
    "model": {
        "metrics": {"total_return": 0.123, "sharpe": 1.4,
                    "max_drawdown": -0.07, "win_rate": 0.55},
        "n_trades": 12, "final_equity": 11230.0,
    },
    "benchmark": {
        "metrics": {"total_return": 0.05, "sharpe": 0.3,
                    "max_drawdown": -0.10, "win_rate": 0.0},
        "n_trades": 1, "final_equity": 10500.0,
    },
}


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    json_p = tmp_path / "bt.json"
    json_p.write_text(json.dumps(PAYLOAD))
    png_p = tmp_path / "bt.png"
    from PyQt6.QtGui import QImage
    img = QImage(2, 2, QImage.Format.Format_RGB32)
    img.fill(0xFFFFFFFF)
    assert img.save(str(png_p), "PNG"), "failed to write fixture PNG"
    return json_p, png_p


def test_backtest_dialog_renders_table(tmp_path, qtbot):
    json_p, png_p = _write_artifacts(tmp_path)
    screenshots_dir = tmp_path / "shots"
    dlg = BacktestResultsDialog("TEST", json_p, png_p,
                                screenshots_dir=screenshots_dir)
    qtbot.addWidget(dlg)
    table = dlg.table
    assert table.rowCount() == 2
    assert table.columnCount() == 5
    assert table.verticalHeaderItem(0).text() == "Model"
    assert table.verticalHeaderItem(1).text() == "Buy & Hold"
    # Model row: total_return = +12.30%
    assert "+12.30%" in table.item(0, 0).text()
    assert table.item(0, 4).text() == "12"
    # Benchmark n_trades cell.
    assert table.item(1, 4).text() == "1"


def test_backtest_dialog_auto_saves_screenshot(tmp_path, qtbot):
    json_p, png_p = _write_artifacts(tmp_path)
    shots = tmp_path / "shots"
    dlg = BacktestResultsDialog("NVDA", json_p, png_p, screenshots_dir=shots)
    qtbot.addWidget(dlg)
    target = shots / "equity_curve_NVDA.png"
    assert target.exists(), f"auto-save did not create {target}"
    assert target.read_bytes() == png_p.read_bytes()
