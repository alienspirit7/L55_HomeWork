"""Backtest results dialog. Built from the JSON artifact + equity PNG.

Auto-saves a copy of the equity PNG to ``screenshots/equity_curve_{ticker}.png``
on construction so the README always has fresh assets without requiring a
button click. Tests pass a ``screenshots_dir`` override so they don't pollute
the real ``screenshots/`` folder.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COLS = ("Total Return", "Sharpe", "Max DD", "Win Rate", "N Trades")
_KEYS = ("total_return", "sharpe", "max_drawdown", "win_rate")


def _fmt_pct(v: float) -> str:
    return f"{float(v) * 100:+.2f}%"


def _row_cells(metrics: dict, n_trades: int) -> list[str]:
    return [
        _fmt_pct(metrics.get("total_return", 0.0)),
        f"{float(metrics.get('sharpe', 0.0)):+.2f}",
        f"{float(metrics.get('max_drawdown', 0.0)) * 100:.1f}%",
        f"{float(metrics.get('win_rate', 0.0)) * 100:.1f}%",
        str(int(n_trades)),
    ]


class BacktestResultsDialog(QDialog):
    """Non-modal dialog: metrics table + equity PNG + auto-save footer."""

    def __init__(
        self,
        ticker: str,
        json_path: str | Path,
        png_path: str | Path,
        parent=None,
        screenshots_dir: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Backtest Results — {ticker}")
        self.setObjectName("backtest_results_dialog")
        self.setModal(False)
        self.ticker = ticker
        self.json_path = Path(json_path)
        self.png_path = Path(png_path)
        self.screenshots_dir = Path(screenshots_dir or PROJECT_ROOT / "screenshots")
        self.payload = json.loads(self.json_path.read_text())
        self._build_ui()
        self._auto_save_screenshot()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.table = QTableWidget(2, len(COLS))
        self.table.setObjectName("metrics_table")
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setVerticalHeaderLabels(["Model", "Buy & Hold"])
        self._fill_row(0, self.payload.get("model", {}))
        self._fill_row(1, self.payload.get("benchmark", {}))
        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        if self.png_path.exists():
            pix = QPixmap(str(self.png_path))
            if not pix.isNull():
                lbl = QLabel(); lbl.setObjectName("equity_label")
                lbl.setPixmap(pix.scaledToWidth(640, Qt.TransformationMode.SmoothTransformation))
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(lbl)

        row = QHBoxLayout()
        self.save_btn = QPushButton("Save copy to screenshots/")
        self.save_btn.setObjectName("save_screenshot_btn")
        self.save_btn.clicked.connect(self._auto_save_screenshot)
        row.addStretch(1); row.addWidget(self.save_btn)
        layout.addLayout(row)
        target = self.screenshots_dir / f"equity_curve_{self.ticker}.png"
        self.footer = QLabel(f"Auto-saved to {target}")
        self.footer.setObjectName("dialog_footer")
        self.footer.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self.footer)

    def _fill_row(self, row: int, section: dict) -> None:
        metrics = section.get("metrics", {}) or {}
        cells = _row_cells(metrics, section.get("n_trades", 0))
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, col, item)

    def _auto_save_screenshot(self) -> None:
        if not self.png_path.exists():
            return
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        target = self.screenshots_dir / f"equity_curve_{self.ticker}.png"
        shutil.copyfile(self.png_path, target)
