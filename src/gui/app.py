"""MainWindow skeleton (Task 5.1): control bar, splitter placeholders, status bar, disclaimer."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from PyQt6.QtCore import QDate, Qt, QRegularExpression
from PyQt6.QtGui import QFont, QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QDateEdit, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar,
    QPushButton, QSplitter, QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from src.gui.candlestick import CandlestickWidget

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "default.yaml")
MODELS_DIR = PROJECT_ROOT / "output" / "models"
DISCLAIMER = (
    "Educational project — not investment advice. "
    "yfinance has known data quality issues."
)
TICKER_RE = QRegularExpression(r"[A-Z0-9.\-]{1,10}")


def _device_text() -> str:
    try:
        from src.utils.device import device_label, pick_device
        return f"Backend: {device_label(pick_device())}"
    except Exception:  # noqa: BLE001 — keep GUI bootable without torch
        return "Backend: CPU"


def _count_models() -> int:
    if not MODELS_DIR.exists():
        return 0
    return sum(1 for _ in MODELS_DIR.rglob("*.pt"))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dueling DQN Stock Trading")
        self.resize(1280, 760)
        self.config_path = DEFAULT_CONFIG
        self._build_toolbar()
        self._build_central()
        self._build_status_bar()

    # ----- top control bar -----------------------------------------------
    def _build_toolbar(self) -> None:
        tb = QToolBar("controls")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.ticker_edit = QLineEdit("NVDA")
        self.ticker_edit.setObjectName("ticker_edit")
        self.ticker_edit.setMaxLength(10)
        self.ticker_edit.setValidator(QRegularExpressionValidator(TICKER_RE))
        self.ticker_edit.setFixedWidth(110)

        self.start_edit = QDateEdit(QDate(2018, 1, 2))
        self.start_edit.setObjectName("start_edit")
        self.start_edit.setCalendarPopup(True)
        self.start_edit.setDisplayFormat("yyyy-MM-dd")

        self.end_edit = QDateEdit(QDate(2024, 12, 31))
        self.end_edit.setObjectName("end_edit")
        self.end_edit.setCalendarPopup(True)
        self.end_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_edit.setMaximumDate(QDate(date.today().year + 1, 12, 31))

        tb.addWidget(QLabel(" Ticker: "))
        tb.addWidget(self.ticker_edit)
        tb.addWidget(QLabel("  Start: "))
        tb.addWidget(self.start_edit)
        tb.addWidget(QLabel("  End: "))
        tb.addWidget(self.end_edit)
        tb.addSeparator()

        self.btn_prepare = self._mk_button("btn_prepare", "Prepare Data", True)
        self.btn_train = self._mk_button("btn_train", "Train Model", False)
        self.btn_backtest = self._mk_button("btn_backtest", "Run Backtest", False)
        self.btn_predict = self._mk_button("btn_predict", "Predict Next", False)
        for b in (self.btn_prepare, self.btn_train, self.btn_backtest, self.btn_predict):
            tb.addWidget(b)

    def _mk_button(self, name: str, text: str, enabled: bool) -> QPushButton:
        b = QPushButton(text)
        b.setObjectName(name)
        b.setEnabled(enabled)
        return b

    # ----- central area --------------------------------------------------
    def _build_central(self) -> None:
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(6, 6, 6, 4)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.candlestick = CandlestickWidget()
        self.candlestick.setObjectName("candlestick")
        self.splitter.addWidget(self.candlestick)
        self.splitter.addWidget(self._placeholder("Analytics goes here", "analytics_placeholder"))
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        v.addWidget(self.splitter, stretch=1)

        self.lbl_disclaimer = QLabel(DISCLAIMER)
        self.lbl_disclaimer.setObjectName("lbl_disclaimer")
        f = QFont(); f.setPointSize(10); f.setItalic(True)
        self.lbl_disclaimer.setFont(f)
        self.lbl_disclaimer.setStyleSheet("color: #888; padding: 4px;")
        self.lbl_disclaimer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_disclaimer)
        self.setCentralWidget(central)

    def _placeholder(self, text: str, name: str) -> QWidget:
        w = QWidget(); w.setObjectName(name); lay = QHBoxLayout(w)
        lbl = QLabel(text); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #666; font-size: 14px;"); lay.addWidget(lbl)
        return w

    # ----- status bar ----------------------------------------------------
    def _build_status_bar(self) -> None:
        sb = QStatusBar(); self.setStatusBar(sb)
        self.lbl_device = QLabel(_device_text()); self.lbl_device.setObjectName("lbl_device")
        self.lbl_models = QLabel(f"Models loaded: {_count_models()}"); self.lbl_models.setObjectName("lbl_models")
        self.progress = QProgressBar(); self.progress.setObjectName("progress")
        self.progress.setRange(0, 0); self.progress.setFixedWidth(140); self.progress.hide()
        sb.addPermanentWidget(self.lbl_device); sb.addPermanentWidget(self.lbl_models, stretch=1)
        sb.addPermanentWidget(self.progress)

    # ----- slots (sequential enable; real wiring in Tasks 5.2-5.4) -------
    def on_prepare_finished(self, payload: dict) -> None:
        self.btn_train.setEnabled(True)
        self.progress.hide()
        ticker = payload.get("ticker", "?")
        self.statusBar().showMessage(f"Prepared {ticker}", 5000)
        self.lbl_models.setText(f"Models loaded: {_count_models()}")
        csv_path = PROJECT_ROOT / "input" / f"{ticker}.csv"
        if csv_path.exists():
            self.candlestick.load_csv(csv_path)

    def on_train_finished(self, payload: dict) -> None:
        self.btn_backtest.setEnabled(True)
        self.progress.hide()
        self.statusBar().showMessage("Training complete.", 5000)
        self.lbl_models.setText(f"Models loaded: {_count_models()}")

    def on_backtest_finished(self, payload: dict) -> None:
        self.btn_predict.setEnabled(True)
        self.progress.hide()
        self.statusBar().showMessage("Backtest complete.", 5000)
