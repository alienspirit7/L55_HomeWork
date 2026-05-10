"""MainWindow: control bar, candlestick + analytics + telemetry, status bar."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from PyQt6.QtCore import QDate, Qt, QRegularExpression
from PyQt6.QtGui import QFont, QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QDateEdit, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QProgressBar,
    QPushButton, QSplitter, QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from src.gui.analytics import AnalyticsPanel
from src.gui.app_utils import (
    chain_button_enable, count_models, device_label_text, models_loaded_text,
    show_backtest_dialog,
)
from src.gui.candlestick import CandlestickWidget
from src.gui.telemetry import TelemetryWidget
from src.gui.worker_runner import make_start_handlers, stop_all_workers

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "default.yaml")
DISCLAIMER = (
    "Educational project — not investment advice. "
    "yfinance has known data quality issues."
)
TICKER_RE = QRegularExpression(r"[A-Z0-9.\-]{1,10}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Dueling DQN Stock Trading")
        self.resize(1280, 760)
        self.config_path = DEFAULT_CONFIG
        self._build_toolbar()
        self._build_central()
        self._build_status_bar()
        self._wire_chain()
        self.telemetry.start()

    # ----- top control bar -----------------------------------------------
    def _build_toolbar(self) -> None:
        tb = QToolBar("controls"); tb.setMovable(False); self.addToolBar(tb)

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

        tb.addWidget(QLabel(" Ticker: ")); tb.addWidget(self.ticker_edit)
        tb.addWidget(QLabel("  Start: ")); tb.addWidget(self.start_edit)
        tb.addWidget(QLabel("  End: ")); tb.addWidget(self.end_edit)
        tb.addSeparator()

        self.btn_prepare = self._mk_button("btn_prepare", "Prepare Data", True)
        self.btn_train = self._mk_button("btn_train", "Train Model", False)
        self.btn_backtest = self._mk_button("btn_backtest", "Run Backtest", False)
        self.btn_predict = self._mk_button("btn_predict", "Predict Next", False)
        for b in (self.btn_prepare, self.btn_train, self.btn_backtest,
                  self.btn_predict):
            tb.addWidget(b)

    def _mk_button(self, name: str, text: str, enabled: bool) -> QPushButton:
        b = QPushButton(text); b.setObjectName(name); b.setEnabled(enabled)
        return b

    # ----- central area --------------------------------------------------
    def _build_central(self) -> None:
        central = QWidget()
        v = QVBoxLayout(central); v.setContentsMargins(6, 6, 6, 4)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.candlestick = CandlestickWidget()
        self.candlestick.setObjectName("candlestick")
        self.splitter.addWidget(self.candlestick)
        self.splitter.addWidget(self._right_panel())
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

    def _right_panel(self) -> QWidget:
        right = QSplitter(Qt.Orientation.Vertical)
        right.setObjectName("right_panel")
        self.analytics = AnalyticsPanel()
        self.telemetry = TelemetryWidget()
        right.addWidget(self.analytics)
        right.addWidget(self.telemetry)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 1)
        return right

    # ----- status bar ----------------------------------------------------
    def _build_status_bar(self) -> None:
        sb = QStatusBar(); self.setStatusBar(sb)
        self.lbl_device = QLabel(device_label_text())
        self.lbl_device.setObjectName("lbl_device")
        self.lbl_models = QLabel(models_loaded_text())
        self.lbl_models.setObjectName("lbl_models")
        self.progress = QProgressBar(); self.progress.setObjectName("progress")
        self.progress.setRange(0, 0); self.progress.setFixedWidth(140)
        self.progress.hide()
        sb.addPermanentWidget(self.lbl_device)
        sb.addPermanentWidget(self.lbl_models, stretch=1)
        sb.addPermanentWidget(self.progress)

    # ----- slots / wiring -----------------------------------------------
    def _wire_chain(self) -> None:
        self._slots = chain_button_enable(
            self.btn_prepare, self.btn_train, self.btn_backtest,
            self.btn_predict, self.progress, self.statusBar(),
            self.candlestick, self.lbl_models, PROJECT_ROOT,
        )
        self._handlers = make_start_handlers(self, PROJECT_ROOT)
        self.btn_prepare.clicked.connect(self._handlers["start_prepare"])
        self.btn_train.clicked.connect(self._handlers["start_train"])
        self.btn_backtest.clicked.connect(self._handlers["start_backtest"])
        self.btn_predict.clicked.connect(self._handlers["start_predict"])

    def on_prepare_finished(self, payload): self._slots["on_prepare_finished"](payload)
    def on_train_finished(self, payload): self._slots["on_train_finished"](payload)
    def on_backtest_finished(self, payload: dict) -> None:
        self._slots["on_backtest_finished"](payload)
        show_backtest_dialog(self, payload)

    def closeEvent(self, event):  # noqa: N802
        stop_all_workers(self); self.telemetry.stop(); super().closeEvent(event)
