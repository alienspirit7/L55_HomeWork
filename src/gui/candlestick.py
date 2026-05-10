"""Candlestick chart widget (Task 5.2).

Reads OHLCV from ``input/{TICKER}.csv`` directly. Rationale: the NPZ produced
by ``scripts/prepare_data.py`` only stores Open/Close (no High/Low), so the
CSV is the simplest source for full OHLC. CSVs are populated in Task 1.3 and
always available offline.

pyqtgraph 0.13 has no built-in CandlestickItem; we subclass ``GraphicsObject``
following the pattern from upstream examples (``customGraphicsItem.py``).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QPainter, QPicture
from PyQt6.QtWidgets import QVBoxLayout, QWidget

GREEN, RED, GRAY, WICK = "#26a69a", "#ef5350", "#888888", "#444444"


def _color_for(o: float, c: float) -> str:
    return GREEN if c > o else RED if c < o else GRAY


class _CandleItem(pg.GraphicsObject):
    """One QPicture: rect bodies + line wicks."""

    def __init__(self, bars, colors) -> None:
        super().__init__()
        self._picture = QPicture()
        p = QPainter(self._picture)
        w = 0.4
        for i, ((o, h, lo, c), col) in enumerate(zip(bars, colors)):
            p.setPen(pg.mkPen(WICK, width=1))
            p.drawLine(QPointF(i, lo), QPointF(i, h))
            qcol = QColor(col)
            p.setBrush(pg.mkBrush(qcol))
            p.setPen(pg.mkPen(qcol))
            top, bot = max(o, c), min(o, c)
            p.drawRect(QRectF(i - w, bot, w * 2, max(top - bot, 1e-6)))
        p.end()

    def paint(self, painter, option, widget=None) -> None:
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QRectF:
        return QRectF(self._picture.boundingRect())


class _DateAxis(pg.AxisItem):
    """Maps integer x positions to date strings from the loaded CSV."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dates: list[str] = []

    def set_dates(self, dates: Iterable[str]) -> None:
        self._dates = list(dates)

    def tickStrings(self, values, scale, spacing):  # noqa: N802 — pg API
        return [self._dates[int(round(v))] if 0 <= int(round(v)) < len(self._dates)
                else "" for v in values]


class CandlestickWidget(QWidget):
    """OHLC candlestick panel; wires to ``PrepareDataWorker.finished``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "#222")
        self._axis = _DateAxis(orientation="bottom")
        self._glw = pg.GraphicsLayoutWidget()
        self._plot = self._glw.addPlot(axisItems={"bottom": self._axis})
        self._plot.showGrid(x=False, y=True, alpha=0.2)
        self._plot.setLabel("left", "Price ($)")
        self._candle_item: _CandleItem | None = None
        self._colors: list[str] = []
        self._all_rows: list = []
        self._ticker: str = ""
        self.first_date = self.last_date = ""
        self.bar_count = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._glw)
        self._plot.setTitle("No data loaded")

    # ----- public API ----------------------------------------------------
    def clear(self) -> None:
        self._plot.clear()
        self._candle_item = None
        self._colors = []
        self.first_date = self.last_date = ""
        self.bar_count = 0
        self._axis.set_dates([])
        self._plot.setTitle("No data loaded")

    def load_csv(self, path: str | Path, *, max_bars: int = 500) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        rows = self._read_csv(path)
        if max_bars and len(rows) > max_bars:
            rows = rows[-max_bars:]  # keep most recent N bars
        self._render_rows(rows, ticker=path.stem)

    def set_date_range(self, start: str, end: str) -> None:
        if not self._all_rows:
            return
        filtered = [r for r in self._all_rows if start <= r[0] <= end]
        if filtered:
            self._render_rows(filtered, ticker=self._ticker)

    def colors(self) -> list[str]:
        return list(self._colors)

    # ----- internals -----------------------------------------------------
    @staticmethod
    def _read_csv(path: Path) -> list[tuple[str, float, float, float, float]]:
        out = []
        with path.open() as fh:
            for r in csv.DictReader(fh):
                out.append((r["Date"], float(r["Open"]), float(r["High"]),
                            float(r["Low"]), float(r["Close"])))
        return out

    def _render_rows(self, rows, ticker: str) -> None:
        self._plot.clear()
        if not rows:
            self._plot.setTitle("No data loaded")
            return
        dates = [r[0] for r in rows]
        bars = [(o, h, lo, c) for _, o, h, lo, c in rows]
        self._colors = [_color_for(o, c) for (o, _h, _lo, c) in bars]
        self._candle_item = _CandleItem(bars, self._colors)
        self._plot.addItem(self._candle_item)
        self._axis.set_dates(dates)
        self.first_date, self.last_date = dates[0], dates[-1]
        self.bar_count = len(bars)
        self._ticker = ticker
        self._all_rows = list(rows)
        self._plot.setTitle(f"{ticker} — daily OHLC ({dates[0]} -> {dates[-1]})")
        self._plot.enableAutoRange()
