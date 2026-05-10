"""Structural tests for the candlestick widget (Task 5.2).

The widget reads OHLC straight from ``input/{TICKER}.csv`` (Tier 3 fallback);
the NPZ produced by ``prepare_data.py`` only stores Open/Close arrays.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless Qt: must precede any QApplication import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GREEN = "#26a69a"
RED = "#ef5350"
GRAY = "#888888"


def _write_csv(path: Path, rows: list[tuple[str, float, float, float, float, int]]) -> None:
    lines = ["Date,Open,High,Low,Close,Volume"]
    for d, o, h, l, c, v in rows:
        lines.append(f"{d},{o},{h},{l},{c},{v}")
    path.write_text("\n".join(lines) + "\n")


def _synthetic(n: int) -> list[tuple[str, float, float, float, float, int]]:
    rows = []
    base = 100.0
    for i in range(n):
        o = base + i
        c = o + (1.0 if i % 2 == 0 else -1.0)
        h = max(o, c) + 0.5
        lo = min(o, c) - 0.5
        rows.append((f"2024-01-{(i % 28) + 1:02d}", o, h, lo, c, 1000 + i))
    return rows


@pytest.fixture
def widget(qtbot):
    from src.gui.candlestick import CandlestickWidget
    w = CandlestickWidget()
    qtbot.addWidget(w)
    return w


def test_widget_constructs(widget):
    import pyqtgraph as pg
    # Either a GraphicsLayoutWidget or PlotWidget child must exist.
    has_pg = any(
        isinstance(c, (pg.GraphicsLayoutWidget, pg.PlotWidget))
        for c in widget.findChildren(object)
    )
    assert has_pg, "no pyqtgraph plot host found"
    widget.clear()  # must not raise on empty state


def test_load_csv_renders_known_count(widget, tmp_path):
    csv = tmp_path / "tiny.csv"
    _write_csv(csv, _synthetic(10))
    widget.load_csv(csv, max_bars=10)
    assert widget.bar_count == 10


def test_load_csv_clamps_max_bars(widget, tmp_path):
    csv = tmp_path / "long.csv"
    rows = _synthetic(100)
    _write_csv(csv, rows)
    widget.load_csv(csv, max_bars=50)
    assert widget.bar_count == 50
    # last 50 dates means the rendered range starts at row 50.
    assert widget.first_date == rows[50][0]
    assert widget.last_date == rows[-1][0]


def test_color_assignment(widget, tmp_path):
    csv = tmp_path / "colors.csv"
    # bull, bear, doji, bull
    _write_csv(csv, [
        ("2024-01-01", 10.0, 11.0, 9.5, 11.0, 100),  # bull (close > open)
        ("2024-01-02", 11.0, 11.5, 10.0, 10.0, 100),  # bear (close < open)
        ("2024-01-03", 10.0, 10.5, 9.5, 10.0, 100),  # doji (close == open)
        ("2024-01-04", 10.0, 11.5, 9.8, 11.2, 100),  # bull
    ])
    widget.load_csv(csv, max_bars=10)
    assert widget.bar_count == 4
    assert widget.colors() == [GREEN, RED, GRAY, GREEN]


def test_load_csv_missing_file(widget, tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        widget.load_csv(missing)


def test_main_window_uses_candlestick_widget(qtbot):
    from src.gui.app import MainWindow
    from src.gui.candlestick import CandlestickWidget
    win = MainWindow()
    qtbot.addWidget(win)
    left = win.splitter.widget(0)
    assert isinstance(left, CandlestickWidget), type(left).__name__
