"""Entry point for the Dueling DQN trading GUI.

Usage:
    python scripts/run_gui.py [--config config/default.yaml] [--screenshot PATH]

``--screenshot`` shows the window briefly, grabs it to PATH, and exits — used
to regenerate ``screenshots/gui_skeleton.png`` for the README.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.gui.app import MainWindow  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Launch the trading GUI.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--screenshot", type=Path, default=None,
                   help="Save window grab to this PNG and exit.")
    p.add_argument("--ticker", default=None,
                   help="Ticker to preload (used with --autoload).")
    p.add_argument("--autoload", action="store_true",
                   help="Load input/{TICKER}.csv into the chart on startup.")
    return p.parse_args(argv)


def _capture_and_quit(window: MainWindow, path: Path, app: QApplication) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pix = window.grab()
    pix.save(str(path), "PNG")
    print(f"saved screenshot: {path}")
    app.quit()


def main(argv=None) -> int:
    args = _parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    win = MainWindow()
    win.config_path = str(args.config)
    if args.ticker:
        win.ticker_edit.setText(args.ticker.upper())
    if args.autoload:
        ticker = (args.ticker or win.ticker_edit.text()).upper()
        csv = PROJECT_ROOT / "input" / f"{ticker}.csv"
        if csv.exists():
            win.candlestick.load_csv(csv)
    win.show()
    if args.screenshot:
        QTimer.singleShot(400, lambda: _capture_and_quit(win, args.screenshot, app))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
