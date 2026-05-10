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
    p.add_argument("--demo-prediction", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--predict-on-load", action="store_true",
                   help="Trigger Predict Next after autoload.")
    p.add_argument("--backtest-on-load", action="store_true",
                   help="Trigger Run Backtest after autoload.")
    return p.parse_args(argv)


def _apply_demo_prediction(win) -> None:
    import numpy as np
    win.analytics.set_prediction(action=1, q_values=np.array([0.4, 1.8, 0.6]))
    win.analytics.set_reasoning({
        "log_return": 1.4, "rsi_14": 0.6, "macd_hist": 0.45, "bbp": 1.15,
        "vwap_dist": 0.02, "volume_norm": 1.3, "position_flag": 0.0,
    })


def _capture_and_quit(window: MainWindow, path: Path, app: QApplication) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = getattr(window, "backtest_dialog", None) or window
    pix = target.grab()
    pix.save(str(path), "PNG")
    print(f"saved screenshot: {path}")
    app.quit()


def _do_predict(win, ticker: str) -> None:
    """In-process Predict Next: run worker synchronously, render to panel."""
    from src.gui.app_utils import apply_prediction
    from src.gui.predict_worker import (
        PredictNextWorker, resolve_latest_ckpt, resolve_npz,
    )
    ckpt = resolve_latest_ckpt(ticker, PROJECT_ROOT)
    npz = resolve_npz(ticker, PROJECT_ROOT)
    if ckpt is None or npz is None:
        print(f"predict-on-load: missing ckpt or npz for {ticker}"); return
    w = PredictNextWorker(ticker, str(ckpt), str(npz), str(DEFAULT_CONFIG))
    out = {}
    w.finished.connect(lambda p: out.update(p))
    w.error.connect(lambda e: print(f"predict err: {e}"))
    w.run()
    if out:
        apply_prediction(win, out)


def _do_backtest_dialog(win, ticker: str) -> None:
    """Open the BacktestResultsDialog using existing JSON+PNG artifacts."""
    from src.gui.app_utils import show_backtest_dialog
    base = PROJECT_ROOT / "output" / "backtests" / f"{ticker}_seed0"
    json_p = base.with_suffix(".json"); png_p = Path(f"{base}_equity.png")
    if not json_p.exists() or not png_p.exists():
        print(f"backtest-on-load: missing artifacts for {ticker}"); return
    show_backtest_dialog(win, {"ticker": ticker, "json_path": str(json_p),
                               "png_path": str(png_p)})


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
    if args.demo_prediction:
        _apply_demo_prediction(win)
    ticker = (args.ticker or win.ticker_edit.text()).upper()
    if args.predict_on_load:
        _do_predict(win, ticker)
    if args.backtest_on_load:
        _do_backtest_dialog(win, ticker)
    win.show()
    if args.screenshot:
        QTimer.singleShot(400, lambda: _capture_and_quit(win, args.screenshot, app))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
