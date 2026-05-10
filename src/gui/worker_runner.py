"""QThread/QObject worker runner + button handler factory for MainWindow.

Split from `app_utils.py` to keep both modules under the 150-line budget.
`run_worker` owns thread lifetime; `make_start_handlers` builds the four
button click handlers wiring `MainWindow` to the worker classes.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QMessageBox


def run_worker(window, worker, on_finished, on_error=None) -> None:
    """Spin up `worker` on a fresh QThread owned by `window`.

    Stores the thread + worker on the window so they survive past this call.
    """
    thread = QThread(window)
    worker.moveToThread(thread)
    window._worker_threads = getattr(window, "_worker_threads", [])
    window._worker_threads.append((thread, worker))

    def _cleanup():
        try:
            thread.quit(); thread.wait(2000)
        except Exception:  # noqa: BLE001
            pass

    def _on_ok(payload):
        try:
            on_finished(payload)
        finally:
            _cleanup()

    def _on_err(msg):
        if on_error is not None:
            on_error(msg)
        else:
            QMessageBox.critical(window, "Worker error", str(msg))
            window.progress.hide()
        _cleanup()

    worker.finished.connect(_on_ok)
    worker.error.connect(_on_err)
    thread.started.connect(worker.run)
    window.progress.show()
    thread.start()


def stop_all_workers(window) -> None:
    """Quit any QThreads owned by `window`. Safe on a window with none."""
    for thread, _w in getattr(window, "_worker_threads", []):
        try:
            thread.quit(); thread.wait(2000)
        except Exception:  # noqa: BLE001
            pass
    window._worker_threads = []


def make_start_handlers(window, project_root: Path) -> dict:
    """Build click handlers for the four pipeline buttons.

    Returns dict ``start_prepare/train/backtest/predict`` -> no-arg callables.
    """
    from src.gui.app_utils import apply_prediction
    from src.gui.predict_worker import (
        PredictNextWorker, resolve_latest_ckpt, resolve_npz,
    )
    from src.gui.workers import BacktestWorker, PrepareDataWorker, TrainWorker

    def _t() -> str:
        return window.ticker_edit.text().strip().upper()

    def _start_prepare() -> None:
        run_worker(window, PrepareDataWorker(
            _t(), window.start_edit.date().toString("yyyy-MM-dd"),
            window.end_edit.date().toString("yyyy-MM-dd"), window.config_path,
        ), window.on_prepare_finished)

    def _start_train() -> None:
        run_worker(window, TrainWorker(_t(), [0], None, window.config_path),
                   window.on_train_finished)

    def _start_backtest() -> None:
        ckpt = resolve_latest_ckpt(_t(), project_root)
        if ckpt is None:
            QMessageBox.warning(window, "No checkpoint",
                                f"No *_latest.pt found for {_t()}")
            return
        run_worker(window, BacktestWorker(str(ckpt), _t(), window.config_path),
                   window.on_backtest_finished)

    def _start_predict() -> None:
        ckpt = resolve_latest_ckpt(_t(), project_root)
        npz = resolve_npz(_t(), project_root)
        if ckpt is None or npz is None:
            QMessageBox.warning(window, "Missing artifact",
                                f"Need ckpt and NPZ for {_t()}")
            return
        run_worker(window, PredictNextWorker(_t(), str(ckpt), str(npz),
                                             window.config_path),
                   lambda p: apply_prediction(window, p))

    return {
        "start_prepare": _start_prepare, "start_train": _start_train,
        "start_backtest": _start_backtest, "start_predict": _start_predict,
    }
