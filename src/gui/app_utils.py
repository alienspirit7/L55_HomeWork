"""Free-function helpers extracted from `src/gui/app.py` to keep MainWindow
under the 150-line budget. No Qt-window state; only tiny utilities.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "output" / "models"


def device_label_text() -> str:
    """Return 'Backend: <CUDA|MPS|CPU>'. Lazy-imports torch so the GUI is
    bootable on a torch-less environment."""
    try:
        from src.utils.device import device_label, pick_device
        return f"Backend: {device_label(pick_device())}"
    except Exception:  # noqa: BLE001
        return "Backend: CPU"


def count_models() -> int:
    """Count *.pt files anywhere under output/models/."""
    if not MODELS_DIR.exists():
        return 0
    return sum(1 for _ in MODELS_DIR.rglob("*.pt"))


def models_loaded_text() -> str:
    return f"Models loaded: {count_models()}"


def save_screenshot(window, path: Path) -> None:
    """Grab the window pixmap and save to `path` as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pix = window.grab()
    pix.save(str(path), "PNG")


def chain_button_enable(prepare_btn, train_btn, backtest_btn, predict_btn,
                        progress_bar, status_bar, candlestick, lbl_models,
                        project_root: Path) -> dict:
    """Build the dict of finish-event slots for the sequential button chain.

    Each slot enables the next button in the pipeline, hides the busy bar,
    refreshes the model count, and writes a transient status message.
    """
    def _on_prepare(payload: dict) -> None:
        train_btn.setEnabled(True)
        progress_bar.hide()
        ticker = payload.get("ticker", "?")
        status_bar.showMessage(f"Prepared {ticker}", 5000)
        lbl_models.setText(models_loaded_text())
        csv_path = project_root / "input" / f"{ticker}.csv"
        if csv_path.exists():
            candlestick.load_csv(csv_path)

    def _on_train(_payload: dict) -> None:
        backtest_btn.setEnabled(True)
        progress_bar.hide()
        status_bar.showMessage("Training complete.", 5000)
        lbl_models.setText(models_loaded_text())

    def _on_backtest(_payload: dict) -> None:
        predict_btn.setEnabled(True)
        progress_bar.hide()
        status_bar.showMessage("Backtest complete.", 5000)

    return {
        "on_prepare_finished": _on_prepare,
        "on_train_finished": _on_train,
        "on_backtest_finished": _on_backtest,
    }
