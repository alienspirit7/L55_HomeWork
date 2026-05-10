"""System telemetry widget (Task 5.3).

Refreshes CPU%, process memory %, accelerator memory MB, and backend label
once per second via QTimer. Torch is imported lazily inside `_refresh` so the
module can load (and the widget can construct) on a torch-less environment.
"""
from __future__ import annotations

import psutil
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

REFRESH_MS = 1000


def _accelerator_mem_mb() -> tuple[str, str]:
    """Return (backend_label, accel_mem_str). Lazy-imports torch."""
    try:
        import torch
        from src.utils.device import device_label, pick_device
    except Exception:  # noqa: BLE001
        return "CPU", "—"
    dev = pick_device()
    label = device_label(dev)
    if dev.type == "cuda":
        try:
            mb = torch.cuda.memory_allocated() / (1024 ** 2)
            return label, f"{mb:.1f} MB"
        except Exception:  # noqa: BLE001
            return label, "—"
    if dev.type == "mps":
        try:
            mb = torch.mps.current_allocated_memory() / (1024 ** 2)
            return label, f"{mb:.1f} MB"
        except Exception:  # noqa: BLE001
            return label, "—"
    return label, "—"


class TelemetryWidget(QWidget):
    """Periodic system + accelerator telemetry display."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("telemetry_widget")
        self._proc = psutil.Process()
        # Prime the per-process and global cpu_percent so subsequent
        # interval=None calls return non-blocking deltas.
        psutil.cpu_percent(interval=None)
        try:
            self._proc.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001
            pass

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self._refresh)
        self._refresh()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        box = QGroupBox("System telemetry")
        box.setObjectName("telemetry_group")
        lay = QVBoxLayout(box)

        self.backend_label = QLabel("Backend: —")
        self.backend_label.setObjectName("telemetry_backend")
        self.cpu_label = QLabel("CPU: —%")
        self.cpu_label.setObjectName("telemetry_cpu")
        self.mem_label = QLabel("MEM: —%")
        self.mem_label.setObjectName("telemetry_mem")
        self.accel_label = QLabel("Accelerator: —")
        self.accel_label.setObjectName("telemetry_accel")

        for w in (self.backend_label, self.cpu_label, self.mem_label,
                  self.accel_label):
            lay.addWidget(w)
        outer.addWidget(box)
        outer.addStretch(1)

    # ----- public API ----------------------------------------------------
    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    # ----- slot ----------------------------------------------------------
    def _refresh(self) -> None:
        cpu = psutil.cpu_percent(interval=None)
        try:
            mem = self._proc.memory_percent()
        except Exception:  # noqa: BLE001
            mem = 0.0
        backend, accel = _accelerator_mem_mb()
        self.backend_label.setText(f"Backend: {backend}")
        self.cpu_label.setText(f"CPU: {cpu:.1f}%")
        self.mem_label.setText(f"MEM: {mem:.2f}%")
        self.accel_label.setText(f"Accelerator: {accel}")

    def closeEvent(self, event):  # noqa: N802
        self.stop()
        super().closeEvent(event)
