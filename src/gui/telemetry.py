"""System telemetry widget.

When a worker subprocess (run_experiment.py / train.py / backtest.py /
prepare_data.py) is running as a child of the GUI, its metrics are shown
instead of the GUI's own — that's where the heavy work happens. On Apple
Silicon (MPS) the unified-memory architecture means MPS allocations are
part of the process RSS, so MEM is reported in MB rather than as a
percentage of host RAM.
"""
from __future__ import annotations

import psutil
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

REFRESH_MS = 1000
WORKER_SCRIPTS = (
    "run_experiment.py", "train.py", "backtest.py", "prepare_data.py",
)


def _backend_label() -> str:
    try:
        from src.utils.device import device_label, pick_device
        return device_label(pick_device())
    except Exception:  # noqa: BLE001
        return "CPU"


def _find_worker_child(parent: psutil.Process) -> psutil.Process | None:
    """Return the first descendant python running a workload script, or None."""
    try:
        kids = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    for child in kids:
        try:
            cmd = child.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(any(s in arg for s in WORKER_SCRIPTS) for arg in cmd):
            return child
    return None


class TelemetryWidget(QWidget):
    """Periodic system + accelerator telemetry display."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("telemetry_widget")
        self._proc = psutil.Process()
        # Cache Process objects so psutil's per-object cpu_percent baseline survives.
        self._proc_cache: dict[int, psutil.Process] = {self._proc.pid: self._proc}
        self._primed: set[int] = set()
        self._prime(self._proc)
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self._refresh)
        self._refresh()

    def _prime(self, proc: psutil.Process) -> None:
        """Prime per-process cpu_percent so subsequent calls return deltas."""
        if proc.pid in self._primed:
            return
        try:
            proc.cpu_percent(interval=None)
            self._primed.add(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        box = QGroupBox("System telemetry")
        box.setObjectName("telemetry_group")
        lay = QVBoxLayout(box)
        self.proc_label = QLabel("Process: —")
        self.proc_label.setObjectName("telemetry_process")
        self.backend_label = QLabel("Backend: —")
        self.backend_label.setObjectName("telemetry_backend")
        self.cpu_label = QLabel("CPU: —%")
        self.cpu_label.setObjectName("telemetry_cpu")
        self.mem_label = QLabel("MEM: — MB")
        self.mem_label.setObjectName("telemetry_mem")
        self.accel_label = QLabel("Accelerator: —")
        self.accel_label.setObjectName("telemetry_accel")
        for w in (self.proc_label, self.backend_label, self.cpu_label,
                  self.mem_label, self.accel_label):
            lay.addWidget(w)
        outer.addWidget(box)
        outer.addStretch(1)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _refresh(self) -> None:
        worker = _find_worker_child(self._proc)
        if worker is not None:
            # Reuse cached object so psutil cpu_percent baseline is preserved.
            target = self._proc_cache.setdefault(worker.pid, worker)
            self._prime(target)
            tag = f"Worker PID {target.pid}"
            # Evict stale worker entries (PIDs of finished children).
            for pid in list(self._proc_cache):
                if pid != self._proc.pid and pid != target.pid:
                    self._proc_cache.pop(pid, None)
                    self._primed.discard(pid)
        else:
            target = self._proc
            tag = f"GUI PID {self._proc.pid}"
        try:
            cpu = target.cpu_percent(interval=None)
            mem_mb = target.memory_info().rss / (1024 ** 2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            cpu, mem_mb = 0.0, 0.0
        backend = _backend_label()
        # Unified memory on MPS / single host RAM: report what kind of memory MEM covers.
        accel_note = "shared with MEM (unified RAM)" if backend == "MPS" else "—"
        self.proc_label.setText(f"Process: {tag}")
        self.backend_label.setText(f"Backend: {backend}")
        self.cpu_label.setText(f"CPU: {cpu:.1f}%")
        self.mem_label.setText(f"MEM: {mem_mb:.0f} MB")
        self.accel_label.setText(f"Accelerator: {accel_note}")

    def closeEvent(self, event):  # noqa: N802
        self.stop()
        super().closeEvent(event)
