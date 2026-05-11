"""Tests for the TelemetryWidget (Task 5.3)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def widget(qtbot):
    from src.gui.telemetry import TelemetryWidget
    w = TelemetryWidget()
    qtbot.addWidget(w)
    return w


def test_telemetry_constructs_and_stops(widget):
    widget.start()
    assert widget._timer.isActive()
    widget.stop()
    assert not widget._timer.isActive()


def test_refresh_updates_labels(widget):
    widget._refresh()
    assert "%" in widget.cpu_label.text()
    assert "MB" in widget.mem_label.text()
    from src.utils.device import device_label, pick_device
    assert device_label(pick_device()) in widget.backend_label.text()
    # When no worker child is running, must show the GUI process.
    assert "GUI PID" in widget.proc_label.text()


def test_refresh_picks_up_worker_child(widget, monkeypatch):
    """Regression: when a workload subprocess is running, telemetry
    must switch to reporting its metrics, not the GUI's."""
    import psutil
    from src.gui import telemetry as tel

    class FakeProc:
        pid = 99999
        def cpu_percent(self, interval=None): return 91.7
        def memory_info(self):
            class M: rss = 620 * 1024 * 1024
            return M()
        def cmdline(self): return ["python", "scripts/run_experiment.py", "--ticker", "TSLA"]

    monkeypatch.setattr(tel, "_find_worker_child", lambda parent: FakeProc())
    widget._refresh()
    assert "Worker PID 99999" in widget.proc_label.text()
    assert "91.7%" in widget.cpu_label.text()
    assert "620 MB" in widget.mem_label.text()


def test_telemetry_no_torch_required_for_construct():
    # Constructor must not import torch at module load time. We simply
    # verify the module imports without any torch calls — we can't easily
    # test laziness, so we check that the module text contains no
    # top-level `import torch`.
    import src.gui.telemetry as tel
    src = Path(tel.__file__).read_text()
    # Find any non-comment, non-string line starting with 'import torch'
    # at column 0.
    bad = [
        ln for ln in src.splitlines()
        if ln.strip().startswith("import torch")
        and not ln.lstrip().startswith("#")
        and ln == ln.lstrip()  # column 0 only — top-level
    ]
    assert not bad, f"telemetry.py imports torch at top level: {bad}"
