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
    assert "%" in widget.mem_label.text()
    from src.utils.device import device_label, pick_device
    assert device_label(pick_device()) in widget.backend_label.text()


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
