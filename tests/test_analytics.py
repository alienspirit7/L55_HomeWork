"""Tests for the AnalyticsPanel widget (Task 5.3)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def panel(qtbot):
    from src.gui.analytics import AnalyticsPanel
    p = AnalyticsPanel()
    qtbot.addWidget(p)
    return p


def test_analytics_panel_constructs(panel):
    from PyQt6.QtWidgets import QProgressBar
    bars = panel.findChildren(QProgressBar)
    # Three action bars: HOLD, BUY, SELL
    assert len(bars) >= 3
    assert panel.reasoning_label is not None


def test_set_prediction_argmax(panel):
    # Action ordering: HOLD=0, BUY=1, SELL=2.
    panel.set_prediction(action=1, q_values=np.array([0.2, 1.5, 0.3]))
    assert "BUY" in panel.argmax_label.text().upper()
    # Margin = 1.5 - max(0.2, 0.3) = 1.2
    assert "1.2" in panel.margin_label.text()
    # Soft confidence: softmax([0.2, 1.5, 0.3])[1]
    z = np.array([0.2, 1.5, 0.3]) - 1.5
    p = np.exp(z) / np.exp(z).sum()
    expected_pct = p[1] * 100.0
    txt = panel.softconf_label.text()
    # Must include a percent sign and the rounded value.
    assert "%" in txt
    assert f"{expected_pct:.1f}" in txt


def test_set_prediction_negative_q_values(panel):
    # action 2 = SELL per HOLD=0/BUY=1/SELL=2 ordering. Choose argmax = SELL.
    q = np.array([-0.5, -2.0, -0.1])
    panel.set_prediction(action=2, q_values=q)
    assert "SELL" in panel.argmax_label.text().upper()
    # Margin = q[2] - max(q[0], q[1]) = -0.1 - (-0.5) = 0.4 (always >= 0)
    txt = panel.margin_label.text()
    # Find the numeric value, ensure non-negative
    assert "0.4" in txt
    # No leading minus before the number
    assert "-0.4" not in txt


def test_reasoning_rules_fire(panel):
    panel.set_reasoning({
        "log_return": -2.0,
        "rsi_14": 0.0,
        "macd": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.5,
        "bbp": 1.2,
        "vwap_dist": 0.0,
        "volume_norm": 0.0,
        "position_flag": 0.0,
        "unrealized_pnl_pct": 0.0,
    })
    text = panel.reasoning_label.text().lower()
    assert "macd" in text
    assert "bollinger" in text
    assert "negative" in text


def test_reasoning_neutral_state(panel):
    panel.set_reasoning({k: 0.0 for k in (
        "log_return", "rsi_14", "macd", "macd_signal", "macd_hist", "bbp",
        "vwap_dist", "volume_norm", "position_flag", "unrealized_pnl_pct",
    )})
    text = panel.reasoning_label.text()
    assert "Neutral" in text


def test_reasoning_caveat_present(panel):
    panel.set_reasoning({"macd_hist": 0.5})
    assert "Heuristic" in panel.reasoning_label.text() or \
        "heuristic" in panel.reasoning_label.text()


def test_clear_resets(panel):
    panel.set_prediction(action=1, q_values=np.array([0.2, 1.5, 0.3]))
    panel.clear()
    assert "—" in panel.argmax_label.text() or "-" in panel.argmax_label.text()
