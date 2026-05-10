"""Right-panel analytics widget (Task 5.3).

Renders a softmax-Q action gauge with three horizontal progress bars
(HOLD/BUY/SELL) plus argmax-Q margin and "soft confidence" labels per the
locked GUI confidence decision (PLAN.md). Reasoning panel turns the latest
post-normalization feature dict into 3-6 readable bullets via simple rules.

Action ordering matches `src/env/trading_env.py`: HOLD=0, BUY=1, SELL=2.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGroupBox, QLabel, QProgressBar, QVBoxLayout, QWidget,
)

ACTION_NAMES = ("HOLD", "BUY", "SELL")
ACTION_COLORS = {"HOLD": "#888888", "BUY": "#2e8b57", "SELL": "#c0392b"}
CAVEAT = "Heuristic interpretation — not the model's actual decision rationale."

# (feature, low_thr, high_thr, low_msg, high_msg). None disables a side.
_RULES = [
    ("log_return", -1.0, 1.0,
     "Recent return is unusually negative.",
     "Recent return is unusually positive."),
    ("rsi_14", -1.0, 1.0,
     "RSI is below recent norm.", "RSI is above recent norm."),
    ("macd_hist", -0.1, 0.1,
     "MACD histogram negative — bearish momentum.",
     "MACD histogram positive — bullish momentum."),
    ("bbp", 0.0, 1.0,
     "Price near lower Bollinger band.",
     "Price near upper Bollinger band."),
    ("vwap_dist", 0.0, 0.0,
     "Trading below VWAP approximation.",
     "Trading above VWAP approximation."),
    ("volume_norm", None, 1.0, None, "Volume above its train-period mean."),
]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(); e = np.exp(z); return e / e.sum()


def _argmax_margin(q: np.ndarray, a_star: int) -> float:
    return float(q[a_star] - np.delete(q, a_star).max())


def _bar_style(color: str) -> str:
    return ("QProgressBar { border: 1px solid #555; border-radius: 3px;"
            " text-align: center; height: 18px; }"
            f"QProgressBar::chunk {{ background-color: {color}; }}")


def _build_rules(f: dict) -> list[str]:
    out: list[str] = []
    for key, lo, hi, lo_msg, hi_msg in _RULES:
        v = f.get(key, 0.0)
        if lo is not None and lo_msg is not None and v < lo:
            out.append(lo_msg)
        elif hi is not None and hi_msg is not None and v > hi:
            out.append(hi_msg)
    if f.get("position_flag", 0.0) >= 0.5:
        out.append("Currently long position.")
    return out[:6]


class AnalyticsPanel(QWidget):
    """Action gauge + reasoning bullets. See module docstring."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("analytics_panel")
        self.bars: dict[str, QProgressBar] = {}
        self.bar_labels: dict[str, QLabel] = {}
        self._build_ui()
        self.clear()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self); outer.setContentsMargins(6, 6, 6, 6)
        self.asof_label = QLabel("asof: — → next bar"); self.asof_label.setObjectName("asof_label")
        self.asof_label.setStyleSheet("color: #888; font-style: italic;")
        outer.addWidget(self.asof_label)
        gauge = QGroupBox("Action gauge"); gauge.setObjectName("gauge_group")
        glay = QVBoxLayout(gauge)
        for name in ACTION_NAMES:
            row = QLabel(name); row.setObjectName(f"gauge_label_{name}")
            bar = QProgressBar(); bar.setObjectName(f"gauge_bar_{name}")
            bar.setRange(0, 100); bar.setStyleSheet(_bar_style(ACTION_COLORS[name]))
            self.bars[name] = bar; self.bar_labels[name] = row
            glay.addWidget(row); glay.addWidget(bar)
        self.argmax_label = QLabel("—"); self.argmax_label.setObjectName("argmax_label")
        f = QFont(); f.setPointSize(14); f.setBold(True); self.argmax_label.setFont(f)
        self.argmax_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        glay.addWidget(self.argmax_label)

        self.margin_label = QLabel("Argmax-Q margin: —"); self.margin_label.setObjectName("margin_label")
        self.softconf_label = QLabel("Soft confidence: —"); self.softconf_label.setObjectName("softconf_label")
        glay.addWidget(self.margin_label); glay.addWidget(self.softconf_label)
        outer.addWidget(gauge)
        reasoning = QGroupBox("Reasoning"); reasoning.setObjectName("reasoning_group")
        rlay = QVBoxLayout(reasoning)
        self.reasoning_label = QLabel("(no prediction yet)"); self.reasoning_label.setObjectName("reasoning_label")
        self.reasoning_label.setWordWrap(True)
        self.reasoning_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        rlay.addWidget(self.reasoning_label)
        outer.addWidget(reasoning, stretch=1)

    def set_prediction(self, action: int, q_values) -> None:
        """Render gauge from the 3-element Q vector. action is the argmax."""
        q = np.asarray(q_values, dtype=np.float64).reshape(-1)
        if q.size != 3:
            raise ValueError(f"q_values must have 3 elements, got {q.size}")
        probs = _softmax(q)
        for i, name in enumerate(ACTION_NAMES):
            self.bars[name].setValue(int(round(probs[i] * 100)))
            font = self.bar_labels[name].font()
            font.setBold(i == action); self.bar_labels[name].setFont(font)
        a_name = ACTION_NAMES[int(action)]
        self.argmax_label.setText(a_name)
        self.argmax_label.setStyleSheet(f"color: {ACTION_COLORS[a_name]};")
        margin = _argmax_margin(q, int(action))
        self.margin_label.setText(f"Argmax-Q margin: {margin:+.4f}")
        self.softconf_label.setText(
            f"Soft confidence: {probs[int(action)] * 100.0:.1f}%")

    def set_reasoning(self, features_row: dict) -> None:
        """Render bullets from a post-normalization feature dict (z-scores)."""
        bullets = _build_rules(features_row or {})
        if not bullets:
            bullets = ["Neutral state — no strong signals."]
        body = "\n".join(f"• {b}" for b in bullets)
        self.reasoning_label.setText(f"{body}\n\n{CAVEAT}")

    def set_asof(self, asof: str) -> None:
        """Set the asof timestamp shown in the panel header."""
        self.asof_label.setText(f"asof: {asof} → next bar")

    def clear(self) -> None:
        for bar in self.bars.values():
            bar.setValue(0)
        for lbl in self.bar_labels.values():
            f = lbl.font(); f.setBold(False); lbl.setFont(f)
        self.argmax_label.setText("—"); self.argmax_label.setStyleSheet("")
        self.margin_label.setText("Argmax-Q margin: —")
        self.softconf_label.setText("Soft confidence: —")
        self.reasoning_label.setText(f"(no prediction yet)\n\n{CAVEAT}")
