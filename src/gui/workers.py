"""QObject-based worker wrappers around the existing CLI scripts.

Use the moveToThread pattern (preferred over QThread subclassing): each
worker is a QObject that runs blocking ``subprocess.run`` calls; the
MainWindow owns a QThread and moves the worker onto it.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable

# "TICKER: N rows -> train T, val V, test S -> path/to/file.npz"
_PREP_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9.\-]+):\s+(?P<rows>\d+)\s+rows\s*->\s*"
    r"train\s+(?P<train>\d+),\s+val\s+(?P<val>\d+),\s+test\s+(?P<test>\d+)"
    r"\s*->\s*(?P<path>\S+)",
    re.MULTILINE,
)


def _last_summary_line(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        if line.strip():
            return line.strip()
    return ""


class _BaseWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def _spawn(self, cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )


class PrepareDataWorker(_BaseWorker):
    def __init__(self, ticker: str, start: str, end: str, config_path: str) -> None:
        super().__init__()
        self.ticker = ticker
        self.start = start
        self.end = end
        self.config_path = config_path

    def run(self) -> None:
        cmd = [
            PYTHON, "scripts/prepare_data.py",
            "--ticker", self.ticker,
            "--start", self.start, "--end", self.end,
            "--config", self.config_path,
        ]
        try:
            cp = self._spawn(cmd)
        except OSError as e:
            self.error.emit(f"prepare_data spawn failed: {e}")
            return
        if cp.returncode != 0:
            self.error.emit(f"prepare_data exit {cp.returncode}: {cp.stderr.strip()}")
            return
        m = _PREP_RE.search(cp.stdout)
        if not m:
            self.error.emit(f"prepare_data: unparseable output: {cp.stdout!r}")
            return
        self.finished.emit({
            "ticker": m.group("ticker"),
            "rows": int(m.group("rows")),
            "train": int(m.group("train")),
            "val": int(m.group("val")),
            "test": int(m.group("test")),
            "npz_path": m.group("path"),
            "stdout": cp.stdout,
        })


class TrainWorker(_BaseWorker):
    def __init__(self, ticker: str, seeds, steps, config_path: str) -> None:
        super().__init__()
        self.ticker = ticker
        self.seeds = list(seeds) if seeds else None
        self.steps = steps
        self.config_path = config_path

    def run(self) -> None:
        cmd = [PYTHON, "scripts/run_experiment.py",
               "--ticker", self.ticker, "--config", self.config_path]
        if self.seeds:
            cmd += ["--seeds", *[str(s) for s in self.seeds]]
        if self.steps is not None:
            cmd += ["--steps", str(self.steps)]
        try:
            cp = self._spawn(cmd)
        except OSError as e:
            self.error.emit(f"train spawn failed: {e}")
            return
        if cp.returncode not in (0, 2):  # 2 = some seeds diverged but artifacts written
            self.error.emit(f"train exit {cp.returncode}: {cp.stderr.strip()}")
            return
        summary_match = re.search(r"wrote \(summary_md\):\s*(\S+)", cp.stdout)
        self.finished.emit({
            "ticker": self.ticker,
            "summary_path": summary_match.group(1) if summary_match else "",
            "returncode": cp.returncode,
            "stdout": cp.stdout,
        })


class BacktestWorker(_BaseWorker):
    def __init__(self, model_path: str, ticker: str, config_path: str) -> None:
        super().__init__()
        self.model_path = model_path
        self.ticker = ticker
        self.config_path = config_path

    def run(self) -> None:
        cmd = [
            PYTHON, "scripts/backtest.py",
            "--model", self.model_path, "--ticker", self.ticker,
            "--config", self.config_path,
        ]
        try:
            cp = self._spawn(cmd)
        except OSError as e:
            self.error.emit(f"backtest spawn failed: {e}")
            return
        if cp.returncode != 0:
            self.error.emit(f"backtest exit {cp.returncode}: {cp.stderr.strip()}")
            return
        self.finished.emit({
            "ticker": self.ticker,
            "summary": _last_summary_line(cp.stdout),
            "stdout": cp.stdout,
        })
