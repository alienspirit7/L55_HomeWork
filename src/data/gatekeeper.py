"""ApiGatekeeper: token-bucket rate limiter for outbound API calls.

Two token buckets (per-minute, per-hour), a burst-window guard, and a
concurrency semaphore. All timing uses an injected monotonic clock and
sleep function for testability — the production defaults are
``time.monotonic`` and ``time.sleep``.

The two buckets and the burst window are checked atomically under a
single lock. A caller is admitted only when ALL three constraints permit
it; otherwise the lock is released and the caller sleeps for the maximum
of the three required wait times before re-checking.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Callable, Iterator


class RateLimitExceeded(RuntimeError):
    """Raised when ``acquire(timeout=...)`` exceeds its deadline."""


_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,8}[A-Z0-9]$|^[A-Z0-9]$")


def sanitize_ticker(s: str) -> str:
    """Return a normalized ticker or raise ``ValueError``.

    Rules: strip + uppercase, allow only ``[A-Z0-9.\\-]``, length 1–10,
    reject path separators, ``..``, and leading/trailing dots/dashes.
    """
    if not isinstance(s, str):
        raise ValueError(f"ticker must be str, got {type(s).__name__}")
    cleaned = s.strip().upper()
    if not cleaned:
        raise ValueError("ticker is empty")
    if any(ch in cleaned for ch in (" ", "/", "\\", ";")):
        raise ValueError(f"ticker contains forbidden character: {cleaned!r}")
    if ".." in cleaned:
        raise ValueError(f"ticker contains '..': {cleaned!r}")
    if not _TICKER_RE.match(cleaned):
        raise ValueError(f"ticker fails [A-Z0-9.\\-]{{1,10}} (no leading/trailing . or -): {cleaned!r}")
    return cleaned


class _Bucket:
    """Simple token bucket. Not thread-safe by itself — caller holds the lock."""

    __slots__ = ("capacity", "rate", "tokens", "last")

    def __init__(self, capacity: float, rate: float, now: float) -> None:
        self.capacity = float(capacity)
        self.rate = float(rate)  # tokens per second
        self.tokens = float(capacity)
        self.last = now

    def refill(self, now: float) -> None:
        dt = now - self.last
        if dt > 0:
            self.tokens = min(self.capacity, self.tokens + dt * self.rate)
            self.last = now

    def wait_for_one(self, now: float) -> float:
        """Seconds until one token is available (0 if already)."""
        self.refill(now)
        if self.tokens >= 1.0:
            return 0.0
        if self.rate <= 0:
            return float("inf")
        return (1.0 - self.tokens) / self.rate

    def consume_one(self) -> None:
        self.tokens -= 1.0


class ApiGatekeeper:
    """Token-bucket gatekeeper with burst and concurrency limits."""

    def __init__(
        self,
        per_minute: int,
        per_hour: int,
        max_concurrent: int,
        burst: int,
        burst_window_sec: int,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if min(per_minute, per_hour, max_concurrent, burst, burst_window_sec) <= 0:
            raise ValueError("all gatekeeper limits must be positive")
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max_concurrent)
        now = time_fn()
        self._minute = _Bucket(per_minute, per_minute / 60.0, now)
        self._hour = _Bucket(per_hour, per_hour / 3600.0, now)
        self._burst = burst
        self._burst_window = float(burst_window_sec)
        self._burst_log: deque[float] = deque()
        self._cfg = (per_minute, per_hour, max_concurrent, burst, burst_window_sec)

    def __repr__(self) -> str:
        pm, ph, mc, b, bw = self._cfg
        return f"ApiGatekeeper(per_minute={pm}, per_hour={ph}, max_concurrent={mc}, burst={b}, burst_window_sec={bw})"

    def _wait_required(self, now: float) -> float:
        """Compute seconds to wait for ALL three constraints. 0 ⇒ admit now."""
        wait_m = self._minute.wait_for_one(now)
        wait_h = self._hour.wait_for_one(now)
        # Burst: drop entries older than window, then check size.
        cutoff = now - self._burst_window
        while self._burst_log and self._burst_log[0] <= cutoff:
            self._burst_log.popleft()
        if len(self._burst_log) >= self._burst:
            wait_b = self._burst_log[0] + self._burst_window - now
            wait_b = max(wait_b, 0.0)
        else:
            wait_b = 0.0
        return max(wait_m, wait_h, wait_b)

    @contextmanager
    def acquire(self, timeout: float | None = None) -> Iterator[None]:
        deadline = None if timeout is None else self._time_fn() + timeout
        # 1) Wait for token+burst admission under the state lock.
        while True:
            with self._lock:
                now = self._time_fn()
                wait = self._wait_required(now)
                if wait <= 0.0:
                    self._minute.consume_one()
                    self._hour.consume_one()
                    self._burst_log.append(now)
                    break
            if deadline is not None and self._time_fn() + wait > deadline:
                raise RateLimitExceeded(
                    f"rate limit not satisfied within timeout={timeout}s (need {wait:.3f}s more)"
                )
            self._sleep_fn(wait)
        # 2) Concurrency slot.
        if not self._sem.acquire(timeout=None if deadline is None else max(0.0, deadline - self._time_fn())):
            raise RateLimitExceeded("concurrency slot unavailable within timeout")
        try:
            yield
        finally:
            self._sem.release()
