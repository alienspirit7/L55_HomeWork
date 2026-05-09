"""Tests for ApiGatekeeper token-bucket rate limiter and sanitize_ticker."""
from __future__ import annotations

import threading
import time

import pytest

from src.data.gatekeeper import ApiGatekeeper, RateLimitExceeded, sanitize_ticker


class FakeClock:
    """Deterministic clock with manual advance and recorded sleeps."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start
        self._lock = threading.Lock()

    def time(self) -> float:
        with self._lock:
            return self.t

    def sleep(self, dt: float) -> None:
        # Treat sleep as deterministic clock advance — no real wait.
        if dt <= 0:
            return
        with self._lock:
            self.t += dt

    def advance(self, dt: float) -> None:
        with self._lock:
            self.t += dt


def _make(**overrides):
    clock = FakeClock()
    defaults = dict(
        per_minute=10,
        per_hour=100,
        max_concurrent=2,
        burst=5,
        burst_window_sec=10,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
    )
    defaults.update(overrides)
    gk = ApiGatekeeper(**defaults)
    return gk, clock


def test_token_bucket_per_minute_refill():
    # per_minute=10 → 10 immediate tokens, 11th waits ~6s for one refill.
    # Use a large per-hour and burst to isolate the per-minute bucket.
    gk, clock = _make(per_minute=10, per_hour=10_000, burst=10_000, burst_window_sec=1)
    for _ in range(10):
        with gk.acquire():
            pass
    # 11th: bucket empty → must sleep ~6s (= 60/10) for one token.
    t_before = clock.time()
    with gk.acquire():
        pass
    elapsed = clock.time() - t_before
    assert 5.5 <= elapsed <= 6.5, f"expected ~6s, got {elapsed}"


def test_hard_cap_per_hour():
    # per_hour=100, per_minute very high so per-min not the binder.
    gk, clock = _make(per_minute=10_000, per_hour=100, burst=10_000, burst_window_sec=1)
    for _ in range(100):
        with gk.acquire():
            pass
    t_before = clock.time()
    with gk.acquire():
        pass
    elapsed = clock.time() - t_before
    # Refill rate = 100/3600 ≈ 0.0278/s → 1 token in 36s.
    assert 35.0 <= elapsed <= 37.0, f"expected ~36s, got {elapsed}"


def test_burst_window():
    # burst=5 in 10s. Generous bucket capacities.
    gk, clock = _make(per_minute=10_000, per_hour=10_000, burst=5, burst_window_sec=10)
    for _ in range(5):
        with gk.acquire():
            pass
    # 6th must wait until clock advances to t>=10 (oldest burst entry expires).
    t_before = clock.time()
    with gk.acquire():
        pass
    elapsed = clock.time() - t_before
    assert 9.5 <= elapsed <= 10.5, f"expected ~10s, got {elapsed}"


def test_max_concurrent():
    # Real threads, real sleep — small budget. max_concurrent=2.
    gk = ApiGatekeeper(
        per_minute=10_000,
        per_hour=10_000,
        max_concurrent=2,
        burst=10_000,
        burst_window_sec=1,
    )
    inside = 0
    max_inside = 0
    lock = threading.Lock()

    def worker():
        nonlocal inside, max_inside
        with gk.acquire():
            with lock:
                inside += 1
                if inside > max_inside:
                    max_inside = inside
            time.sleep(0.2)
            with lock:
                inside -= 1

    threads = [threading.Thread(target=worker) for _ in range(3)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    duration = time.monotonic() - t0
    assert max_inside <= 2, f"saw {max_inside} concurrent, expected ≤2"
    # 3 jobs, 2 slots, 0.2s each → wall time ≥ ~0.4s and well under 1s.
    assert 0.35 <= duration < 1.0, f"unexpected duration {duration}"


def test_timeout_raises():
    # No tokens available; timeout=0.05; clock advances only 0.01.
    gk, clock = _make(per_minute=1, per_hour=1, burst=1, burst_window_sec=1)
    # consume the single token
    with gk.acquire():
        pass

    # Patch sleep_fn so it advances clock by only 0.01s per call,
    # ensuring timeout check trips before refill provides another token.
    def tiny_sleep(_dt: float) -> None:
        clock.advance(0.01)

    gk._sleep_fn = tiny_sleep  # type: ignore[attr-defined]

    with pytest.raises(RateLimitExceeded):
        with gk.acquire(timeout=0.05):
            pass


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("aapl", "AAPL"),
        ("BRK.B", "BRK.B"),
        ("  nvda  ", "NVDA"),
        ("BF-B", "BF-B"),
    ],
)
def test_sanitize_ticker_accepts_typical(raw, expected):
    assert sanitize_ticker(raw) == expected


@pytest.mark.parametrize(
    "bad",
    [
        "../etc/passwd",
        "AAPL/",
        "..",
        "",
        "TOOLONGTICKER",
        "AAP L",
        ".AAPL",
        "AAPL.",
        "-AAPL",
        "AAPL\\",
        "AAPL;rm",
    ],
)
def test_sanitize_ticker_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        sanitize_ticker(bad)
