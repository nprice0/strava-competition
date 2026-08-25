import threading
import time
from datetime import datetime, timezone

import pytest

from strava_competition.errors import StravaRateLimitError
from strava_competition.strava_api import RateLimiter
from strava_competition.strava_client import rate_limiter as rl_mod
from strava_competition.strava_client import telemetry

# Generous timeout for positive assertions (worker SHOULD proceed).
_POS_TIMEOUT = 1.0
# Time to wait when asserting a worker is still blocked.  The
# ``blocking_worker`` helper below guarantees the thread has reached the
# blocking ``before_request`` call before we start this timer, so even a
# modest wait is reliable.
_NEG_TIMEOUT = 0.2


def worker(
    limiter: RateLimiter, started_evt: threading.Event, release_evt: threading.Event
) -> None:
    """Acquire a slot, signal start, wait until release, then free slot."""
    limiter.before_request()
    started_evt.set()
    release_evt.wait()
    limiter.after_response(None, 200)


def blocking_worker(
    limiter: RateLimiter,
    attempting_evt: threading.Event,
    started_evt: threading.Event,
    release_evt: threading.Event,
) -> None:
    """Like *worker* but signals *attempting_evt* before the blocking acquire.

    This lets the test synchronise on the thread having reached the
    semaphore, eliminating the race between thread creation and the
    negative assertion.
    """
    attempting_evt.set()
    limiter.before_request()
    started_evt.set()
    release_evt.wait()
    limiter.after_response(None, 200)


def _assert_blocked(
    attempting: threading.Event,
    started: threading.Event,
    msg: str,
) -> None:
    """Assert a *blocking_worker* is alive but stuck in ``before_request``."""
    assert attempting.wait(_POS_TIMEOUT), (
        f"Worker thread did not reach before_request: {msg}"
    )
    assert not started.wait(_NEG_TIMEOUT), msg


def test_rate_limiter_resize_behavior() -> None:
    """Validate dynamic resize semantics (grow then shrink)."""
    limiter = RateLimiter(max_concurrent=2)

    # Start two initial workers (should both start immediately)
    workers: list[tuple[threading.Event, threading.Event, threading.Thread]] = []
    for _ in range(2):
        started = threading.Event()
        release = threading.Event()
        t = threading.Thread(target=worker, args=(limiter, started, release))
        workers.append((started, release, t))
        t.start()

    for started, _, _ in workers:
        assert started.wait(_POS_TIMEOUT), "Initial worker failed to start in time"

    # Third worker should block (limit=2)
    c_attempting = threading.Event()
    c_started, c_release = threading.Event(), threading.Event()
    c_thread = threading.Thread(
        target=blocking_worker,
        args=(limiter, c_attempting, c_started, c_release),
    )
    c_thread.start()
    _assert_blocked(
        c_attempting,
        c_started,
        "Third worker should have been blocked before resize",
    )

    # Grow limit -> unblock waiting worker
    limiter.resize(3)
    assert c_started.wait(_POS_TIMEOUT), (
        "Blocked worker did not start after resize increase"
    )

    # Fourth worker should now block (A,B,C consume 3 slots)
    d_attempting = threading.Event()
    d_started, d_release = threading.Event(), threading.Event()
    d_thread = threading.Thread(
        target=blocking_worker,
        args=(limiter, d_attempting, d_started, d_release),
    )
    d_thread.start()
    _assert_blocked(
        d_attempting,
        d_started,
        "Fourth worker should be blocked until a slot frees",
    )

    # Release first worker -> fourth should start
    workers[0][1].set()  # release A
    workers[0][2].join(timeout=_POS_TIMEOUT)
    assert d_started.wait(_POS_TIMEOUT), (
        "Fourth worker failed to start after slot freed"
    )

    # Release remaining blocked workers
    workers[1][1].set()  # release B
    c_release.set()  # release C
    workers[1][2].join(timeout=_POS_TIMEOUT)
    c_thread.join(timeout=_POS_TIMEOUT)

    # Release D
    d_release.set()
    d_thread.join(timeout=_POS_TIMEOUT)

    # Shrink limit to 1 and validate blocking behavior
    limiter.resize(1)
    e_started, e_release = threading.Event(), threading.Event()
    e_thread = threading.Thread(target=worker, args=(limiter, e_started, e_release))
    e_thread.start()
    assert e_started.wait(_POS_TIMEOUT), "First worker after shrink did not start"

    f_attempting = threading.Event()
    f_started, f_release = threading.Event(), threading.Event()
    f_thread = threading.Thread(
        target=blocking_worker,
        args=(limiter, f_attempting, f_started, f_release),
    )
    f_thread.start()
    _assert_blocked(
        f_attempting,
        f_started,
        "Second worker should block with limit=1",
    )

    # Free e -> f should proceed
    e_release.set()
    e_thread.join(timeout=_POS_TIMEOUT)
    assert f_started.wait(_POS_TIMEOUT), (
        "Second worker did not start after first released under shrunken limit"
    )
    f_release.set()
    f_thread.join(timeout=_POS_TIMEOUT)

    # Basic sanity on final snapshot
    snap = limiter.snapshot()
    assert snap["in_flight"] == 0, "All workers should have completed"
    assert snap["max_allowed"] == 1, "Limiter should retain last resized value"


# ---------------------------------------------------------------------------
# Window reset boundary math
# ---------------------------------------------------------------------------


class TestSecondsUntilWindowReset:
    """Boundary math for quarter-hour UTC window resets (5s safety buffer)."""

    def test_mid_window(self) -> None:
        """Half-way into a window waits to the next boundary plus buffer."""
        now = datetime(2026, 8, 25, 12, 7, 30, tzinfo=timezone.utc)
        assert rl_mod.seconds_until_window_reset(now) == pytest.approx(450.0 + 5.0)

    def test_exactly_on_boundary(self) -> None:
        """Exactly on a boundary waits a full window plus buffer, never ~0."""
        now = datetime(2026, 8, 25, 12, 15, 0, tzinfo=timezone.utc)
        assert rl_mod.seconds_until_window_reset(now) == pytest.approx(900.0 + 5.0)

    def test_just_before_boundary(self) -> None:
        """One second before a boundary waits that second plus buffer."""
        now = datetime(2026, 8, 25, 12, 14, 59, tzinfo=timezone.utc)
        assert rl_mod.seconds_until_window_reset(now) == pytest.approx(1.0 + 5.0)


# ---------------------------------------------------------------------------
# Reset-wait behaviour
# ---------------------------------------------------------------------------


EXHAUSTED_HEADERS: dict[str, str] = {
    "X-RateLimit-Usage": "300,500",
    "X-RateLimit-Limit": "300,3000",
}

READ_EXHAUSTED_HEADERS: dict[str, str] = {
    "X-ReadRateLimit-Usage": "200,400",
    "X-ReadRateLimit-Limit": "200,2000",
}

DAILY_EXHAUSTED_HEADERS: dict[str, str] = {
    "X-RateLimit-Usage": "10,3000",
    "X-RateLimit-Limit": "300,3000",
}


def _quiet_limiter() -> RateLimiter:
    """Return a limiter without jitter for deterministic wait assertions."""
    return RateLimiter(max_concurrent=1, jitter_range=(0, 0))


def _patch_waits(monkeypatch: pytest.MonkeyPatch, limiter: RateLimiter) -> list[float]:
    """Replace the limiter's sleep-slice mechanism, recording requested waits."""
    waits: list[float] = []
    monkeypatch.setattr(limiter, "_interruptible_sleep", waits.append)
    return waits


class TestProactiveResetWait:
    """before_request waits for the window boundary when usage is exhausted."""

    def test_waits_then_clears_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exhausted short window triggers one boundary wait, then clears."""
        telemetry.reset()
        monkeypatch.setattr(rl_mod, "RATE_LIMIT_THROTTLE_SECONDS", 0)
        monkeypatch.setattr(
            rl_mod, "seconds_until_window_reset", lambda now=None: 123.0
        )
        limiter = _quiet_limiter()
        waits = _patch_waits(monkeypatch, limiter)

        limiter.after_response(EXHAUSTED_HEADERS, 200)
        limiter.before_request()
        assert waits == [123.0]
        assert limiter.snapshot()["short_used"] == 0

        limiter.after_response(None, 200)
        limiter.before_request()
        assert waits == [123.0], "usage cleared, second request must not wait"
        limiter.after_response(None, 200)
        assert telemetry.snapshot()[telemetry.RESET_WAITS] == 1

    def test_read_limit_exhaustion_also_waits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exhausted read window counts as exhausted too."""
        monkeypatch.setattr(rl_mod, "RATE_LIMIT_THROTTLE_SECONDS", 0)
        monkeypatch.setattr(rl_mod, "seconds_until_window_reset", lambda now=None: 60.0)
        limiter = _quiet_limiter()
        waits = _patch_waits(monkeypatch, limiter)

        limiter.after_response(READ_EXHAUSTED_HEADERS, 200)
        limiter.before_request()
        assert waits == [60.0]
        limiter.after_response(None, 200)


class TestReactiveResetThrottle:
    """A 429 sets the shared throttle deadline to the window boundary."""

    def test_429_sets_boundary_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """throttle_deadline reflects the boundary set by a 429."""
        monkeypatch.setattr(
            rl_mod, "seconds_until_window_reset", lambda now=None: 300.0
        )
        limiter = _quiet_limiter()
        limiter.before_request()
        throttled, _ = limiter.after_response({}, 429)
        assert throttled
        deadline = limiter.throttle_deadline()
        assert deadline is not None
        assert deadline == pytest.approx(time.time() + 300.0, abs=2.0)

    def test_other_requests_respect_boundary_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A later before_request waits out the boundary deadline."""
        monkeypatch.setattr(
            rl_mod, "seconds_until_window_reset", lambda now=None: 300.0
        )
        limiter = _quiet_limiter()
        waits = _patch_waits(monkeypatch, limiter)

        limiter.before_request()
        limiter.after_response({}, 429)
        limiter.before_request()
        assert len(waits) == 1
        assert waits[0] == pytest.approx(300.0, abs=2.0)
        limiter.after_response(None, 200)


class TestDailyExhaustion:
    """Daily limit exhaustion fails fast instead of waiting until midnight."""

    def test_raises_without_waiting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """before_request raises with used/limit in the message, no wait."""
        limiter = _quiet_limiter()
        waits = _patch_waits(monkeypatch, limiter)
        limiter.before_request()
        limiter.after_response(DAILY_EXHAUSTED_HEADERS, 200)

        with pytest.raises(StravaRateLimitError, match=r"3000/3000"):
            limiter.before_request()
        assert waits == []
        assert limiter.snapshot()["in_flight"] == 0, "no slot may leak on raise"

    def test_no_headers_never_raises(self) -> None:
        """A limiter that never saw headers must not fail fast."""
        limiter = _quiet_limiter()
        limiter.before_request()
        limiter.after_response(None, 200)


class TestFlagOff:
    """RATE_LIMIT_WAIT_FOR_RESET=false restores the fixed-throttle behaviour."""

    def test_429_uses_fixed_throttle_and_no_boundary_logic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flag off: fixed throttle seconds, no deadline, no raise, no wait."""
        monkeypatch.setattr(rl_mod, "RATE_LIMIT_WAIT_FOR_RESET", False)
        monkeypatch.setattr(rl_mod, "RATE_LIMIT_THROTTLE_SECONDS", 7)
        limiter = _quiet_limiter()
        reset_waits = _patch_waits(monkeypatch, limiter)
        sleeps: list[float] = []
        monkeypatch.setattr(rl_mod.time, "sleep", sleeps.append)

        limiter.before_request()
        throttled, _ = limiter.after_response(EXHAUSTED_HEADERS, 429)
        assert throttled
        assert limiter.throttle_deadline() is None
        throttle_until = limiter.snapshot()["throttle_until"]
        assert isinstance(throttle_until, float)
        assert throttle_until == pytest.approx(time.time() + 7.0, abs=2.0)

        # Exhausted short usage + daily headers seen: neither waits nor raises.
        limiter.before_request()
        assert reset_waits == []
        assert sleeps, "legacy fixed throttle sleep should have happened"
        assert sleeps[0] == pytest.approx(7.0, abs=2.0)
        limiter.after_response(None, 200)


class TestCancelEvent:
    """A set cancel event aborts a reset wait promptly."""

    def test_cancel_mid_wait_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The wait returns well before the requested duration once cancelled."""
        monkeypatch.setattr(rl_mod, "RATE_LIMIT_THROTTLE_SECONDS", 0)
        monkeypatch.setattr(rl_mod, "seconds_until_window_reset", lambda now=None: 30.0)
        limiter = _quiet_limiter()
        cancel = threading.Event()
        limiter.set_cancel_event(cancel)
        limiter.after_response(EXHAUSTED_HEADERS, 200)

        timer = threading.Timer(0.02, cancel.set)
        timer.start()
        start = time.monotonic()
        limiter.before_request()
        elapsed = time.monotonic() - start
        timer.cancel()
        assert elapsed < 1.0, f"wait did not abort promptly ({elapsed:.2f}s)"
        limiter.after_response(None, 200)
