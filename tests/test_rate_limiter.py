import threading

from strava_competition.strava_api import RateLimiter

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
