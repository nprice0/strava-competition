import threading
import time

from strava_competition.strava_api import RateLimiter


def worker(limiter: RateLimiter, started_evt: threading.Event, release_evt: threading.Event):
    """Acquire a slot, signal start, wait until release, then free slot."""
    limiter.before_request()
    started_evt.set()
    release_evt.wait()
    limiter.after_response(None, 200)


def test_rate_limiter_resize_behavior():
    """Validate dynamic resize semantics (grow then shrink)."""
    limiter = RateLimiter(max_concurrent=2)

    # Start two initial workers (should both start immediately)
    workers = []
    for _ in range(2):
        started = threading.Event()
        release = threading.Event()
        t = threading.Thread(target=worker, args=(limiter, started, release))
        workers.append((started, release, t))
        t.start()

    for started, _, _ in workers:
        assert started.wait(0.3), "Initial worker failed to start in time"

    # Third worker should block (limit=2)
    c_started, c_release = threading.Event(), threading.Event()
    c_thread = threading.Thread(target=worker, args=(limiter, c_started, c_release))
    c_thread.start()
    assert not c_started.wait(0.07), "Third worker should have been blocked before resize"

    # Grow limit -> unblock waiting worker
    limiter.resize(3)
    assert c_started.wait(0.3), "Blocked worker did not start after resize increase"

    # Fourth worker should now block (A,B,C consume 3 slots)
    d_started, d_release = threading.Event(), threading.Event()
    d_thread = threading.Thread(target=worker, args=(limiter, d_started, d_release))
    d_thread.start()
    assert not d_started.wait(0.07), "Fourth worker should be blocked until a slot frees"

    # Release first worker -> fourth should start
    workers[0][1].set()  # release A
    workers[0][2].join(timeout=0.6)
    assert d_started.wait(0.3), "Fourth worker failed to start after slot freed"

    # Release remaining blocked workers
    workers[1][1].set()  # release B
    c_release.set()      # release C
    workers[1][2].join(timeout=0.6)
    c_thread.join(timeout=0.6)

    # Release D
    d_release.set()
    d_thread.join(timeout=0.6)

    # Shrink limit to 1 and validate blocking behavior
    limiter.resize(1)
    e_started, e_release = threading.Event(), threading.Event()
    e_thread = threading.Thread(target=worker, args=(limiter, e_started, e_release))
    e_thread.start()
    assert e_started.wait(0.3), "First worker after shrink did not start"

    f_started, f_release = threading.Event(), threading.Event()
    f_thread = threading.Thread(target=worker, args=(limiter, f_started, f_release))
    f_thread.start()
    assert not f_started.wait(0.07), "Second worker should block with limit=1"

    # Free e -> f should proceed
    e_release.set()
    e_thread.join(timeout=0.6)
    assert f_started.wait(0.3), "Second worker did not start after first released under shrunken limit"
    f_release.set()
    f_thread.join(timeout=0.6)

    # Basic sanity on final snapshot
    snap = limiter.snapshot()
    assert snap["in_flight"] == 0, "All workers should have completed"
    assert snap["max_allowed"] == 1, "Limiter should retain last resized value"
