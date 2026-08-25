"""Tests for SegmentService processing and default time handling."""

from __future__ import annotations

import threading
import time
from types import MethodType

import pytest

import warnings

from strava_competition.models import Segment, Runner, SegmentResult
from strava_competition.services.segment_service import SegmentService


@pytest.fixture
def runner() -> Runner:
    return Runner(name="Runner", strava_id="1", refresh_token="token", segment_team="A")


@pytest.fixture
def segment() -> Segment:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Segment(
            id=123,
            name="Test Segment",
            start_date=now - timedelta(days=1),
            end_date=now,
        )


def test_segment_processes_runners_in_parallel(
    runner: Runner, segment: Segment
) -> None:
    """Ensure runners are processed concurrently via activity scan."""

    service = SegmentService(max_workers=2)

    runners = []
    for idx in range(2):
        r = Runner(
            name=f"Runner {idx}",
            strava_id=str(idx + 1),
            refresh_token="token",
            segment_team="A",
        )
        runners.append(r)

    start_times: list[float] = []
    lock = threading.Lock()

    def fake_scan(
        self: SegmentService,
        runner_arg: Runner,
        segment_arg: Segment,
        cancel_event: threading.Event | None = None,
    ) -> SegmentResult | None:
        with lock:
            start_times.append(time.perf_counter())
            order = len(start_times)
        if order == 1:
            for _ in range(100):
                time.sleep(0.005)
                with lock:
                    if len(start_times) >= 2:
                        break
            else:
                raise AssertionError("Tasks did not start concurrently")
        time.sleep(0.02)
        return None

    service._result_from_activity_scan = MethodType(fake_scan, service)  # type: ignore[method-assign]

    service._process_segment(segment, runners, 1, 1, threading.Event(), None)

    assert len(start_times) == len(runners)
    assert max(start_times) - min(start_times) < 0.1


def test_segment_skips_when_cancelled(runner: Runner, segment: Segment) -> None:
    """Verify cancellation prevents processing."""

    service = SegmentService(max_workers=2)

    r = Runner(
        name="Cancelled",
        strava_id="42",
        refresh_token="token",
        segment_team="A",
    )

    cancel_event = threading.Event()
    cancel_event.set()

    invoked = threading.Event()

    def fake_scan(
        self: SegmentService,
        runner_arg: Runner,
        segment_arg: Segment,
        cancel_event_arg: threading.Event | None = None,
    ) -> SegmentResult | None:
        invoked.set()
        return None

    service._result_from_activity_scan = MethodType(fake_scan, service)  # type: ignore[method-assign]

    results = service._process_segment(segment, [r], 1, 1, cancel_event, None)

    assert results == {}
    assert not invoked.is_set()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_default_time_results_added_for_missing_runners() -> None:
    from datetime import datetime, timedelta, timezone

    service = SegmentService(max_workers=1)
    now = datetime.now(timezone.utc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        segment = Segment(
            id=42,
            name="Defaulted",
            start_date=now - timedelta(days=5),
            end_date=now,
            default_time_seconds=150.0,
        )
    runners = [
        Runner("Alice", "1", "rt1", segment_team="Alpha"),
        Runner("Bob", "2", "rt2", segment_team="Alpha"),
        Runner("Cara", "3", "rt3", segment_team="Beta"),
        Runner("Skip", "4", "rt4", segment_team=None),
    ]
    segment_results = {
        "Alpha": [
            SegmentResult(
                runner="Alice",
                team="Alpha",
                segment=segment.name,
                attempts=2,
                fastest_time=140.0,
                fastest_date=None,
            )
        ]
    }

    service._inject_default_segment_results(segment, runners, segment_results)

    alpha_results = {res.runner: res for res in segment_results["Alpha"]}
    assert "Bob" in alpha_results
    default_result = alpha_results["Bob"]
    assert default_result.fastest_time == pytest.approx(150.0)
    assert default_result.attempts == 0
    assert default_result.fastest_date == segment.start_date
    assert default_result.source == "default_time"
    assert default_result.diagnostics.get("reason") == "default_time_applied"
    assert default_result.fastest_distance_m == 0.0
    assert "Beta" in segment_results
    beta_default = segment_results["Beta"][0]
    assert beta_default.runner == "Cara"
    assert beta_default.fastest_time == pytest.approx(150.0)
    assert beta_default.fastest_date == segment.start_date
    assert beta_default.fastest_distance_m == 0.0


def _segment_with_default(default_time: float = 150.0) -> Segment:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Segment(
            id=77,
            name="Failure Segment",
            start_date=now - timedelta(days=5),
            end_date=now,
            default_time_seconds=default_time,
        )


def test_failed_scan_excluded_from_defaults_and_warned(
    caplog: "pytest.LogCaptureFixture",
) -> None:
    """A runner whose scan raises must not get default_time; a warning is logged."""
    import logging

    service = SegmentService(max_workers=1)
    segment = _segment_with_default()
    boom = Runner("Boom", "1", "rt1", segment_team="A")
    ok = Runner("OK", "2", "rt2", segment_team="A")

    def fake_scan(
        self: SegmentService,
        runner_arg: Runner,
        segment_arg: Segment,
        cancel_event: threading.Event | None = None,
    ) -> SegmentResult | None:
        if runner_arg.name == "Boom":
            raise ValueError("corrupt payload")
        return None  # OK: scanned, no effort → default_time

    service._result_from_activity_scan = MethodType(fake_scan, service)  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING, logger="SegmentService"):
        results = service._process_segment(segment, [boom, ok], 1, 1, None, None)

    all_results = {r.runner: r for team in results.values() for r in team}
    assert "Boom" not in all_results
    assert all_results["OK"].source == "default_time"
    assert "Activity scan failed runner=Boom" in caplog.text
    assert "Scan failed for 1 runners" in caplog.text


def test_cancellation_mid_batch_skips_default_injection() -> None:
    """Cancellation during a batch must not inject any default_time results."""
    service = SegmentService(max_workers=1)
    segment = _segment_with_default()
    first = Runner("First", "1", "rt1", segment_team="A")
    second = Runner("Second", "2", "rt2", segment_team="A")
    cancel_event = threading.Event()

    def fake_scan(
        self: SegmentService,
        runner_arg: Runner,
        segment_arg: Segment,
        cancel_event_arg: threading.Event | None = None,
    ) -> SegmentResult | None:
        cancel_event.set()  # cancel mid-batch while scans run
        return None

    service._result_from_activity_scan = MethodType(fake_scan, service)  # type: ignore[method-assign]

    results = service._process_segment(
        segment, [first, second], 1, 1, cancel_event, None
    )

    all_results = [r for team in results.values() for r in team]
    assert all_results == []


def test_fetch_failure_none_excluded_from_defaults(
    monkeypatch: pytest.MonkeyPatch,
    caplog: "pytest.LogCaptureFixture",
) -> None:
    """get_activities returning None (fetch failure) must not look like
    'no activities': the runner gets neither a result nor default_time."""
    import logging

    import strava_competition.services.segment_service as mod

    monkeypatch.setattr(mod, "get_activities", lambda *a, **k: None)
    service = SegmentService(max_workers=1)
    segment = _segment_with_default()
    runner = Runner("Unlucky", "1", "rt1", segment_team="A")

    with caplog.at_level(logging.WARNING, logger="SegmentService"):
        results = service._process_segment(segment, [runner], 1, 1, None, None)

    all_results = [r for team in results.values() for r in team]
    assert all_results == []
    assert "Activity scan failed runner=Unlucky" in caplog.text


def test_fetch_empty_list_still_gets_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuinely empty activity list remains a normal 'no effort' result."""
    import strava_competition.services.segment_service as mod

    monkeypatch.setattr(mod, "get_activities", lambda *a, **k: [])
    service = SegmentService(max_workers=1)
    segment = _segment_with_default()
    runner = Runner("Idle", "1", "rt1", segment_team="A")

    results = service._process_segment(segment, [runner], 1, 1, None, None)

    all_results = [r for team in results.values() for r in team]
    assert len(all_results) == 1
    assert all_results[0].source == "default_time"


def test_rate_limit_retry_cooldown_is_cancellation_aware() -> None:
    """A set cancel event must abort retry rounds without sleeping."""
    from strava_competition.errors import StravaRateLimitError

    service = SegmentService(max_workers=1)
    segment = _segment_with_default()
    runner = Runner("Limited", "1", "rt1", segment_team="A")
    cancel_event = threading.Event()

    def rate_limited_scan(
        self: SegmentService,
        runner_arg: Runner,
        segment_arg: Segment,
        cancel_event_arg: threading.Event | None = None,
    ) -> SegmentResult | None:
        cancel_event.set()  # cancel while the first round is in flight
        raise StravaRateLimitError("429")

    service._result_from_activity_scan = MethodType(rate_limited_scan, service)  # type: ignore[method-assign]

    # Default cooldown is 60s per round: with cancellation-aware waiting the
    # call must return promptly instead of sleeping through retry rounds.
    start = time.perf_counter()
    results = service._process_segment(segment, [runner], 1, 1, cancel_event, None)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0
    all_results = [r for team in results.values() for r in team]
    assert all_results == []
