"""Tests for SegmentService fallback activity filtering."""

from __future__ import annotations

import threading
import time
from types import MethodType
from typing import List

import pytest

from strava_competition.models import Segment, Runner, SegmentResult
from strava_competition.matching.models import MatchResult
from strava_competition.services.segment_service import SegmentService


@pytest.fixture
def runner() -> Runner:
    return Runner(name="Runner", strava_id="1", refresh_token="token", segment_team="A")


@pytest.fixture
def segment() -> Segment:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    return Segment(
        id=123, name="Test Segment", start_date=now - timedelta(days=1), end_date=now
    )


def _make_activity(activity_id: int, activity_type: str, distance_m: float) -> dict:
    return {
        "id": activity_id,
        "type": activity_type,
        "distance": distance_m,
        "start_date_local": "2024-01-01T00:00:00Z",
    }


def test_matcher_skips_non_run_and_short_distance(
    monkeypatch: pytest.MonkeyPatch, runner: Runner, segment: Segment
) -> None:
    activities: List[dict] = [
        _make_activity(1, "Ride", 2000.0),
        _make_activity(2, "Run", 300.0),
        _make_activity(3, "Run", 1200.0),
    ]

    monkeypatch.setattr(
        "strava_competition.services.segment_service.get_activities",
        lambda *_: activities,
    )

    call_ids: List[int] = []

    def fake_match(
        runner_arg: Runner, activity_id: int, segment_id: int
    ) -> MatchResult:
        call_ids.append(activity_id)
        return MatchResult(matched=True, elapsed_time_s=500.0)

    monkeypatch.setattr(
        "strava_competition.services.segment_service.match_activity_to_segment",
        fake_match,
    )

    from strava_competition.matching.models import SegmentGeometry

    monkeypatch.setattr(
        "strava_competition.services.segment_service.fetch_segment_geometry",
        lambda *_: SegmentGeometry(segment_id=segment.id, points=[], distance_m=1500.0),
    )

    service = SegmentService(max_workers=1)

    result = service._match_runner_segment(runner, segment)

    assert result is not None
    assert call_ids == [3]


def test_fallback_queue_processes_in_parallel(runner: Runner, segment: Segment) -> None:
    """Ensure fallback runners are processed concurrently when queued."""

    service = SegmentService(max_workers=2)

    fallback_runners = []
    for idx in range(2):
        queued_runner = Runner(
            name=f"Runner {idx}",
            strava_id=str(idx + 1),
            refresh_token="token",
            segment_team="A",
        )
        queued_runner.payment_required = True  # type: ignore[attr-defined]
        fallback_runners.append(queued_runner)

    segment_results: dict[str, List[SegmentResult]] = {}
    progress_steps: List[int] = []
    start_times: List[float] = []
    lock = threading.Lock()

    def fake_process_runner(
        self: SegmentService,
        runner_arg: Runner,
        segment_arg: Segment,
        efforts: List[dict] | None,
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
                raise AssertionError("Fallback tasks did not start concurrently")
        time.sleep(0.02)
        return None

    service._process_runner_results = MethodType(fake_process_runner, service)

    completed = service._process_fallback_queue(
        segment,
        segment_results,
        fallback_runners,
        threading.Event(),
        0,
        progress_steps.append,
    )

    assert len(start_times) == len(fallback_runners)
    assert completed == len(fallback_runners)
    assert progress_steps == [1, 2]
    assert max(start_times) - min(start_times) < 0.1


def test_fallback_queue_skips_when_cancelled(runner: Runner, segment: Segment) -> None:
    """Verify cancellation prevents fallback submission or execution."""

    service = SegmentService(max_workers=2)

    queued_runner = Runner(
        name="Cancelled",
        strava_id="42",
        refresh_token="token",
        segment_team="A",
    )
    queued_runner.payment_required = True  # type: ignore[attr-defined]

    cancel_event = threading.Event()
    cancel_event.set()

    segment_results: dict[str, List[SegmentResult]] = {}
    progress_steps: List[int] = []
    invoked = threading.Event()

    def fake_process_runner(
        self: SegmentService,
        runner_arg: Runner,
        segment_arg: Segment,
        efforts: List[dict] | None,
        cancel_event_arg: threading.Event | None = None,
    ) -> SegmentResult | None:
        invoked.set()
        return None

    service._process_runner_results = MethodType(fake_process_runner, service)

    completed = service._process_fallback_queue(
        segment,
        segment_results,
        [queued_runner],
        cancel_event,
        0,
        progress_steps.append,
    )

    assert completed == 0
    assert progress_steps == []
    assert not invoked.is_set()


def test_match_runner_segment_respects_cancel_event(
    monkeypatch: pytest.MonkeyPatch, runner: Runner, segment: Segment
) -> None:
    """Ensure cancellation before matching avoids network fetches."""

    event = threading.Event()
    event.set()

    def fail_if_called(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("get_activities should not be invoked when cancelled")

    monkeypatch.setattr(
        "strava_competition.services.segment_service.get_activities",
        fail_if_called,
    )

    service = SegmentService(max_workers=1)

    result = service._match_runner_segment(runner, segment, event)

    assert result is None


def test_runner_activity_cache_reused_across_segments(
    monkeypatch: pytest.MonkeyPatch, runner: Runner, segment: Segment
) -> None:
    """Runner activities should be fetched once and reused across segments."""

    service = SegmentService(max_workers=1)

    activity = _make_activity(99, "Run", 2000.0)
    fetch_calls: List[int] = []

    def fake_get_activities(
        runner_arg: Runner, start_date_arg, end_date_arg
    ) -> List[dict]:
        fetch_calls.append(1)
        return [activity]

    monkeypatch.setattr(
        "strava_competition.services.segment_service.get_activities",
        fake_get_activities,
    )

    def fake_find_best_match(
        self: SegmentService,
        runner_arg: Runner,
        segment_arg: Segment,
        activities: List[dict],
        segment_distance_m,
        cancel_event,
    ) -> tuple[MatchResult | None, dict | None, int]:
        return MatchResult(matched=True, elapsed_time_s=123.0), activities[0], 1

    service._find_best_match = MethodType(fake_find_best_match, service)
    service._get_segment_distance = MethodType(
        lambda _self, _runner, _segment_id: None, service
    )

    segment_other = Segment(
        id=segment.id + 1,
        name="Second",
        start_date=segment.start_date,
        end_date=segment.end_date,
    )

    first = service._match_runner_segment(runner, segment)
    second = service._match_runner_segment(runner, segment_other)

    assert first is not None and second is not None
    assert len(fetch_calls) == 1
