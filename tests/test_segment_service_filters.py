"""Tests for SegmentService fallback processing and default time handling."""

from __future__ import annotations

import threading
import time
from types import MethodType
from typing import List

import pytest

from strava_competition.models import Segment, Runner, SegmentResult
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


def test_default_time_results_added_for_missing_runners() -> None:
    from datetime import datetime, timedelta, timezone

    service = SegmentService(max_workers=1)
    now = datetime.now(timezone.utc)
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
    assert "Beta" in segment_results
    beta_default = segment_results["Beta"][0]
    assert beta_default.runner == "Cara"
    assert beta_default.fastest_time == pytest.approx(150.0)
    assert beta_default.fastest_date == segment.start_date
