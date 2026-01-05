"""Unit tests for the activity-scan fallback module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pytest

from strava_competition.activity_scan.scanner import ActivityEffortScanner
from strava_competition.activity_scan.models import ActivityScanResult
from strava_competition.errors import StravaAPIError
from strava_competition.models import Runner, Segment
from strava_competition.services.segment_service import SegmentService

TEST_CAPTURE_DIR = Path(__file__).resolve().parent / "strava_api_capture"


@pytest.fixture
def runner() -> Runner:
    return Runner(
        name="Runner", strava_id="123", refresh_token="token", segment_team="A"
    )


@pytest.fixture
def segment() -> Segment:
    now = datetime.now(timezone.utc)
    return Segment(
        id=55, name="Segment", start_date=now - timedelta(days=7), end_date=now
    )


@pytest.fixture
def capture_replay_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force API capture to use the vendored fixtures under tests/strava_api_capture."""

    from strava_competition import api_capture, config, strava_api

    monkeypatch.setattr(
        config, "STRAVA_API_CAPTURE_DIR", str(TEST_CAPTURE_DIR), raising=False
    )
    monkeypatch.setattr(config, "STRAVA_API_REPLAY_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "STRAVA_API_CAPTURE_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "STRAVA_CAPTURE_HASH_IDENTIFIERS", False, raising=False)
    monkeypatch.setattr(config, "STRAVA_OFFLINE_MODE", True, raising=False)

    monkeypatch.setenv("STRAVA_API_CAPTURE_DIR", str(TEST_CAPTURE_DIR))
    monkeypatch.setenv("STRAVA_API_REPLAY_ENABLED", "true")
    monkeypatch.setenv("STRAVA_API_CAPTURE_ENABLED", "false")
    monkeypatch.setenv("STRAVA_CAPTURE_HASH_IDENTIFIERS", "false")
    monkeypatch.setenv("STRAVA_OFFLINE_MODE", "true")

    monkeypatch.setattr(
        api_capture, "STRAVA_API_CAPTURE_DIR", str(TEST_CAPTURE_DIR), raising=False
    )
    monkeypatch.setattr(api_capture, "STRAVA_API_REPLAY_ENABLED", True, raising=False)
    monkeypatch.setattr(api_capture, "STRAVA_API_CAPTURE_ENABLED", False, raising=False)
    monkeypatch.setattr(api_capture, "_CAPTURE", api_capture.APICapture())
    monkeypatch.setattr(strava_api, "STRAVA_OFFLINE_MODE", True, raising=False)
    monkeypatch.setattr(
        strava_api, "STRAVA_CAPTURE_HASH_IDENTIFIERS", False, raising=False
    )

    import strava_competition.strava_client.activities as activities_client
    import strava_competition.strava_client.capture as capture_client

    monkeypatch.setattr(activities_client, "STRAVA_OFFLINE_MODE", True, raising=False)
    monkeypatch.setattr(
        activities_client, "STRAVA_CAPTURE_HASH_IDENTIFIERS", False, raising=False
    )
    monkeypatch.setattr(capture_client, "STRAVA_OFFLINE_MODE", True, raising=False)
    monkeypatch.setattr(
        capture_client, "STRAVA_CAPTURE_HASH_IDENTIFIERS", False, raising=False
    )


def test_activity_scanner_finds_fastest_effort(
    monkeypatch: pytest.MonkeyPatch, runner: Runner, segment: Segment
) -> None:
    activities = [{"id": 9001, "name": "Morning Run"}]

    detail_payload = {
        "segment_efforts": [
            {
                "id": "slow",
                "segment": {"id": segment.id},
                "elapsed_time": 400,
                "moving_time": 390,
                "start_date_local": "2024-01-01T08:10:00Z",
            },
            {
                "id": "fast",
                "segment": {"id": segment.id},
                "elapsed_time": 305,
                "moving_time": 300,
                "start_date_local": "2024-01-01T09:15:00Z",
            },
        ]
    }

    fetch_calls: List[int] = []

    scanner = ActivityEffortScanner(activity_provider=lambda *_args: activities)

    monkeypatch.setattr(
        "strava_competition.activity_scan.scanner.get_activity_with_efforts",
        lambda _runner, activity_id, include_all_efforts=True: fetch_calls.append(
            activity_id
        )
        or detail_payload,
    )

    result = scanner.scan_segment(runner, segment)

    assert result is not None
    assert result.attempts == 2
    assert result.fastest_elapsed == 305
    assert result.fastest_effort_id == "fast"
    assert result.fastest_activity_id == 9001
    assert len(result.effort_ids) == 2
    assert result.inspected_activities == [{"id": 9001, "name": "Morning Run"}]
    assert fetch_calls == [9001]

    second = scanner.scan_segment(runner, segment)
    assert second is not None
    assert fetch_calls == [9001]


def test_activity_scanner_returns_none_without_matching_efforts(
    monkeypatch: pytest.MonkeyPatch, runner: Runner, segment: Segment
) -> None:
    activities = [{"id": 77, "name": "Morning Ride"}]

    scanner = ActivityEffortScanner(activity_provider=lambda *_: activities)

    monkeypatch.setattr(
        "strava_competition.activity_scan.scanner.get_activity_with_efforts",
        lambda *_args, **_kwargs: {"segment_efforts": [{"segment": {"id": 999}}]},
    )

    assert scanner.scan_segment(runner, segment) is None


def test_activity_scanner_ignores_bad_elapsed_times(
    monkeypatch: pytest.MonkeyPatch, runner: Runner, segment: Segment
) -> None:
    activities = [{"id": 123, "name": "Tempo"}]

    detail_payload = {
        "segment_efforts": [
            {"id": "ignored-none", "segment": {"id": segment.id}, "elapsed_time": None},
            {
                "id": "ignored-negative",
                "segment": {"id": segment.id},
                "elapsed_time": -5,
            },
            {
                "id": "kept",
                "segment": {"id": segment.id},
                "elapsed_time": 320,
                "moving_time": 315,
                "start_date_local": "2024-01-01T09:00:00Z",
            },
        ]
    }

    scanner = ActivityEffortScanner(activity_provider=lambda *_: activities)
    monkeypatch.setattr(
        "strava_competition.activity_scan.scanner.get_activity_with_efforts",
        lambda *_args, **_kwargs: detail_payload,
    )

    result = scanner.scan_segment(runner, segment)
    assert result is not None
    assert result.attempts == 1
    assert result.fastest_effort_id == "kept"


def test_activity_scanner_enforces_min_distance(
    monkeypatch: pytest.MonkeyPatch, runner: Runner, segment: Segment
) -> None:
    activities = [{"id": 789, "name": "Long Run"}]
    segment.min_distance_meters = 400.0

    detail_payload = {
        "segment_efforts": [
            {
                "id": "short",
                "segment": {"id": segment.id},
                "elapsed_time": 310,
            },
            {
                "id": "long",
                "segment": {"id": segment.id},
                "elapsed_time": 320,
                "start_date_local": "2024-01-01T09:00:00Z",
            },
        ]
    }

    scanner = ActivityEffortScanner(activity_provider=lambda *_: activities)
    distance_map = {"short": 350.0, "long": 405.0}

    def _fake_distance(_runner, effort, *, allow_stream=True):
        return distance_map[effort["id"]]

    monkeypatch.setattr(
        "strava_competition.activity_scan.scanner.derive_effort_distance_m",
        _fake_distance,
    )
    monkeypatch.setattr(
        "strava_competition.activity_scan.scanner.get_activity_with_efforts",
        lambda *_args, **_kwargs: detail_payload,
    )

    result = scanner.scan_segment(runner, segment)

    assert result is not None
    assert result.attempts == 1
    assert result.fastest_effort_id == "long"
    assert result.filtered_efforts_below_distance == 1
    assert result.fastest_distance_m == pytest.approx(405.0)


def test_activity_scanner_distance_fallbacks_to_payload(
    monkeypatch: pytest.MonkeyPatch, runner: Runner, segment: Segment
) -> None:
    activities = [{"id": 12, "name": "Morning"}]
    segment.min_distance_meters = 300.0

    detail_payload = {
        "segment_efforts": [
            {
                "id": "with-distance",
                "segment": {"id": segment.id},
                "elapsed_time": 200,
                "distance": 305.0,
                "start_date_local": "2024-01-05T08:00:00Z",
            }
        ]
    }

    scanner = ActivityEffortScanner(activity_provider=lambda *_: activities)
    monkeypatch.setattr(
        "strava_competition.effort_distance.compute_effort_distance_from_payload",
        lambda *_runner, **_kwargs: None,
    )
    monkeypatch.setattr(
        "strava_competition.activity_scan.scanner.get_activity_with_efforts",
        lambda *_args, **_kwargs: detail_payload,
    )

    result = scanner.scan_segment(runner, segment)

    assert result is not None
    assert result.attempts == 1
    assert result.fastest_effort_id == "with-distance"
    assert result.fastest_distance_m == pytest.approx(305.0)


def test_activity_scanner_propagates_strava_api_error(
    monkeypatch: pytest.MonkeyPatch, runner: Runner, segment: Segment
) -> None:
    activities = [{"id": 456, "name": "Evening"}]
    scanner = ActivityEffortScanner(activity_provider=lambda *_: activities)

    def _raise(*_args, **_kwargs):
        raise StravaAPIError("offline miss")

    monkeypatch.setattr(
        "strava_competition.activity_scan.scanner.get_activity_with_efforts",
        _raise,
    )

    with pytest.raises(StravaAPIError):
        scanner.scan_segment(runner, segment)


def test_segment_service_uses_activity_scan(
    monkeypatch: pytest.MonkeyPatch, runner: Runner, segment: Segment
) -> None:
    from types import MethodType

    service = SegmentService(max_workers=1)

    monkeypatch.setattr(
        "strava_competition.services.segment_service.USE_ACTIVITY_SCAN_FALLBACK",
        True,
    )

    scan_result = ActivityScanResult(
        segment_id=segment.id,
        attempts=1,
        fastest_elapsed=250.0,
        fastest_effort_id="effort",
        fastest_activity_id=999,
        fastest_start_date=segment.start_date,
        moving_time=245.0,
        effort_ids=["effort"],
        inspected_activities=[{"id": 999, "name": "Activity"}],
    )

    service._activity_scanner.scan_segment = MethodType(  # type: ignore[attr-defined]
        lambda _self, _runner, _segment, cancel_event=None: scan_result,
        service._activity_scanner,
    )

    runner.payment_required = True  # type: ignore[attr-defined]

    result = service._process_runner_results(runner, segment, efforts=None)

    assert result is not None
    assert result.source == "activity_scan"
    assert result.fastest_time == scan_result.fastest_elapsed
    assert result.fastest_date == scan_result.fastest_start_date


def test_segment_service_handles_activity_scan_error(
    monkeypatch: pytest.MonkeyPatch, runner: Runner, segment: Segment
) -> None:
    from types import MethodType

    service = SegmentService(max_workers=1)

    monkeypatch.setattr(
        "strava_competition.services.segment_service.USE_ACTIVITY_SCAN_FALLBACK",
        True,
    )

    def _raise_scan(self, *_args, **_kwargs):
        raise StravaAPIError("offline miss")

    service._activity_scanner.scan_segment = MethodType(
        _raise_scan, service._activity_scanner
    )

    runner.payment_required = True  # type: ignore[attr-defined]

    # With scan error and no matcher fallback, result should be None
    result = service._process_runner_results(runner, segment, efforts=None)

    assert result is None


def test_activity_scan_replay_uses_capture(
    capture_replay_env: None,
    runner: Runner,
    segment: Segment,
) -> None:
    scanner = ActivityEffortScanner()

    runner.strava_id = "35599907"  # type: ignore[attr-defined]
    runner.name = "Neil"
    segment.id = 38_987_500
    segment.name = "SS25-02-Fiveways to Hell"
    segment.start_date = datetime.fromisoformat("2025-05-08T00:00:00+00:00")
    segment.end_date = datetime.fromisoformat("2025-06-02T00:00:00+00:00")

    result = scanner.scan_segment(runner, segment)

    assert result is not None
    assert result.attempts == 2
    assert result.fastest_elapsed == 1893
    assert result.fastest_activity_id == 14661937369
    assert result.fastest_effort_id == "scan-effort-fast"
    assert result.effort_ids == ["scan-effort-fast", "scan-effort-slow"]
