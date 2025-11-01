"""Tests for SegmentService fallback activity filtering."""

from __future__ import annotations

from typing import List

import pytest

from strava_competition.models import Segment, Runner
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
