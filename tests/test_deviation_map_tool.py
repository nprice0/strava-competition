"""Tests for the deviation map CLI helper."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import folium
import pytest
import warnings

from strava_competition.models import Runner, Segment
from strava_competition.tools.geometry.models import ActivityTrack, SegmentGeometry
from strava_competition.tools.deviation_map import build_deviation_map_for_effort


def _build_stub_segment() -> SegmentGeometry:
    points = [
        (51.4800, -3.1800),
        (51.4805, -3.1800),
        (51.4810, -3.1800),
    ]
    return SegmentGeometry(segment_id=123, points=points, distance_m=400.0)


def _build_stub_activity() -> ActivityTrack:
    points = [
        (51.4800, -3.1800),
        (51.4803, -3.1800),
        (51.4806, -3.1815),
        (51.4809, -3.1815),
        (51.4810, -3.1800),
    ]
    timestamps = [float(i) for i in range(len(points))]
    return ActivityTrack(activity_id=456, points=points, timestamps_s=timestamps)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_build_deviation_map_for_effort(tmp_path: Path) -> None:
    """Ensure the helper produces an interactive map and saves output."""

    runner = Runner(name="Test Runner", strava_id="1", refresh_token="token")
    now = datetime.now(timezone.utc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        segment = Segment(id=123, name="Stub Segment", start_date=now, end_date=now)
    segment_geometry = _build_stub_segment()
    activity_track = _build_stub_activity()

    def fake_segment_fetcher(_: Runner, segment_id: int) -> SegmentGeometry:
        assert segment_id == segment_geometry.segment_id
        return segment_geometry

    def fake_activity_fetcher(_: Runner, activity_id: int) -> ActivityTrack:
        assert activity_id == activity_track.activity_id
        return activity_track

    output_path = tmp_path / "deviation.html"
    map_object, coverage, diagnostics = build_deviation_map_for_effort(
        runner,
        activity_id=activity_track.activity_id,
        segment_id=segment.id,
        threshold_m=30.0,
        output_html=output_path,
        segment_fetcher=fake_segment_fetcher,
        activity_fetcher=fake_activity_fetcher,
    )

    assert isinstance(map_object, folium.Map)
    assert output_path.exists(), "Expected HTML output to be written"
    assert coverage.coverage_ratio > 0.0
    gate_max = diagnostics.get("gate_trimmed_max_offset_m")
    assert gate_max is None or (isinstance(gate_max, (int, float)) and gate_max >= 0.0)
