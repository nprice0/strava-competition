"""Unit tests covering the segment matching components."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest

from strava_competition.config import (
    MATCHING_MAX_RESAMPLED_POINTS,
    MATCHING_MAX_SIMPLIFIED_POINTS,
)
from strava_competition.matching import (
    MatchResult,
    Tolerances,
    match_activity_to_segment,
)
from strava_competition.matching.models import ActivityTrack, SegmentGeometry
from strava_competition.matching.preprocessing import (
    PreparedActivityTrack,
    PreparedSegmentGeometry,
    prepare_activity,
    prepare_geometry,
)
from strava_competition.matching.similarity import (
    SEGMENT_CACHE,
    build_cached_segment,
    discrete_frechet_distance,
    segment_cache_scope,
    windowed_dtw,
)
from strava_competition.matching.timing import estimate_segment_time
from strava_competition.matching.validation import check_direction, compute_coverage
from strava_competition.models import Runner


@pytest.fixture(autouse=True)
def clear_segment_cache() -> Iterator[None]:
    """Ensure the segment cache is cleared before and after each test."""

    SEGMENT_CACHE.clear()
    yield
    SEGMENT_CACHE.clear()


@pytest.fixture
def segment_geometry() -> SegmentGeometry:
    """Return a simple three-point segment for testing."""

    latlon = [
        (37.0000, -122.0000),
        (37.0003, -122.0000),
        (37.0006, -122.0000),
    ]
    return SegmentGeometry(segment_id=101, points=latlon, distance_m=110.0)


@pytest.fixture
def activity_track(segment_geometry: SegmentGeometry) -> ActivityTrack:
    """Return an activity track that cleanly follows the segment."""

    return ActivityTrack(
        activity_id=555,
        points=list(segment_geometry.points),
        timestamps_s=[0.0, 12.0, 24.0],
    )


def _prepare_pair(
    segment: SegmentGeometry,
    activity: ActivityTrack,
    *,
    simplification_tolerance_m: float = 2.0,
    resample_interval_m: float = 5.0,
) -> tuple[PreparedSegmentGeometry, PreparedActivityTrack]:
    """Prepare segment and activity geometry for downstream tests."""

    prepared_segment = prepare_geometry(
        segment,
        simplification_tolerance_m=simplification_tolerance_m,
        resample_interval_m=resample_interval_m,
    )
    prepared_activity = prepare_activity(
        activity,
        transformer=prepared_segment.transformer,
        simplification_tolerance_m=simplification_tolerance_m,
        resample_interval_m=resample_interval_m,
    )
    return prepared_segment, prepared_activity


def _build_basic_activity(segment: SegmentGeometry) -> ActivityTrack:
    """Build an activity track that mirrors the segment geometry."""

    return ActivityTrack(
        activity_id=0,
        points=list(segment.points),
        timestamps_s=[0.0, 10.0, 20.0],
    )


def test_prepare_geometry_resamples_points(segment_geometry: SegmentGeometry) -> None:
    """prepare_geometry should resample points along the segment."""

    activity = _build_basic_activity(segment_geometry)
    prepared_segment, _ = _prepare_pair(segment_geometry, activity)
    assert prepared_segment.resampled_points.shape[0] >= 3
    assert np.isclose(
        prepared_segment.resampled_points[0, 0],
        prepared_segment.metric_points[0, 0],
    )


def test_prepare_activity_uses_segment_transformer(
    segment_geometry: SegmentGeometry,
    activity_track: ActivityTrack,
) -> None:
    """Activity prep reuses the segment transformer for alignment."""

    prepared_segment, prepared_activity = _prepare_pair(
        segment_geometry,
        activity_track,
    )
    assert prepared_activity.transformer is prepared_segment.transformer
    assert (
        prepared_activity.resampled_points.shape[0]
        == prepared_segment.resampled_points.shape[0]
    )


def test_direction_validation_detects_reverse_motion() -> None:
    """Direction check rejects tracks heading away from the start."""

    segment_points = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    forward = np.array([[0.5, 0.0], [20.0, 0.0], [80.0, 0.0]])
    reverse = forward[::-1]

    forward_result = check_direction(forward, segment_points, start_tolerance_m=5.0)
    reverse_result = check_direction(reverse, segment_points, start_tolerance_m=5.0)

    assert forward_result.matches_direction is True
    assert reverse_result.matches_direction is False


def test_compute_coverage_handles_partial_activity() -> None:
    """Coverage ratio should reflect the portion of the segment visited."""

    segment_points = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    partial_activity = np.array([[40.0, 0.0], [60.0, 0.0]])

    coverage = compute_coverage(partial_activity, segment_points)

    assert coverage.coverage_bounds is not None
    assert coverage.coverage_ratio == pytest.approx(0.2, rel=1e-3)


def test_similarity_metrics_distinguish_offset() -> None:
    """Discrete Fréchet distance should increase when paths diverge."""

    segment = [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]
    identical_activity = [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]]
    offset_activity = [[0.0, 5.0], [50.0, 5.0], [100.0, 5.0]]

    baseline = discrete_frechet_distance(identical_activity, segment)
    offset = discrete_frechet_distance(offset_activity, segment)
    dtw = windowed_dtw(identical_activity, segment, window_size=1)

    assert baseline == pytest.approx(0.0, abs=1e-6)
    assert offset > 0.0
    assert dtw == pytest.approx(0.0, abs=1e-6)


def test_timing_estimation_interpolates_entry_and_exit(
    segment_geometry: SegmentGeometry,
    activity_track: ActivityTrack,
) -> None:
    """estimate_segment_time returns elapsed seconds inside coverage."""

    prepared_segment, prepared_activity = _prepare_pair(
        segment_geometry,
        activity_track,
    )
    coverage = compute_coverage(
        prepared_activity.metric_points,
        prepared_segment.metric_points,
    )
    assert coverage.coverage_bounds is not None

    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        coverage.coverage_bounds,
    )

    assert estimate.elapsed_time_s == pytest.approx(24.0, rel=1e-2)
    assert estimate.entry_time_s == pytest.approx(0.0, abs=1e-3)
    assert estimate.exit_time_s == pytest.approx(24.0, abs=1e-3)


def test_timing_estimation_ignores_post_segment_samples(
    segment_geometry: SegmentGeometry,
) -> None:
    """Elapsed time should stop at the segment finish even if the run continues."""

    extended_points = list(segment_geometry.points) + [
        (segment_geometry.points[-1][0] + 0.0003, segment_geometry.points[-1][1]),
    ]
    activity = ActivityTrack(
        activity_id=777,
        points=extended_points,
        timestamps_s=[0.0, 15.0, 30.0, 90.0],
    )

    prepared_segment, prepared_activity = _prepare_pair(
        segment_geometry,
        activity,
    )
    coverage = compute_coverage(
        prepared_activity.metric_points,
        prepared_segment.metric_points,
    )
    assert coverage.coverage_bounds is not None

    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        coverage.coverage_bounds,
    )

    assert estimate.exit_time_s == pytest.approx(30.0, abs=1e-3)
    assert estimate.elapsed_time_s == pytest.approx(30.0, rel=1e-2)


def test_match_activity_to_segment_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """match_activity_to_segment succeeds when the geometry aligns."""

    runner = Runner(name="Test Runner", strava_id="123", refresh_token="rt")

    segment = SegmentGeometry(
        segment_id=404,
        points=[(37.0, -122.0), (37.0003, -122.0), (37.0006, -122.0)],
        distance_m=120.0,
    )
    activity = ActivityTrack(
        activity_id=808,
        points=list(segment.points),
        timestamps_s=[0.0, 15.0, 30.0],
    )

    monkeypatch.setattr(
        "strava_competition.matching.fetch_segment_geometry",
        lambda _runner, _segment_id: segment,
    )
    monkeypatch.setattr(
        "strava_competition.matching.fetch_activity_stream",
        lambda _runner, _activity_id: activity,
    )

    tolerances = Tolerances(
        start_tolerance_m=50.0,
        frechet_tolerance_m=25.0,
        coverage_threshold=0.9,
        simplification_tolerance_m=2.0,
        resample_interval_m=5.0,
    )

    result: MatchResult = match_activity_to_segment(
        runner,
        activity.activity_id,
        segment.segment_id,
        tolerances,
    )

    assert result.matched is True
    assert result.elapsed_time_s == pytest.approx(30.0, rel=1e-3)
    assert result.diagnostics.get("similarity_method") in {"frechet", "dtw"}
    preprocessing = result.diagnostics.get("preprocessing")
    assert preprocessing is not None
    assert preprocessing["segment"]["resampled_points"] >= 3


def test_match_activity_to_segment_direction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matcher reports failure when the activity starts away from the segment."""

    runner = Runner(name="Mismatch", strava_id="222", refresh_token="rt")

    segment = SegmentGeometry(
        segment_id=505,
        points=[(37.0, -122.0), (37.0003, -122.0)],
        distance_m=80.0,
    )
    shifted_points = [(lat + 0.0005, lon) for lat, lon in segment.points]
    activity = ActivityTrack(
        activity_id=909,
        points=shifted_points,
        timestamps_s=[0.0, 12.0],
    )

    monkeypatch.setattr(
        "strava_competition.matching.fetch_segment_geometry",
        lambda *_: segment,
    )
    monkeypatch.setattr(
        "strava_competition.matching.fetch_activity_stream",
        lambda *_: activity,
    )

    tolerances = Tolerances(
        start_tolerance_m=5.0,
        frechet_tolerance_m=25.0,
        coverage_threshold=0.9,
        simplification_tolerance_m=2.0,
        resample_interval_m=5.0,
    )

    result = match_activity_to_segment(
        runner,
        activity.activity_id,
        segment.segment_id,
        tolerances,
    )

    assert result.matched is False
    assert result.diagnostics.get("failure_reason") in {
        "direction_check_failed",
        "coverage_threshold_not_met",
    }


def test_prepare_geometry_enforces_point_budget() -> None:
    """Large polylines should be reduced to the configured point budgets."""

    point_count = MATCHING_MAX_SIMPLIFIED_POINTS * 2
    latlon = [(37.0 + i * 1e-5, -122.0) for i in range(point_count)]
    segment = SegmentGeometry(
        segment_id=707,
        points=latlon,
        distance_m=float(point_count * 10.0),
    )
    prepared = prepare_geometry(
        segment,
        simplification_tolerance_m=0.5,
        resample_interval_m=1.0,
    )

    assert prepared.simplified_points.shape[0] <= MATCHING_MAX_SIMPLIFIED_POINTS
    assert prepared.resampled_points.shape[0] <= MATCHING_MAX_RESAMPLED_POINTS
    assert prepared.simplification_capped or prepared.resample_capped


def test_segment_cache_scope_clears_between_invocations(
    segment_geometry: SegmentGeometry,
) -> None:
    """segment_cache_scope should isolate cached entries per request."""

    prepared, _ = _prepare_pair(
        segment_geometry, _build_basic_activity(segment_geometry)
    )
    cached = build_cached_segment(prepared)
    cache_key = ("initial", 1)
    SEGMENT_CACHE.set(cache_key, cached)

    with segment_cache_scope():
        assert SEGMENT_CACHE.get(cache_key) is None
        scoped_key = ("scoped", 2)
        SEGMENT_CACHE.set(scoped_key, cached)
        assert SEGMENT_CACHE.get(scoped_key) is not None

    assert SEGMENT_CACHE.get(cache_key) is None
    assert SEGMENT_CACHE.get(("scoped", 2)) is None
