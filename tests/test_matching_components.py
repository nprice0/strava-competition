"""Unit tests covering the segment matching components."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pytest

from strava_competition.config import (
    MATCHING_COVERAGE_THRESHOLD,
    MATCHING_FRECHET_TOLERANCE_M,
    MATCHING_MAX_OFFSET_M,
    MATCHING_MAX_RESAMPLED_POINTS,
    MATCHING_MAX_SIMPLIFIED_POINTS,
    MATCHING_RESAMPLE_INTERVAL_M,
    MATCHING_SIMPLIFICATION_TOLERANCE_M,
    MATCHING_START_TOLERANCE_M,
    STRAVA_API_CAPTURE_DIR,
)
from strava_competition.matching import (
    MatchResult,
    Tolerances,
    _clip_activity_to_gate_window,
    _compute_coverage_diagnostics,
    _plane_crossing_candidates,
    _refine_coverage_window,
    _select_end_crossing,
    _select_timing_indices,
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
from strava_competition.matching.validation import (
    CoverageResult,
    check_direction,
    compute_coverage,
)
from strava_competition.models import Runner
from strava_competition.utils import json_dumps_sorted


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


def _capture_path(signature_payload: dict) -> Path:
    """Return the capture file path for a deterministic request payload."""

    signature = sha256(json_dumps_sorted(signature_payload).encode("utf-8")).hexdigest()
    base = Path(STRAVA_API_CAPTURE_DIR)
    return base / signature[:2] / signature[2:4] / f"{signature}.json"


def _load_capture_response(signature_payload: dict) -> Optional[dict]:
    """Return recorded Strava API response payload, if available."""

    path = _capture_path(signature_payload)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    return record.get("response")


def _require_capture_response(signature_payload: dict) -> dict:
    """Return capture payload or raise a helpful assertion when missing."""

    payload = _load_capture_response(signature_payload)
    if payload is None:
        raise AssertionError(f"Missing capture file {_capture_path(signature_payload)}")
    return payload


_NEIL_PRICE_STRAVA_ID = "35599907"
_NEIL_PRICE_SEGMENT_ID = 38_987_500
_NEIL_PRICE_ACTIVITY_ID = 14_661_937_369
_NEIL_PRICE_ELAPSED_S = 1893.0
_NEIL_PRICE_START_INDEX = 2665
_NEIL_PRICE_END_INDEX = 4550
_NEIL_PRICE_PRIOR_ACTIVITY_ID = 14_471_169_094
_NEIL_PRICE_PRIOR_ELAPSED_S = 2118.0

_ANDREW_KNOTT_STRAVA_ID = "21144580"
_ANDREW_KNOTT_SS25_ACTIVITY_ID = 14_547_101_577
_ANDREW_KNOTT_SS25_ELAPSED_S = 1898.0
_ANDREW_KNOTT_SWORD_ACTIVITY_ID = 15_053_677_684
_ANDREW_KNOTT_SWORD_ELAPSED_S = 1272.0

_LUKE_SIBIETA_STRAVA_ID = "6340998"
_LUKE_SIBIETA_SS25_ACTIVITY_ID = 14_618_107_187
_LUKE_SIBIETA_SS25_ELAPSED_S = 1729.0

_BEN_WERNICK_STRAVA_ID = "19923466"
_BEN_WERNICK_SS25_ACTIVITY_ID = 14_422_381_338
_BEN_WERNICK_SS25_ELAPSED_S = 1579.0
_BEN_WERNICK_CANNON_ACTIVITY_ID = 14_783_453_586
_BEN_WERNICK_CANNON_ELAPSED_S = 619.0
_BEN_WERNICK_CANNON_SEGMENT_ID = 38_955_187

_SS25_START_DATE_LOCAL = "2025-05-08T00:00:00+00:00"
_SS25_END_DATE_LOCAL = "2025-06-02T00:00:00+00:00"
_CANNON_START_DATE_LOCAL = "2025-06-01T00:00:00+00:00"
_CANNON_END_DATE_LOCAL = "2025-06-16T00:00:00+00:00"
_SWORD_SEGMENT_ID = 25_947_126
_SWORD_START_DATE_LOCAL = "2025-07-01T00:00:00+00:00"
_SWORD_END_DATE_LOCAL = "2025-07-14T00:00:00+00:00"


def _load_segment_geometry_from_capture(
    *, strava_id: str, segment_id: int
) -> SegmentGeometry:
    """Return segment geometry based on a captured Strava response."""

    payload = _require_capture_response(
        {
            "method": "GET",
            "url": f"https://www.strava.com/api/v3/segments/{segment_id}",
            "identity": strava_id,
            "params": {},
            "body": {},
        }
    )
    polyline = (payload.get("map") or {}).get("polyline")
    if not polyline:
        raise AssertionError("Segment capture missing polyline data")
    metadata = {
        "name": payload.get("name"),
        "raw": payload,
    }
    return SegmentGeometry(
        segment_id=segment_id,
        points=[],
        distance_m=float(payload.get("distance", 0.0)),
        polyline=polyline,
        metadata=metadata,
    )


def _load_neil_price_segment() -> SegmentGeometry:
    """Return SS25-02 geometry using Neil Price's capture payload."""

    return _load_segment_geometry_from_capture(
        strava_id=_NEIL_PRICE_STRAVA_ID,
        segment_id=_NEIL_PRICE_SEGMENT_ID,
    )


def _load_sword_segment() -> SegmentGeometry:
    """Return the Sword segment geometry from Ben Wernick's capture."""

    return _load_segment_geometry_from_capture(
        strava_id=_BEN_WERNICK_STRAVA_ID,
        segment_id=_SWORD_SEGMENT_ID,
    )


def _load_neil_price_activity_stream_by_id(activity_id: int) -> ActivityTrack:
    """Return Neil Price's SS25-02 activity track for a specific activity."""

    return _load_activity_stream_from_capture(
        strava_id=_NEIL_PRICE_STRAVA_ID,
        activity_id=activity_id,
    )


def _load_neil_price_activity_stream() -> ActivityTrack:
    """Return Neil Price's SS25-02 activity track from capture streams."""

    return _load_neil_price_activity_stream_by_id(_NEIL_PRICE_ACTIVITY_ID)


def _load_neil_price_effort_for_activity(activity_id: int) -> dict:
    """Return Strava's official SS25-02 effort entry for a specific activity."""

    return _load_segment_effort_entry(
        strava_id=_NEIL_PRICE_STRAVA_ID,
        segment_id=_NEIL_PRICE_SEGMENT_ID,
        start_date_local=_SS25_START_DATE_LOCAL,
        end_date_local=_SS25_END_DATE_LOCAL,
        activity_id=activity_id,
    )


def _load_neil_price_effort_entry() -> dict:
    """Return Strava's official effort entry for Neil Price on SS25-02."""

    return _load_neil_price_effort_for_activity(_NEIL_PRICE_ACTIVITY_ID)


def _load_activity_stream_from_capture(
    *, strava_id: str, activity_id: int
) -> ActivityTrack:
    """Return an activity stream for any runner using captured samples."""

    stream_payload = _require_capture_response(
        {
            "method": "GET",
            "url": f"https://www.strava.com/api/v3/activities/{activity_id}/streams",
            "identity": strava_id,
            "params": {
                "keys": "latlng,time",
                "key_by_type": "true",
                "resolution": "high",
                "series_type": "time",
            },
            "body": {},
        }
    )
    latlng_stream = stream_payload.get("latlng") or {}
    time_stream = stream_payload.get("time") or {}
    latlng = latlng_stream.get("data")
    timestamps = time_stream.get("data")
    if not latlng or not timestamps or len(latlng) != len(timestamps):
        raise AssertionError("Activity stream capture missing aligned samples")
    points = [(float(lat), float(lon)) for lat, lon in latlng]
    times = [float(value) for value in timestamps]
    return ActivityTrack(
        activity_id=activity_id,
        points=points,
        timestamps_s=times,
        metadata={"source": "capture"},
    )


def _load_segment_effort_entry(
    *,
    strava_id: str,
    segment_id: int,
    start_date_local: str,
    end_date_local: str,
    activity_id: int,
) -> dict:
    """Return Strava's official effort entry for a runner/activity combo."""

    efforts_payload = _require_capture_response(
        {
            "method": "GET",
            "url": "https://www.strava.com/api/v3/segment_efforts",
            "identity": strava_id,
            "params": {
                "segment_id": segment_id,
                "start_date_local": start_date_local,
                "end_date_local": end_date_local,
                "per_page": 200,
                "page": 1,
            },
            "body": {},
        }
    )
    if not isinstance(efforts_payload, list):
        raise AssertionError("Segment efforts capture returned unexpected payload")
    for effort in efforts_payload:
        activity = effort.get("activity") or {}
        if int(activity.get("id", 0)) == activity_id:
            return effort
    raise AssertionError("Captured efforts missing requested activity")


def _load_ben_wernick_cannon_segment() -> SegmentGeometry:
    """Return the 'It's a ball that comes out a cannon' segment geometry."""

    return _load_segment_geometry_from_capture(
        strava_id=_BEN_WERNICK_STRAVA_ID,
        segment_id=_BEN_WERNICK_CANNON_SEGMENT_ID,
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


def test_compute_coverage_ignores_points_past_finish() -> None:
    """Coverage ratio should be zero when activity samples stay beyond the finish."""

    segment_points = np.array([[0.0, 0.0], [100.0, 0.0]])
    post_finish_activity = np.array([[120.0, 0.0], [150.0, 0.0]])

    coverage = compute_coverage(post_finish_activity, segment_points)

    assert coverage.coverage_ratio == pytest.approx(0.0, abs=1e-6)
    assert coverage.coverage_bounds is not None
    start, end = coverage.coverage_bounds
    assert start == pytest.approx(100.0, abs=1e-6)
    assert end == pytest.approx(100.0, abs=1e-6)


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


def test_timing_estimation_respects_gate_interpolation(
    segment_geometry: SegmentGeometry,
) -> None:
    """Timing interpolation should use gate hints instead of raw sample indices."""

    activity_points = [
        (segment_geometry.points[0][0] - 0.00005, segment_geometry.points[0][1]),
        (segment_geometry.points[0][0] + 0.00005, segment_geometry.points[0][1]),
        segment_geometry.points[1],
        (segment_geometry.points[2][0] - 0.00005, segment_geometry.points[2][1]),
        (segment_geometry.points[2][0] + 0.00005, segment_geometry.points[2][1]),
    ]
    activity = ActivityTrack(
        activity_id=888,
        points=activity_points,
        timestamps_s=[0.0, 10.0, 30.0, 45.0, 55.0],
    )

    prepared_segment, prepared_activity = _prepare_pair(segment_geometry, activity)
    (
        coverage,
        _diag,
        timing_bounds,
        timing_indices,
        gate_hints,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=30.0,
        start_tolerance_m=25.0,
    )

    timing_range = timing_bounds or coverage.coverage_bounds
    assert timing_range is not None
    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        timing_range,
        projections=coverage.projections,
        sample_indices=timing_indices,
        gate_hints=gate_hints,
    )

    assert estimate.entry_time_s == pytest.approx(5.0, abs=1e-2)
    assert estimate.exit_time_s == pytest.approx(50.0, abs=1e-2)
    assert estimate.elapsed_time_s == pytest.approx(45.0, abs=1e-2)


def test_finish_gate_prefers_first_crossing_after_entry(
    segment_geometry: SegmentGeometry,
) -> None:
    """Finish timing should attach to the first post-entry plane crossing."""

    start_lat, start_lon = segment_geometry.points[0]
    mid_lat, mid_lon = segment_geometry.points[1]
    end_lat, end_lon = segment_geometry.points[-1]
    activity_points = [
        (start_lat - 0.00010, start_lon),
        (start_lat + 0.00002, start_lon),
        (mid_lat, mid_lon),
        (end_lat - 0.00002, end_lon),
        (end_lat + 0.00002, end_lon),
        (end_lat + 0.00025, end_lon),
        (end_lat - 0.00010, end_lon),
        (end_lat + 0.00005, end_lon),
    ]
    timestamps = [0.0, 10.0, 40.0, 60.0, 70.0, 90.0, 120.0, 150.0]
    activity = ActivityTrack(
        activity_id=990,
        points=activity_points,
        timestamps_s=timestamps,
    )

    prepared_segment, prepared_activity = _prepare_pair(segment_geometry, activity)
    (
        coverage,
        diag,
        timing_bounds,
        timing_indices,
        gate_hints,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=50.0,
        start_tolerance_m=25.0,
    )

    assert "end_crossing_offsets_m" in diag
    offsets = diag["end_crossing_offsets_m"]
    assert offsets is not None
    assert max(abs(o) for o in offsets if o is not None) < 15.0
    timing_range = timing_bounds or coverage.coverage_bounds
    assert timing_range is not None

    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        timing_range,
        projections=coverage.projections,
        sample_indices=timing_indices,
        gate_hints=gate_hints,
    )

    assert estimate.exit_time_s == pytest.approx(65.0, abs=1e-2)
    assert estimate.elapsed_time_s < 90.0


def test_select_timing_indices_prefers_first_finish() -> None:
    """Timing refinement should favour the earliest crossing that reaches the finish."""

    chunk_indices = np.array([10, 11, 12, 13, 14, 15, 40, 41, 42], dtype=int)
    chunk_projections = np.array(
        [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 10.0, 60.0, 102.0],
        dtype=float,
    )

    entry_idx, exit_idx = _select_timing_indices(
        chunk_indices,
        chunk_projections,
        raw_start=0.0,
        raw_end=102.0,
    )

    assert entry_idx == 10
    assert exit_idx == 15


def test_select_timing_indices_honours_first_gate_entry() -> None:
    """Entry selection should stick with the earliest near-start projection."""

    chunk_indices = np.array([7, 8, 9, 10, 11, 12, 13], dtype=int)
    chunk_projections = np.array(
        [1.4, 1.6, 2.1, 0.0, 200.0, 400.0, 500.0],
        dtype=float,
    )

    entry_idx, exit_idx = _select_timing_indices(
        chunk_indices,
        chunk_projections,
        raw_start=0.0,
        raw_end=500.0,
    )

    assert entry_idx == 7
    assert exit_idx == 13


def test_select_end_crossing_prefers_oriented_pair() -> None:
    """Finish crossing selection should prioritise forward orientation."""

    values = np.array([10.0, 4.5, -2.5, -0.3, 0.1], dtype=float)
    candidates = _plane_crossing_candidates(values, min_span_m=0.2)
    assert candidates  # sanity

    selection = _select_end_crossing(values, candidates, offset=0)
    assert selection == (1, 2)


def test_select_end_crossing_allows_early_mismatch() -> None:
    """Later oriented crossings should not displace an early near-plane finish."""

    values = np.ones(1200, dtype=float) * 10.0
    values[100] = -0.8
    values[101] = 1.4
    values[900] = 12.0
    values[901] = -12.0
    candidates = [(100, 101), (900, 901)]

    selection = _select_end_crossing(values, candidates, offset=0)

    assert selection == (100, 101)


def test_select_end_crossing_skips_backtrack_when_oriented_close() -> None:
    """Finish crossing should favour oriented sample pairs when costs stay low."""

    values = np.ones(500, dtype=float) * 8.0
    values[150] = -1.2
    values[151] = 1.8
    values[320] = 0.3
    values[321] = -0.4
    candidates = [(150, 151), (320, 321)]

    selection = _select_end_crossing(values, candidates, offset=0)

    assert selection == (320, 321)


def test_gate_window_prefers_first_lap_after_loop() -> None:
    """Gate trimming should lock to the first lap even if the runner loops."""

    segment = SegmentGeometry(
        segment_id=202,
        points=[
            (37.0000, -122.0000),
            (37.0040, -122.0000),
            (37.0080, -122.0000),
        ],
        distance_m=800.0,
    )
    activity_points = [
        (36.9995, -122.0000),
        (37.0001, -122.0000),
        (37.0040, -122.0000),
        (37.0080, -122.0000),
        (37.0085, -122.0000),
        (36.9999, -122.0000),
        (37.0002, -122.0000),
        (37.0041, -122.0000),
        (37.0081, -122.0000),
    ]
    timestamps = [0.0, 45.0, 165.0, 225.0, 255.0, 600.0, 660.0, 795.0, 855.0]
    activity = ActivityTrack(
        activity_id=2001,
        points=activity_points,
        timestamps_s=timestamps,
    )

    prepared_segment, prepared_activity = _prepare_pair(segment, activity)
    (
        coverage,
        _diag,
        timing_bounds,
        timing_indices,
        gate_hints,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=60.0,
        start_tolerance_m=25.0,
    )

    timing_range = timing_bounds or coverage.coverage_bounds
    assert timing_range is not None

    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        timing_range,
        projections=coverage.projections,
        sample_indices=timing_indices,
        gate_hints=gate_hints,
    )

    assert estimate.entry_time_s is not None
    assert estimate.exit_time_s is not None
    assert estimate.entry_time_s < 200.0
    assert estimate.exit_time_s < 320.0
    assert estimate.elapsed_time_s == pytest.approx(180.0, abs=15.0)


def test_finish_gate_prioritises_expected_orientation() -> None:
    """Finish gate selection should favour forward crossings even if later."""

    values = np.array([0.0, -1.2, 1.1, -0.8], dtype=float)
    required_span = 0.5
    offset = 1
    candidates = _plane_crossing_candidates(values[offset:], required_span)
    # Sanity: we produced two crossings with opposite orientations.
    assert candidates == [(0, 1), (1, 2)]

    selected = _select_end_crossing(values, candidates, offset)

    assert selected == (1, 2)


def test_finish_gate_uses_near_plane_fallback() -> None:
    """Finish gate selection should fall back to the closest crossing."""

    values = np.array([-25.0, -10.0, -5.0, -1.2, 0.6, 2.0, 8.0], dtype=float)
    required_span = 3.0
    offset = 0

    candidates = _plane_crossing_candidates(values, required_span)
    # The broad span candidate remains first; the near-plane option is appended.
    assert candidates == [(2, 6), (3, 4)]

    selected = _select_end_crossing(values, candidates, offset)

    assert selected == (3, 4)


def test_coverage_trims_offsets_outside_gate_window(
    segment_geometry: SegmentGeometry,
) -> None:
    """Coverage diagnostics should ignore large offsets before the start gate."""

    far_lon = segment_geometry.points[0][1] + 0.03
    activity_points = [
        (segment_geometry.points[0][0] - 0.0001, far_lon),
        (segment_geometry.points[0][0] - 0.00002, segment_geometry.points[0][1]),
        *segment_geometry.points,
        (segment_geometry.points[-1][0] + 0.00002, segment_geometry.points[-1][1]),
    ]
    timestamps = [-30.0, -10.0, 0.0, 12.0, 24.0, 36.0]
    activity = ActivityTrack(
        activity_id=889,
        points=activity_points,
        timestamps_s=timestamps,
    )

    prepared_segment, prepared_activity = _prepare_pair(segment_geometry, activity)
    (
        coverage,
        diag,
        _bounds,
        _indices,
        _gate_hints,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=30.0,
        start_tolerance_m=25.0,
    )

    raw_max = diag.get("raw_max_offset_m")
    assert raw_max is not None and raw_max > 500.0
    assert coverage.max_offset_m is not None
    assert coverage.max_offset_m < 50.0


def test_refine_window_reverts_on_collapsed_span() -> None:
    """Refinement should fall back when the surviving span is too small."""

    projections = np.linspace(0.0, 1000.0, num=100)
    offsets = np.full(100, 200.0)
    offsets[:5] = 1.0
    coverage = CoverageResult(
        coverage_ratio=1.0,
        coverage_bounds=(0.0, 1000.0),
        projections=projections,
        max_offset_m=float(np.max(offsets)),
        offsets=offsets,
    )
    segment_points = np.column_stack((np.linspace(0.0, 1000.0, num=101), np.zeros(101)))

    refined = _refine_coverage_window(
        coverage,
        segment_points,
        max_offset_threshold=30.0,
        start_tolerance_m=25.0,
    )

    assert refined is None


def test_ss25_02_neil_price_elapsed_time() -> None:
    """Neil Price's SS25-02 effort should match Strava's elapsed time."""

    segment = _load_neil_price_segment()
    activity = _load_neil_price_activity_stream()
    effort_entry = _load_neil_price_effort_entry()

    prepared_segment = prepare_geometry(
        segment,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )
    prepared_activity = prepare_activity(
        activity,
        transformer=prepared_segment.transformer,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )

    (
        coverage,
        diag,
        timing_bounds,
        timing_indices,
        gate_hints,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=MATCHING_MAX_OFFSET_M,
        start_tolerance_m=MATCHING_START_TOLERANCE_M,
    )

    assert coverage.coverage_bounds is not None
    assert coverage.projections is not None
    assert diag.get("crosses_start") is True
    assert diag.get("crosses_end") is True

    assert timing_indices is not None
    entry_idx, exit_idx = timing_indices
    assert entry_idx < exit_idx
    gate_slice = diag.get("gate_slice_indices")
    assert gate_slice is not None
    gate_start, gate_end = gate_slice
    assert gate_start <= entry_idx <= gate_end
    assert gate_start <= exit_idx <= gate_end

    timing_range = timing_bounds or coverage.coverage_bounds
    assert timing_range is not None
    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        timing_range,
        projections=coverage.projections,
        sample_indices=timing_indices,
        gate_hints=gate_hints,
    )

    expected_elapsed = float(effort_entry.get("elapsed_time", _NEIL_PRICE_ELAPSED_S))
    assert estimate.elapsed_time_s == pytest.approx(expected_elapsed, abs=0.5)
    assert estimate.entry_index is not None
    assert estimate.exit_index is not None
    assert abs(estimate.entry_index - _NEIL_PRICE_START_INDEX) <= 1
    assert abs(estimate.exit_index - _NEIL_PRICE_END_INDEX) <= 1

    official_start_time = activity.timestamps_s[_NEIL_PRICE_START_INDEX]
    official_end_time = activity.timestamps_s[_NEIL_PRICE_END_INDEX]
    assert estimate.entry_time_s == pytest.approx(official_start_time, abs=0.2)
    assert estimate.exit_time_s == pytest.approx(official_end_time, abs=0.2)


def test_ss25_02_neil_price_prior_elapsed_time() -> None:
    """Neil Price's earlier SS25-02 effort should keep Strava's elapsed time."""

    segment = _load_neil_price_segment()
    activity = _load_neil_price_activity_stream_by_id(_NEIL_PRICE_PRIOR_ACTIVITY_ID)
    effort_entry = _load_neil_price_effort_for_activity(_NEIL_PRICE_PRIOR_ACTIVITY_ID)

    prepared_segment = prepare_geometry(
        segment,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )
    prepared_activity = prepare_activity(
        activity,
        transformer=prepared_segment.transformer,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )

    (
        coverage,
        diag,
        timing_bounds,
        timing_indices,
        gate_hints,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=MATCHING_MAX_OFFSET_M,
        start_tolerance_m=MATCHING_START_TOLERANCE_M,
    )

    assert diag.get("crosses_start") is True
    assert diag.get("crosses_end") is True
    timing_range = timing_bounds or coverage.coverage_bounds
    assert timing_range is not None
    assert timing_indices is not None
    entry_idx, exit_idx = timing_indices
    assert entry_idx < exit_idx

    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        timing_range,
        projections=coverage.projections,
        sample_indices=timing_indices,
        gate_hints=gate_hints,
    )

    expected_elapsed = float(
        effort_entry.get("elapsed_time", _NEIL_PRICE_PRIOR_ELAPSED_S)
    )
    assert estimate.elapsed_time_s == pytest.approx(expected_elapsed, abs=0.5)
    assert estimate.entry_time_s is not None
    assert estimate.exit_time_s is not None
    assert estimate.exit_time_s - estimate.entry_time_s == pytest.approx(
        expected_elapsed,
        abs=0.5,
    )
    assert estimate.entry_time_s <= 60.0


def test_ss25_02_andrew_knott_elapsed_time() -> None:
    """Andrew Knott's fastest SS25-02 attempt should track Strava's effort."""

    segment = _load_neil_price_segment()
    activity = _load_activity_stream_from_capture(
        strava_id=_ANDREW_KNOTT_STRAVA_ID,
        activity_id=_ANDREW_KNOTT_SS25_ACTIVITY_ID,
    )
    effort_entry = _load_segment_effort_entry(
        strava_id=_ANDREW_KNOTT_STRAVA_ID,
        segment_id=_NEIL_PRICE_SEGMENT_ID,
        start_date_local=_SS25_START_DATE_LOCAL,
        end_date_local=_SS25_END_DATE_LOCAL,
        activity_id=_ANDREW_KNOTT_SS25_ACTIVITY_ID,
    )

    prepared_segment = prepare_geometry(
        segment,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )
    prepared_activity = prepare_activity(
        activity,
        transformer=prepared_segment.transformer,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )

    (
        coverage,
        diag,
        timing_bounds,
        timing_indices,
        gate_hints,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=MATCHING_MAX_OFFSET_M,
        start_tolerance_m=MATCHING_START_TOLERANCE_M,
    )

    assert diag.get("crosses_start") is True
    assert diag.get("crosses_end") is True
    timing_range = timing_bounds or coverage.coverage_bounds
    assert timing_range is not None
    assert timing_indices is not None
    gate_slice = diag.get("gate_slice_indices")
    assert gate_slice is not None
    entry_idx, exit_idx = timing_indices
    gate_start, gate_end = gate_slice
    assert gate_start <= entry_idx <= gate_end
    assert gate_start <= exit_idx <= gate_end

    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        timing_range,
        projections=coverage.projections,
        sample_indices=timing_indices,
        gate_hints=gate_hints,
    )

    expected_elapsed = float(
        effort_entry.get("elapsed_time", _ANDREW_KNOTT_SS25_ELAPSED_S)
    )
    assert estimate.elapsed_time_s == pytest.approx(expected_elapsed, abs=0.5)

    official_start = int(effort_entry.get("start_index", 0))
    official_end = int(effort_entry.get("end_index", gate_end))
    assert estimate.entry_index is not None
    assert abs(estimate.entry_index - official_start) <= 1
    assert estimate.exit_index is not None
    assert abs(estimate.exit_index - official_end) <= 50


def test_ss25_02_luke_sibieta_elapsed_time() -> None:
    """Luke Sibieta's SS25-02 effort should stay aligned with Strava."""

    segment = _load_neil_price_segment()
    activity = _load_activity_stream_from_capture(
        strava_id=_LUKE_SIBIETA_STRAVA_ID,
        activity_id=_LUKE_SIBIETA_SS25_ACTIVITY_ID,
    )
    effort_entry = _load_segment_effort_entry(
        strava_id=_LUKE_SIBIETA_STRAVA_ID,
        segment_id=_NEIL_PRICE_SEGMENT_ID,
        start_date_local=_SS25_START_DATE_LOCAL,
        end_date_local=_SS25_END_DATE_LOCAL,
        activity_id=_LUKE_SIBIETA_SS25_ACTIVITY_ID,
    )

    prepared_segment = prepare_geometry(
        segment,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )
    prepared_activity = prepare_activity(
        activity,
        transformer=prepared_segment.transformer,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )

    (
        coverage,
        diag,
        timing_bounds,
        timing_indices,
        gate_hints,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=MATCHING_MAX_OFFSET_M,
        start_tolerance_m=MATCHING_START_TOLERANCE_M,
    )

    assert diag.get("crosses_start") is True
    assert diag.get("crosses_end") is True
    timing_range = timing_bounds or coverage.coverage_bounds
    assert timing_range is not None
    assert timing_indices is not None
    gate_slice = diag.get("gate_slice_indices")
    assert gate_slice is not None
    entry_idx, exit_idx = timing_indices
    gate_start, gate_end = gate_slice
    assert gate_start <= entry_idx <= gate_end
    assert gate_start <= exit_idx <= gate_end

    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        timing_range,
        projections=coverage.projections,
        sample_indices=timing_indices,
        gate_hints=gate_hints,
    )

    expected_elapsed = float(
        effort_entry.get("elapsed_time", _LUKE_SIBIETA_SS25_ELAPSED_S)
    )
    assert estimate.elapsed_time_s == pytest.approx(expected_elapsed, abs=0.5)

    official_start = int(effort_entry.get("start_index", 0))
    official_end = int(effort_entry.get("end_index", gate_end))
    assert estimate.entry_index is not None
    assert abs(estimate.entry_index - official_start) <= 1
    assert estimate.exit_index is not None
    assert abs(estimate.exit_index - official_end) <= 1


def test_ss25_02_ben_wernick_elapsed_time() -> None:
    """Ben Wernick's SS25-02 effort should remain near Strava's elapsed time."""

    segment = _load_neil_price_segment()
    activity = _load_activity_stream_from_capture(
        strava_id=_BEN_WERNICK_STRAVA_ID,
        activity_id=_BEN_WERNICK_SS25_ACTIVITY_ID,
    )
    effort_entry = _load_segment_effort_entry(
        strava_id=_BEN_WERNICK_STRAVA_ID,
        segment_id=_NEIL_PRICE_SEGMENT_ID,
        start_date_local=_SS25_START_DATE_LOCAL,
        end_date_local=_SS25_END_DATE_LOCAL,
        activity_id=_BEN_WERNICK_SS25_ACTIVITY_ID,
    )

    prepared_segment = prepare_geometry(
        segment,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )
    prepared_activity = prepare_activity(
        activity,
        transformer=prepared_segment.transformer,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )

    (
        coverage,
        diag,
        timing_bounds,
        timing_indices,
        gate_hints,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=MATCHING_MAX_OFFSET_M,
        start_tolerance_m=MATCHING_START_TOLERANCE_M,
    )

    assert diag.get("crosses_start") is True
    assert diag.get("crosses_end") is True
    timing_range = timing_bounds or coverage.coverage_bounds
    assert timing_range is not None
    assert timing_indices is not None
    gate_slice = diag.get("gate_slice_indices")
    assert gate_slice is not None
    entry_idx, exit_idx = timing_indices
    gate_start, gate_end = gate_slice
    assert gate_start <= entry_idx <= gate_end
    assert gate_start <= exit_idx <= gate_end

    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        timing_range,
        projections=coverage.projections,
        sample_indices=timing_indices,
        gate_hints=gate_hints,
    )

    expected_elapsed = float(
        effort_entry.get("elapsed_time", _BEN_WERNICK_SS25_ELAPSED_S)
    )
    assert estimate.elapsed_time_s == pytest.approx(expected_elapsed, abs=1.0)

    official_start = int(effort_entry.get("start_index", 0))
    official_end = int(effort_entry.get("end_index", gate_end))
    assert estimate.entry_index is not None
    assert abs(estimate.entry_index - official_start) <= 1
    assert estimate.exit_index is not None
    assert abs(estimate.exit_index - official_end) <= 1


def test_sword_andrew_knott_elapsed_time() -> None:
    """Andrew Knott's Sword effort should respect the official gate window."""

    segment = _load_sword_segment()
    activity = _load_activity_stream_from_capture(
        strava_id=_ANDREW_KNOTT_STRAVA_ID,
        activity_id=_ANDREW_KNOTT_SWORD_ACTIVITY_ID,
    )
    effort_entry = _load_segment_effort_entry(
        strava_id=_ANDREW_KNOTT_STRAVA_ID,
        segment_id=_SWORD_SEGMENT_ID,
        start_date_local=_SWORD_START_DATE_LOCAL,
        end_date_local=_SWORD_END_DATE_LOCAL,
        activity_id=_ANDREW_KNOTT_SWORD_ACTIVITY_ID,
    )

    prepared_segment = prepare_geometry(
        segment,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )
    prepared_activity = prepare_activity(
        activity,
        transformer=prepared_segment.transformer,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )

    (
        coverage,
        diag,
        timing_bounds,
        timing_indices,
        gate_hints,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=MATCHING_MAX_OFFSET_M,
        start_tolerance_m=MATCHING_START_TOLERANCE_M,
    )

    assert diag.get("crosses_start") is True
    assert diag.get("crosses_end") is True
    timing_range = timing_bounds or coverage.coverage_bounds
    assert timing_range is not None
    assert timing_indices is not None
    gate_slice = diag.get("gate_slice_indices")
    assert gate_slice is not None
    entry_idx, exit_idx = timing_indices
    gate_start, gate_end = gate_slice
    assert gate_start <= entry_idx <= gate_end
    assert gate_start <= exit_idx <= gate_end

    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        timing_range,
        projections=coverage.projections,
        sample_indices=timing_indices,
        gate_hints=gate_hints,
    )

    expected_elapsed = float(
        effort_entry.get("elapsed_time", _ANDREW_KNOTT_SWORD_ELAPSED_S)
    )
    assert estimate.elapsed_time_s == pytest.approx(expected_elapsed, abs=3.0)

    official_start = int(effort_entry.get("start_index", gate_start))
    official_end = int(effort_entry.get("end_index", gate_end))
    assert estimate.entry_index is not None
    assert abs(estimate.entry_index - official_start) <= 2
    assert estimate.exit_index is not None
    assert abs(estimate.exit_index - official_end) <= 2


def test_cannon_ben_wernick_elapsed_time() -> None:
    """Ben Wernick's cannon-segment effort should match Strava's elapsed time."""

    segment = _load_ben_wernick_cannon_segment()
    activity = _load_activity_stream_from_capture(
        strava_id=_BEN_WERNICK_STRAVA_ID,
        activity_id=_BEN_WERNICK_CANNON_ACTIVITY_ID,
    )
    effort_entry = _load_segment_effort_entry(
        strava_id=_BEN_WERNICK_STRAVA_ID,
        segment_id=_BEN_WERNICK_CANNON_SEGMENT_ID,
        start_date_local=_CANNON_START_DATE_LOCAL,
        end_date_local=_CANNON_END_DATE_LOCAL,
        activity_id=_BEN_WERNICK_CANNON_ACTIVITY_ID,
    )

    prepared_segment = prepare_geometry(
        segment,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )
    prepared_activity = prepare_activity(
        activity,
        transformer=prepared_segment.transformer,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )

    (
        coverage,
        diag,
        timing_bounds,
        timing_indices,
        gate_hints,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=MATCHING_MAX_OFFSET_M,
        start_tolerance_m=MATCHING_START_TOLERANCE_M,
    )

    assert diag.get("crosses_start") is True
    assert diag.get("crosses_end") is True
    timing_range = timing_bounds or coverage.coverage_bounds
    assert timing_range is not None
    assert timing_indices is not None
    gate_slice = diag.get("gate_slice_indices")
    assert gate_slice is not None
    entry_idx, exit_idx = timing_indices
    gate_start, gate_end = gate_slice
    assert gate_start <= entry_idx <= gate_end
    assert gate_start <= exit_idx <= gate_end

    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        timing_range,
        projections=coverage.projections,
        sample_indices=timing_indices,
        gate_hints=gate_hints,
    )

    expected_elapsed = float(
        effort_entry.get("elapsed_time", _BEN_WERNICK_CANNON_ELAPSED_S)
    )
    assert estimate.elapsed_time_s == pytest.approx(expected_elapsed, abs=0.5)

    official_start = int(effort_entry.get("start_index", 0))
    official_end = int(effort_entry.get("end_index", gate_end))
    assert estimate.entry_index is not None
    assert abs(estimate.entry_index - official_start) <= 1
    assert estimate.exit_index is not None
    assert abs(estimate.exit_index - official_end) <= 1


def test_match_activity_accepts_neil_price_fast_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """match_activity_to_segment should now validate Neil Price's fastest run."""

    runner = Runner(
        name="Neil Price", strava_id=_NEIL_PRICE_STRAVA_ID, refresh_token="tk"
    )
    segment = _load_neil_price_segment()
    activity = _load_neil_price_activity_stream()

    monkeypatch.setattr(
        "strava_competition.matching.fetch_segment_geometry",
        lambda *_args, **_kwargs: segment,
    )
    monkeypatch.setattr(
        "strava_competition.matching.fetch_activity_stream",
        lambda *_args, **_kwargs: activity,
    )

    tolerances = Tolerances(
        start_tolerance_m=MATCHING_START_TOLERANCE_M,
        frechet_tolerance_m=MATCHING_FRECHET_TOLERANCE_M,
        coverage_threshold=MATCHING_COVERAGE_THRESHOLD,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )

    result = match_activity_to_segment(
        runner,
        activity.activity_id,
        segment.segment_id,
        tolerances,
    )

    assert result.matched is True
    assert result.elapsed_time_s == pytest.approx(_NEIL_PRICE_ELAPSED_S, abs=1.0)
    diagnostics = result.diagnostics
    assert diagnostics.get("similarity_method") in {"frechet", "dtw"}
    similarity_window = diagnostics.get("similarity_window")
    assert similarity_window is not None
    assert similarity_window.get("length_ratio_threshold_m") is not None


def test_match_activity_to_segment_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """match_activity_to_segment succeeds when the geometry aligns."""

    runner = Runner(name="Test Runner", strava_id="123", refresh_token="rt")

    segment = SegmentGeometry(
        segment_id=404,
        points=[(37.0, -122.0), (37.0003, -122.0), (37.0006, -122.0)],
        distance_m=120.0,
    )
    activity_points = [
        (segment.points[0][0] - 0.00005, segment.points[0][1]),
        *segment.points,
        (segment.points[-1][0] + 0.00005, segment.points[-1][1]),
    ]
    activity = ActivityTrack(
        activity_id=808,
        points=activity_points,
        timestamps_s=[-5.0, 0.0, 15.0, 30.0, 40.0],
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


def test_matcher_ignores_post_finish_diversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Similarity stage should drop divergent samples captured after the finish."""

    runner = Runner(name="Divergent", strava_id="456", refresh_token="tok")

    segment = SegmentGeometry(
        segment_id=4242,
        points=[
            (37.0, -122.0),
            (37.0002, -122.0),
            (37.0004, -122.0),
        ],
        distance_m=130.0,
    )
    tail_lon = segment.points[-1][1] - 0.02
    post_finish = (segment.points[-1][0] + 0.00002, segment.points[-1][1])
    activity = ActivityTrack(
        activity_id=9090,
        points=[
            (segment.points[0][0] - 0.00005, segment.points[0][1]),
            *segment.points,
            post_finish,
            (segment.points[-1][0], tail_lon),
            (segment.points[-1][0], tail_lon - 0.003),
        ],
        timestamps_s=[-8.0, 0.0, 12.0, 24.0, 30.0, 42.0, 70.0],
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
        start_tolerance_m=60.0,
        frechet_tolerance_m=40.0,
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

    assert result.matched is True
    similarity_window = result.diagnostics.get("similarity_window")
    assert similarity_window is not None
    assert similarity_window.get("gate_trimmed") is True


def test_gate_clipping_uses_full_activity_context() -> None:
    """Gate clipping should trim using the full resampled track when available."""

    segment_points = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
    full_activity = np.array(
        [
            [-10.0, 0.0],
            [0.0, 0.0],
            [50.0, 0.0],
            [100.0, 0.0],
            [105.0, 0.0],
            [110.0, 0.0],
        ],
        dtype=float,
    )
    trimmed_activity = full_activity[3:]

    clipped = _clip_activity_to_gate_window(
        trimmed_activity,
        segment_points,
        start_tolerance_m=60.0,
        full_activity_points=full_activity,
    )
    assert clipped.shape[0] == 2
    assert np.allclose(clipped, trimmed_activity[:2])
    assert clipped[-1, 0] == pytest.approx(105.0)

    fallback = _clip_activity_to_gate_window(
        trimmed_activity,
        segment_points,
        start_tolerance_m=60.0,
    )
    assert fallback.shape[0] == trimmed_activity.shape[0]
    assert np.allclose(fallback, trimmed_activity)


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
        "coverage_offset_exceeded",
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
