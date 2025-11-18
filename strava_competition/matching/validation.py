"""Validation helpers for directionality and coverage checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

MetricArray = NDArray[np.float64]


@dataclass(slots=True)
class DirectionCheckResult:
    """Outcome of the direction validation step."""

    matches_direction: bool
    max_start_distance_m: Optional[float]
    direction_score: Optional[float]


@dataclass(slots=True)
class CoverageResult:
    """Aggregate coverage metrics for a projected activity."""

    coverage_ratio: float
    coverage_bounds: Optional[Tuple[float, float]]
    projections: Optional[MetricArray] = None
    max_offset_m: Optional[float] = None
    offsets: Optional[MetricArray] = None


def check_direction(
    activity_points: MetricArray,
    segment_points: MetricArray,
    *,
    start_tolerance_m: float,
    projections: Optional[MetricArray] = None,
    coverage_bounds: Optional[Tuple[float, float]] = None,
) -> DirectionCheckResult:
    """Check whether the activity begins near the segment start and moves forward."""

    if len(activity_points) == 0 or len(segment_points) < 2:
        return DirectionCheckResult(
            False, max_start_distance_m=None, direction_score=None
        )

    window_points = activity_points
    window_projections = projections
    if (
        projections is not None
        and coverage_bounds is not None
        and len(projections) == len(activity_points)
    ):
        cov_start, cov_end = coverage_bounds
        # Expand the window slightly to tolerate GPS noise at entry/exit.
        margin = max(start_tolerance_m, 25.0)
        lower = max(cov_start - margin, 0.0)
        upper = cov_end + margin
        mask = (projections >= lower) & (projections <= upper)
        indices = np.nonzero(mask)[0]
        if indices.size >= 2:
            window_points = activity_points[indices[0] : indices[-1] + 1]
            window_projections = projections[indices[0] : indices[-1] + 1]

    start_point = segment_points[0]
    distances = np.linalg.norm(activity_points - start_point, axis=1)
    min_distance = float(np.min(distances))
    if min_distance > start_tolerance_m:
        return DirectionCheckResult(
            False, max_start_distance_m=min_distance, direction_score=None
        )

    seg_vector = segment_points[1] - start_point
    seg_norm = np.linalg.norm(seg_vector)
    if seg_norm == 0:
        return DirectionCheckResult(
            False, max_start_distance_m=min_distance, direction_score=None
        )
    seg_unit = seg_vector / seg_norm

    # Evaluate the fraction of movement aligned with the segment direction within the window.
    activity_vectors = np.diff(window_points, axis=0)
    if activity_vectors.size == 0:
        return DirectionCheckResult(
            True, max_start_distance_m=min_distance, direction_score=1.0
        )

    forward_fraction: Optional[float] = None
    direction_ok: Optional[bool] = None
    if window_projections is not None and coverage_bounds is not None:
        forward_fraction, direction_ok = _analyze_projection_direction(
            window_projections, coverage_bounds, start_tolerance_m
        )
    if direction_ok is None:
        dot_projections = activity_vectors @ seg_unit
        forward_fraction = float(np.mean(dot_projections > 0))
        direction_ok = forward_fraction > 0.5 and float(np.sum(dot_projections)) > 0.0

    score = max(forward_fraction or 0.0, 0.0)
    return DirectionCheckResult(
        direction_ok, max_start_distance_m=min_distance, direction_score=score
    )


def compute_coverage(
    activity_points: MetricArray,
    segment_points: MetricArray,
) -> CoverageResult:
    """Project activity points onto the segment and measure covered ratio."""

    if len(activity_points) == 0 or len(segment_points) < 2:
        return CoverageResult(0.0, None, None, None, None)

    cumulative_segment = _cumulative_distances(segment_points)
    total_length = cumulative_segment[-1]
    if total_length == 0:
        return CoverageResult(0.0, (0.0, 0.0), None, 0.0, None)

    projected, offsets = _project_onto_polyline(activity_points, segment_points)
    coverage_start_raw = float(np.min(projected))
    coverage_end_raw = float(np.max(projected))
    coverage_start = float(min(max(coverage_start_raw, 0.0), total_length))
    coverage_end = float(min(max(coverage_end_raw, 0.0), total_length))
    coverage_range = max(coverage_end - coverage_start, 0.0)
    ratio = coverage_range / total_length if total_length > 0 else 0.0
    ratio = float(min(max(ratio, 0.0), 1.0))
    max_offset = None
    if offsets.size:
        finite = offsets[np.isfinite(offsets)]
        if finite.size:
            max_offset = float(np.max(finite))
    return CoverageResult(
        coverage_ratio=ratio,
        coverage_bounds=(coverage_start, coverage_end),
        projections=projected,
        max_offset_m=max_offset,
        offsets=offsets,
    )


def _project_onto_polyline(
    points: MetricArray,
    polyline: MetricArray,
) -> Tuple[MetricArray, MetricArray]:
    """Project each point onto the polyline, returning distances and offsets."""

    segments = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    cumulative = _cumulative_distances(polyline)
    projections = np.zeros(len(points), dtype=float)
    offsets = np.full(len(points), np.inf, dtype=float)

    # Greedy search: walk each segment and find the nearest in-line projection.
    for idx, point in enumerate(points):
        best_distance = 0.0
        best_offset = np.inf
        for i, (seg_start, seg_vec, seg_len) in enumerate(
            zip(polyline[:-1], segments, segment_lengths)
        ):
            if seg_len == 0:
                continue
            vec_to_point = point - seg_start
            t = np.dot(vec_to_point, seg_vec) / (seg_len**2)
            t_clamped = min(max(t, 0.0), 1.0)
            nearest = seg_start + t_clamped * seg_vec
            offset = np.linalg.norm(point - nearest)
            if offset < best_offset:
                best_offset = offset
                best_distance = cumulative[i] + t_clamped * seg_len
        if not np.isfinite(best_offset):
            # Fallback to the closest vertex when projection lies outside both endpoints.
            dists = np.linalg.norm(polyline - point, axis=1)
            best_index = int(np.argmin(dists))
            best_distance = cumulative[best_index]
            best_offset = float(np.min(dists))
        projections[idx] = best_distance
        offsets[idx] = best_offset
    return projections, offsets


def _cumulative_distances(points: MetricArray) -> MetricArray:
    """Return cumulative distances along a polyline."""

    if len(points) == 0:
        return np.zeros(1, dtype=float)
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return cumulative


def _analyze_projection_direction(
    projections: MetricArray,
    coverage_bounds: Tuple[float, float],
    start_tolerance_m: float,
) -> Tuple[Optional[float], Optional[bool]]:
    """Infer directionality using projected distances within the covered segment span."""

    if projections.size <= 1:
        return None, None

    cov_start, cov_end = coverage_bounds
    span = max(cov_end - cov_start, 0.0)
    if span <= 0.0:
        return None, None

    band = max(min(span * 0.1, start_tolerance_m), 5.0)
    entry_threshold = cov_start + band * 0.2
    exit_threshold = cov_end - band * 0.2
    entry_candidates = np.nonzero(projections >= entry_threshold)[0]
    exit_candidates = np.nonzero(projections <= exit_threshold)[0]
    if entry_candidates.size == 0 or exit_candidates.size == 0:
        return None, None

    entry_idx = int(entry_candidates[0])
    exit_idx = int(exit_candidates[-1])
    if exit_idx <= entry_idx:
        return None, None

    focus_proj = projections[entry_idx : exit_idx + 1]
    if focus_proj.size <= 1:
        return None, None

    projection_steps = np.diff(focus_proj)
    jitter = max(1.0, span * 0.02)
    forward_fraction = float(np.mean(projection_steps > -jitter))
    net_progress = float(focus_proj[-1] - focus_proj[0])
    backtrack_tolerance = max(jitter, span * 0.25)

    if net_progress > -backtrack_tolerance and forward_fraction > 0.4:
        return forward_fraction, True

    if forward_fraction > 0.6 and _progression_hits_forward(
        focus_proj, cov_start, span
    ):
        return forward_fraction, True

    if forward_fraction > 0.6 and _median_progress_sufficient(
        focus_proj,
        backtrack_tolerance,
        jitter,
    ):
        return forward_fraction, True

    return forward_fraction, False


def _progression_hits_forward(
    focus_proj: MetricArray,
    cov_start: float,
    span: float,
) -> bool:
    """Return True when the activity reaches the far end after the near end."""

    if span <= 0.0:
        return False
    low_threshold = cov_start + span * 0.1
    high_threshold = cov_start + span * 0.9
    low_hits = np.nonzero(focus_proj >= low_threshold)[0]
    high_hits = np.nonzero(focus_proj >= high_threshold)[0]
    if low_hits.size == 0 or high_hits.size == 0:
        return False
    return bool(high_hits[0] >= low_hits[0])


def _median_progress_sufficient(
    focus_proj: MetricArray,
    backtrack_tolerance: float,
    jitter: float,
) -> bool:
    """Return True if the median entry/exit projections show forward travel."""

    window = max(5, int(focus_proj.size * 0.1))
    if window * 2 > focus_proj.size:
        return False
    median_start = float(np.median(focus_proj[:window]))
    median_end = float(np.median(focus_proj[-window:]))
    median_tolerance = max(backtrack_tolerance * 0.5, jitter)
    return bool(median_end - median_start > -median_tolerance)
