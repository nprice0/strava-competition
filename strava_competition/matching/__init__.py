"""Public entry points for the GPS-based segment matching package."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..models import Runner
from ..config import (
    MATCHING_SIMPLIFICATION_TOLERANCE_M,
    MATCHING_RESAMPLE_INTERVAL_M,
    MATCHING_START_TOLERANCE_M,
    MATCHING_FRECHET_TOLERANCE_M,
    MATCHING_COVERAGE_THRESHOLD,
    MATCHING_MAX_OFFSET_M,
)

from .fetchers import fetch_activity_stream, fetch_segment_geometry
from .models import MatchResult, Tolerances
from .preprocessing import (
    PreparedActivityTrack,
    PreparedSegmentGeometry,
    prepare_activity,
    prepare_geometry,
)
from .similarity import (
    SEGMENT_CACHE,
    build_cached_segment,
    discrete_frechet_distance,
    windowed_dtw,
)
from .timing import SegmentTimingEstimate, estimate_segment_time
from .validation import CoverageResult, check_direction, compute_coverage


@dataclass(slots=True)
class _RefinedCoverage:
    """Refined overlap metrics derived from raw coverage projections."""

    raw_bounds: Tuple[float, float]
    expanded_bounds: Tuple[float, float]
    ratio: float
    max_offset: float
    entry_index: int
    exit_index: int


def _select_timing_indices(
    projections: np.ndarray,
    indices: np.ndarray,
    raw_start: float,
    raw_end: float,
    start_tolerance_m: float,
) -> tuple[int, int]:
    """Return entry/exit sample indices nearest the segment boundaries."""

    if indices.size == 0:
        return 0, 0

    mask_values = projections[indices]
    entry_idx = int(indices[0])
    exit_idx = int(indices[-1])

    start_buffer = min(25.0, max(5.0, start_tolerance_m * 0.25))
    end_buffer = min(50.0, max(5.0, start_tolerance_m * 0.5))
    end_threshold = max(raw_end - end_buffer, raw_start)

    low_positions = np.nonzero(mask_values <= start_buffer)[0]
    if low_positions.size:
        candidate = low_positions[-1] + 1
        if candidate < indices.size:
            entry_idx = int(indices[candidate])

    high_positions = np.nonzero(mask_values >= end_threshold)[0]
    if high_positions.size:
        for pos in high_positions:
            if np.any(mask_values[pos:] < end_threshold):
                continue
            exit_idx = int(indices[pos])
            break
        else:
            exit_idx = int(indices[high_positions[0]])

    if exit_idx < entry_idx:
        entry_idx, exit_idx = exit_idx, entry_idx

    return entry_idx, exit_idx


_LOG = logging.getLogger(__name__)


def match_activity_to_segment(
    runner: Runner,
    activity_id: int,
    segment_id: int,
    tolerances: Optional[Tolerances] = None,
) -> MatchResult:
    """Run the full segment matching pipeline for a given runner and activity."""

    tolerances = tolerances or Tolerances(
        start_tolerance_m=MATCHING_START_TOLERANCE_M,
        frechet_tolerance_m=MATCHING_FRECHET_TOLERANCE_M,
        coverage_threshold=MATCHING_COVERAGE_THRESHOLD,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )
    diagnostics: Dict[str, object] = {
        "segment_id": segment_id,
        "activity_id": activity_id,
    }

    cache_hit = False
    try:
        prepared_segment, cache_hit = _get_prepared_segment(
            segment_id,
            tolerances,
            runner,
        )
        prepared_activity = _prepare_activity(
            runner,
            activity_id,
            prepared_segment,
            tolerances,
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics["failure_reason"] = "preprocessing_failed"
        diagnostics["preprocessing_error"] = str(exc)
        _log_match_outcome(segment_id, activity_id, cache_hit, False, diagnostics)
        raise
    diagnostics["preprocessing"] = {
        "segment": {
            "metric_points": int(prepared_segment.metric_points.shape[0]),
            "simplified_points": int(prepared_segment.simplified_points.shape[0]),
            "resampled_points": int(prepared_segment.resampled_points.shape[0]),
            "effective_simplification_m": prepared_segment.simplification_tolerance_m,
            "effective_resample_m": prepared_segment.resample_interval_m,
            "simplification_capped": prepared_segment.simplification_capped,
            "resample_capped": prepared_segment.resample_capped,
        },
        "activity": {
            "metric_points": int(prepared_activity.metric_points.shape[0]),
            "simplified_points": int(prepared_activity.simplified_points.shape[0]),
            "resampled_points": int(prepared_activity.resampled_points.shape[0]),
            "effective_simplification_m": prepared_activity.simplification_tolerance_m,
            "effective_resample_m": prepared_activity.resample_interval_m,
            "simplification_capped": prepared_activity.simplification_capped,
            "resample_capped": prepared_activity.resample_capped,
        },
    }

    (
        coverage,
        coverage_diag,
        timing_bounds,
        timing_indices,
    ) = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=MATCHING_MAX_OFFSET_M,
        start_tolerance_m=tolerances.start_tolerance_m,
    )
    diagnostics["coverage"] = coverage_diag
    if (
        coverage.max_offset_m is not None
        and coverage.max_offset_m > MATCHING_MAX_OFFSET_M
    ):
        diagnostics["failure_reason"] = "coverage_offset_exceeded"
        _log_match_outcome(segment_id, activity_id, cache_hit, False, diagnostics)
        return MatchResult(
            False,
            coverage_ratio=coverage.coverage_ratio,
            diagnostics=diagnostics,
        )
    if (
        coverage.coverage_ratio < tolerances.coverage_threshold
        or coverage.coverage_bounds is None
    ):
        diagnostics["failure_reason"] = "coverage_threshold_not_met"
        _log_match_outcome(segment_id, activity_id, cache_hit, False, diagnostics)
        return MatchResult(
            False, coverage_ratio=coverage.coverage_ratio, diagnostics=diagnostics
        )

    direction = check_direction(
        prepared_activity.metric_points,
        prepared_segment.metric_points,
        start_tolerance_m=tolerances.start_tolerance_m,
        projections=coverage.projections,
        coverage_bounds=coverage.coverage_bounds,
    )
    diagnostics["direction"] = {
        "matches": direction.matches_direction,
        "max_start_distance_m": direction.max_start_distance_m,
        "direction_score": direction.direction_score,
    }
    if not direction.matches_direction:
        diagnostics["failure_reason"] = "direction_check_failed"
        _log_match_outcome(segment_id, activity_id, cache_hit, False, diagnostics)
        return MatchResult(
            False,
            coverage_ratio=coverage.coverage_ratio,
            diagnostics=diagnostics,
        )

    (
        matched,
        frechet_distance,
        _dtw_distance,
        similarity_threshold_m,
        similarity_diag,
    ) = _evaluate_similarity(
        prepared_activity,
        prepared_segment,
        coverage,
        start_tolerance_m=tolerances.start_tolerance_m,
        max_offset_threshold=MATCHING_MAX_OFFSET_M,
        base_threshold_m=tolerances.frechet_tolerance_m,
    )
    diagnostics.update(similarity_diag)

    if not matched:
        diagnostics.setdefault("failure_reason", "similarity_threshold_not_met")
        _log_match_outcome(segment_id, activity_id, cache_hit, False, diagnostics)
        return MatchResult(
            False,
            score=frechet_distance,
            max_deviation_m=frechet_distance,
            coverage_ratio=coverage.coverage_ratio,
            diagnostics=diagnostics,
        )

    elapsed_time = None
    timing_diag: Dict[str, object] = {}
    try:
        timing_range = timing_bounds or coverage.coverage_bounds
        if timing_range is None:
            raise ValueError("Missing coverage bounds for timing estimate")
        estimate = estimate_segment_time(
            prepared_activity,
            prepared_segment,
            timing_range,
            projections=coverage.projections,
            sample_indices=timing_indices,
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics["timing_error"] = str(exc)
        _LOG.debug("Timing estimation failed", exc_info=True)
        estimate = SegmentTimingEstimate(0.0, None, None, None, None)
    elapsed_time = estimate.elapsed_time_s
    timing_diag = {
        "elapsed_time_s": estimate.elapsed_time_s,
        "entry_index": estimate.entry_index,
        "exit_index": estimate.exit_index,
        "entry_time_s": estimate.entry_time_s,
        "exit_time_s": estimate.exit_time_s,
    }
    diagnostics["timing"] = timing_diag

    similarity_method = (
        "frechet" if frechet_distance <= similarity_threshold_m else "dtw"
    )
    diagnostics["similarity_method"] = similarity_method

    _log_match_outcome(segment_id, activity_id, cache_hit, True, diagnostics)
    return MatchResult(
        True,
        score=frechet_distance,
        max_deviation_m=frechet_distance,
        coverage_ratio=coverage.coverage_ratio,
        elapsed_time_s=elapsed_time,
        diagnostics=diagnostics,
    )


def _get_prepared_segment(
    segment_id: int,
    tolerances: Tolerances,
    runner: Runner,
) -> Tuple[PreparedSegmentGeometry, bool]:
    """Return prepared segment geometry, computing and caching if necessary."""

    cache_key = (
        segment_id,
        tolerances.simplification_tolerance_m,
        tolerances.resample_interval_m,
    )
    cached = SEGMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached.prepared, True

    segment_geometry = fetch_segment_geometry(runner, segment_id)
    prepared_segment = prepare_geometry(
        segment_geometry,
        simplification_tolerance_m=tolerances.simplification_tolerance_m,
        resample_interval_m=tolerances.resample_interval_m,
    )
    SEGMENT_CACHE.set(cache_key, build_cached_segment(prepared_segment))
    return prepared_segment, False


def _prepare_activity(
    runner: Runner,
    activity_id: int,
    prepared_segment: PreparedSegmentGeometry,
    tolerances: Tolerances,
) -> PreparedActivityTrack:
    """Fetch and preprocess an activity using the segment's transformer."""

    activity_track = fetch_activity_stream(runner, activity_id)
    return prepare_activity(
        activity_track,
        transformer=prepared_segment.transformer,
        simplification_tolerance_m=tolerances.simplification_tolerance_m,
        resample_interval_m=tolerances.resample_interval_m,
    )


def _dtw_window(activity_len: int, segment_len: int) -> int:
    """Return a Sakoe-Chiba style window for DTW fallback."""

    max_len = max(activity_len, segment_len)
    if max_len == 0:
        return 1
    return max(1, int(max_len * 0.1))


def _log_match_outcome(
    segment_id: int,
    activity_id: int,
    cache_hit: bool,
    matched: bool,
    diagnostics: Mapping[str, Any],
) -> None:
    """Emit structured logging for matcher diagnostics."""

    preprocessing = diagnostics.get("preprocessing")
    segment_stats: Dict[str, Any] = {}
    activity_stats: Dict[str, Any] = {}
    if isinstance(preprocessing, Mapping):
        raw_segment = preprocessing.get("segment")
        raw_activity = preprocessing.get("activity")
        if isinstance(raw_segment, Mapping):
            segment_stats = dict(raw_segment)
        if isinstance(raw_activity, Mapping):
            activity_stats = dict(raw_activity)

    coverage = diagnostics.get("coverage")
    coverage_ratio = None
    if isinstance(coverage, Mapping):
        coverage_ratio = coverage.get("ratio")

    timing = diagnostics.get("timing")
    elapsed = None
    if isinstance(timing, Mapping):
        elapsed = timing.get("elapsed_time_s")

    _LOG.info(
        (
            "matcher outcome segment=%s activity=%s matched=%s cache_hit=%s "
            "frechet_m=%s dtw_m=%s coverage_ratio=%s elapsed_s=%s method=%s "
            "failure=%s seg_resampled=%s act_resampled=%s"
        ),
        segment_id,
        activity_id,
        matched,
        cache_hit,
        diagnostics.get("frechet_distance_m"),
        diagnostics.get("dtw_distance_m"),
        coverage_ratio,
        elapsed,
        diagnostics.get("similarity_method"),
        diagnostics.get("failure_reason"),
        segment_stats.get("resampled_points"),
        activity_stats.get("resampled_points"),
    )


def _trim_activity_resampled(
    activity_points: np.ndarray,
    segment_points: np.ndarray,
    coverage_bounds: Optional[Tuple[float, float]],
    start_tolerance_m: float,
    max_offset_threshold: float,
) -> np.ndarray:
    """Return the portion of the activity resampled points covering the segment."""

    coverage = compute_coverage(activity_points, segment_points)
    trimmed = _extract_offset_window(
        activity_points,
        getattr(coverage, "offsets", None),
        max_offset_threshold,
    )
    if trimmed is not None:
        return trimmed

    projections = coverage.projections
    if coverage_bounds is None or projections is None:
        return activity_points
    if projections.shape[0] != activity_points.shape[0]:
        return activity_points

    start_bound, end_bound = coverage_bounds
    margin = max(start_tolerance_m, 5.0)
    lower = max(start_bound - margin, 0.0)
    upper = end_bound + margin
    mask = (projections >= lower) & (projections <= upper)
    indices = np.nonzero(mask)[0]
    if indices.size < 2:
        return activity_points
    return activity_points[indices[0] : indices[-1] + 1]


def _extract_offset_window(
    activity_points: np.ndarray,
    offsets: Optional[np.ndarray],
    max_offset_threshold: float,
) -> Optional[np.ndarray]:
    """Return a filtered slice using offset thresholds, or ``None`` if unavailable."""

    if offsets is None or offsets.shape[0] != activity_points.shape[0]:
        return None

    finite = np.isfinite(offsets)
    if np.count_nonzero(finite) < 2:
        return None

    base_mask = finite & (offsets <= max_offset_threshold)
    if np.count_nonzero(base_mask) < 2:
        relaxed = max(max_offset_threshold * 1.25, max_offset_threshold + 25.0)
        base_mask = finite & (offsets <= relaxed)
        if np.count_nonzero(base_mask) < 2:
            return None

    indices = np.nonzero(base_mask)[0]
    if indices.size < 2:
        return None
    start_idx, end_idx = int(indices[0]), int(indices[-1])

    window = activity_points[start_idx : end_idx + 1]
    window_offsets = offsets[start_idx : end_idx + 1]
    tight_threshold = max(max_offset_threshold * 0.5, 25.0)
    tight_mask = window_offsets <= tight_threshold
    if np.count_nonzero(tight_mask) >= 2:
        tight_mask[0] = True
        tight_mask[-1] = True
        window = window[tight_mask]

    if window.shape[0] < 2:
        return None
    return window


def _slice_segment_to_bounds(
    segment_points: np.ndarray,
    coverage_bounds: Optional[Tuple[float, float]],
    start_tolerance_m: float,
) -> np.ndarray:
    """Return segment points cropped to the coverage window with optional margin."""

    if coverage_bounds is None or segment_points.shape[0] < 2:
        return segment_points

    distances = _cumulative_distances_np(segment_points)
    span_end = float(distances[-1]) if distances.size else 0.0
    margin = max(start_tolerance_m, 5.0)
    lower = max(coverage_bounds[0] - margin, 0.0)
    upper = min(coverage_bounds[1] + margin, span_end)
    mask = (distances >= lower) & (distances <= upper)
    indices = np.nonzero(mask)[0]
    if indices.size < 2:
        return segment_points
    return segment_points[indices[0] : indices[-1] + 1]


def _cumulative_distances_np(points: np.ndarray) -> np.ndarray:
    """Return cumulative distances along a polyline represented as an array."""

    if points.shape[0] == 0:
        return np.zeros(1, dtype=float)
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths)))


def _polyline_length(points: np.ndarray) -> float:
    """Return the cumulative polyline length in meters."""

    if points.shape[0] < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    return float(np.sum(segment_lengths))


def _refine_coverage_window(
    coverage: "CoverageResult",
    segment_points: np.ndarray,
    *,
    max_offset_threshold: float,
    start_tolerance_m: float,
) -> Optional[_RefinedCoverage]:
    """Derive an overlap window by filtering projections with acceptable offsets."""

    projections = coverage.projections
    offsets = getattr(coverage, "offsets", None)
    bounds = coverage.coverage_bounds
    if (
        projections is None
        or offsets is None
        or bounds is None
        or projections.shape[0] != offsets.shape[0]
    ):
        return None

    finite = np.isfinite(offsets)
    if not np.any(finite):
        return None

    strict_mask = finite & (offsets <= max_offset_threshold)
    if np.count_nonzero(strict_mask) < 2:
        relaxed_threshold = max(
            max_offset_threshold * 1.25, max_offset_threshold + 25.0
        )
        strict_mask = finite & (offsets <= relaxed_threshold)
        if np.count_nonzero(strict_mask) < 2:
            return None

    indices = np.nonzero(strict_mask)[0]

    selected_proj = projections[strict_mask]
    selected_offsets = offsets[strict_mask]
    if selected_proj.size == 0:
        return None

    total_length = _polyline_length(segment_points)
    if total_length <= 0.0:
        return None

    raw_start = float(np.min(selected_proj))
    raw_end = float(np.max(selected_proj))
    raw_start = max(0.0, min(raw_start, total_length))
    raw_end = max(0.0, min(raw_end, total_length))
    if raw_end <= raw_start:
        return None

    entry_idx, exit_idx = _select_timing_indices(
        projections,
        indices,
        raw_start,
        raw_end,
        start_tolerance_m,
    )

    ratio = float(min(max((raw_end - raw_start) / total_length, 0.0), 1.0))
    max_offset = float(np.max(selected_offsets))

    margin = max(start_tolerance_m, 30.0)
    expanded_lower = max(raw_start - margin, 0.0)
    expanded_upper = min(raw_end + margin, total_length)
    expanded_bounds = (expanded_lower, expanded_upper)
    raw_bounds = (raw_start, raw_end)
    return _RefinedCoverage(
        raw_bounds,
        expanded_bounds,
        ratio,
        max_offset,
        entry_idx,
        exit_idx,
    )


def _compute_coverage_diagnostics(
    prepared_activity: PreparedActivityTrack,
    prepared_segment: PreparedSegmentGeometry,
    *,
    max_offset_threshold: float,
    start_tolerance_m: float,
) -> Tuple[
    CoverageResult,
    Dict[str, object],
    Optional[Tuple[float, float]],
    Optional[Tuple[int, int]],
]:
    """Evaluate coverage metrics and associated diagnostics for an activity."""

    coverage = compute_coverage(
        prepared_activity.metric_points, prepared_segment.metric_points
    )
    raw_bounds = coverage.coverage_bounds
    raw_max_offset = coverage.max_offset_m
    refined = _refine_coverage_window(
        coverage,
        prepared_segment.metric_points,
        max_offset_threshold=max_offset_threshold,
        start_tolerance_m=start_tolerance_m,
    )
    timing_bounds: Optional[Tuple[float, float]] = raw_bounds
    timing_indices: Optional[Tuple[int, int]] = None
    if refined is not None:
        coverage.coverage_bounds = refined.expanded_bounds
        coverage.coverage_ratio = refined.ratio
        coverage.max_offset_m = refined.max_offset
        timing_indices = (refined.entry_index, refined.exit_index)
    coverage_diag: Dict[str, object] = {
        "ratio": coverage.coverage_ratio,
        "bounds_m": coverage.coverage_bounds,
        "max_offset_m": coverage.max_offset_m,
    }
    if refined is not None:
        coverage_diag["raw_bounds_m"] = refined.raw_bounds
        if raw_max_offset is not None:
            coverage_diag["raw_max_offset_m"] = raw_max_offset
    else:
        if raw_bounds is not None:
            coverage_diag["raw_bounds_m"] = raw_bounds
        if raw_max_offset is not None:
            coverage_diag["raw_max_offset_m"] = raw_max_offset
    if timing_indices is not None:
        coverage_diag["timing_indices"] = timing_indices
    return coverage, coverage_diag, timing_bounds, timing_indices


def _evaluate_similarity(
    prepared_activity: PreparedActivityTrack,
    prepared_segment: PreparedSegmentGeometry,
    coverage: CoverageResult,
    *,
    start_tolerance_m: float,
    max_offset_threshold: float,
    base_threshold_m: float,
) -> Tuple[bool, float, Optional[float], float, Dict[str, object]]:
    """Compute similarity metrics between an activity and segment geometry."""

    segment_resampled = prepared_segment.resampled_points
    segment_window = _slice_segment_to_bounds(
        segment_resampled,
        coverage.coverage_bounds,
        start_tolerance_m,
    )
    trimmed_activity = _trim_activity_resampled(
        prepared_activity.resampled_points,
        segment_window,
        coverage.coverage_bounds,
        start_tolerance_m,
        max_offset_threshold,
    )

    trimmed_max_offset: Optional[float] = None
    if trimmed_activity.shape[0] >= 2 and segment_window.shape[0] >= 2:
        trimmed_stats = compute_coverage(trimmed_activity, segment_window)
        trimmed_max_offset = trimmed_stats.max_offset_m

    similarity_threshold = base_threshold_m
    if coverage.max_offset_m is not None and np.isfinite(coverage.max_offset_m):
        similarity_threshold = max(
            similarity_threshold,
            min(coverage.max_offset_m, max_offset_threshold) * 0.85,
        )
    if trimmed_max_offset is not None and np.isfinite(trimmed_max_offset):
        similarity_threshold = max(
            similarity_threshold,
            min(trimmed_max_offset, max_offset_threshold) * 1.2,
        )

    frechet_distance = discrete_frechet_distance(trimmed_activity, segment_window)
    matched = frechet_distance <= similarity_threshold
    dtw_distance: Optional[float] = None
    if not matched:
        window = _dtw_window(len(trimmed_activity), len(segment_window))
        dtw_distance = windowed_dtw(
            trimmed_activity,
            segment_window,
            window_size=window,
        )
        if np.isfinite(dtw_distance):
            matched = dtw_distance <= similarity_threshold

    override_by_offset = (
        not matched
        and trimmed_max_offset is not None
        and np.isfinite(trimmed_max_offset)
        and trimmed_max_offset <= max_offset_threshold
        and coverage.coverage_ratio >= 0.98
    )
    if override_by_offset:
        matched = True

    diag: Dict[str, object] = {
        "frechet_distance_m": frechet_distance,
        "similarity_threshold_m": similarity_threshold,
        "similarity_window": {
            "trimmed_points": int(trimmed_activity.shape[0]),
            "trimmed_max_offset_m": trimmed_max_offset,
        },
    }
    if dtw_distance is not None:
        diag["dtw_distance_m"] = dtw_distance
    if override_by_offset:
        diag["similarity_override"] = "offset_within_bounds"

    return matched, frechet_distance, dtw_distance, similarity_threshold, diag


__all__ = ["match_activity_to_segment", "MatchResult", "Tolerances"]
