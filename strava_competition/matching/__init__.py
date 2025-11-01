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


def _unwrap_monotonic(mask_values: np.ndarray, total_length: float) -> np.ndarray:
    """Return projections adjusted to follow a non-decreasing progression."""

    if mask_values.size == 0:
        return mask_values.astype(float, copy=True)

    unwrapped = np.empty_like(mask_values, dtype=float)
    unwrapped[0] = float(mask_values[0])
    if not np.isfinite(total_length) or total_length <= 0.0:
        for idx in range(1, mask_values.size):
            raw = float(mask_values[idx])
            unwrapped[idx] = max(unwrapped[idx - 1], raw)
        return unwrapped

    half_length = total_length * 0.5
    prev = unwrapped[0]
    for idx in range(1, mask_values.size):
        raw = float(mask_values[idx])
        prev_mod = prev % total_length
        delta = raw - prev_mod
        if delta > half_length:
            delta -= total_length
        elif delta < -half_length:
            delta += total_length
        candidate = prev + delta
        if candidate < prev:
            candidate += total_length
        unwrapped[idx] = candidate
        prev = candidate
    return unwrapped


def _best_chunk_slice(
    projections: np.ndarray,
    indices: np.ndarray,
    total_length: float,
) -> tuple[np.ndarray, float]:
    """Return a slice whose projections increase monotonically."""

    if indices.size < 2:
        empty = np.empty(0, dtype=int)
        return empty, 0.0

    mask_values = np.asarray(projections[indices], dtype=float)
    if mask_values.size < 2:
        empty = np.empty(0, dtype=int)
        return empty, 0.0

    unwrapped = _unwrap_monotonic(mask_values, total_length)
    entry_offset = int(np.argmin(unwrapped))
    exit_offset = int(np.argmax(unwrapped))
    if exit_offset <= entry_offset:
        empty = np.empty(0, dtype=int)
        return empty, 0.0

    start = entry_offset
    stop = exit_offset + 1
    trimmed_indices = indices[start:stop]
    trimmed_unwrapped = unwrapped[start:stop]
    if trimmed_indices.size < 2 or trimmed_unwrapped.size < 2:
        empty = np.empty(0, dtype=int)
        return empty, 0.0

    normalised = trimmed_unwrapped - float(trimmed_unwrapped[0])
    span_value = float(normalised[-1])
    if span_value <= 0.0:
        empty = np.empty(0, dtype=int)
        return empty, 0.0

    return trimmed_indices, span_value


def _projection_bounds(
    projections: np.ndarray,
    indices: np.ndarray,
) -> Optional[tuple[float, float, np.ndarray]]:
    """Return forward projection bounds for the refined chunk."""

    if indices.size < 2:
        return None

    subset = np.asarray(projections[indices], dtype=float)
    finite = subset[np.isfinite(subset)]
    if finite.size < 2:
        return None

    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if upper <= lower:
        return None
    return lower, upper, subset


def _max_offset_value(offsets: np.ndarray) -> float:
    """Return the maximum finite offset within the refined chunk."""

    if offsets.size == 0:
        return 0.0
    finite = offsets[np.isfinite(offsets)]
    if finite.size:
        return float(np.max(finite))
    return float(np.max(offsets))


def _plane_crossing_indices(
    values: np.ndarray, min_span_m: float, *, prefer_last: bool = False
) -> tuple[bool, Optional[int], Optional[int]]:
    """Return crossing status and bracketing sample indices for a plane."""

    if values.size < 2:
        return False, None, None

    classifications = np.zeros(values.shape[0], dtype=int)
    classifications[values >= min_span_m] = 1
    classifications[values <= -min_span_m] = -1
    signed_indices = np.nonzero(classifications != 0)[0]
    if signed_indices.size < 2:
        return False, None, None

    candidates: list[tuple[int, int]] = []
    prev_idx = int(signed_indices[0])
    prev_state = int(classifications[prev_idx])
    for idx in map(int, signed_indices[1:]):
        curr_state = int(classifications[idx])
        if curr_state == prev_state:
            prev_idx = idx
            continue
        # The change in sign brackets a crossing between prev_idx and idx.
        before_idx = min(prev_idx, idx)
        after_idx = max(prev_idx, idx)
        candidates.append((before_idx, after_idx))
        prev_idx = idx
        prev_state = curr_state

    if not candidates:
        return False, None, None

    before_idx, after_idx = candidates[-1] if prefer_last else candidates[0]
    return True, before_idx, after_idx


def _has_plane_crossing(values: np.ndarray, min_span_m: float) -> bool:
    crossed, _, _ = _plane_crossing_indices(values, min_span_m)
    return crossed


def _clamp_timing_indices(
    entry_idx: int,
    exit_idx: int,
    start_entry_idx: Optional[int],
    end_exit_idx: Optional[int],
) -> tuple[int, int]:
    """Adjust timing indices so they respect the detected gate crossings."""

    adjusted_entry = entry_idx
    adjusted_exit = exit_idx
    if start_entry_idx is not None:
        adjusted_entry = max(adjusted_entry, start_entry_idx)
    if end_exit_idx is not None:
        adjusted_exit = min(adjusted_exit, end_exit_idx)
    if adjusted_exit > adjusted_entry:
        return adjusted_entry, adjusted_exit
    return entry_idx, exit_idx


def _gate_crossings(
    activity_points: np.ndarray,
    segment_points: np.ndarray,
    start_tolerance_m: float,
) -> tuple[bool, bool, Optional[int], Optional[int]]:
    """Determine whether the activity crosses the synthetic start and finish planes."""

    required_span = min(3.0, max(0.5, start_tolerance_m * 0.02))
    crosses_start = False
    crosses_end = False
    start_entry_idx: Optional[int] = None
    end_exit_idx: Optional[int] = None
    if activity_points.shape[0] < 2 or segment_points.shape[0] < 2:
        return crosses_start, crosses_end, start_entry_idx, end_exit_idx

    start_vec = segment_points[1] - segment_points[0]
    start_norm = np.linalg.norm(start_vec)
    if start_norm > 0.0:
        start_unit = start_vec / start_norm
        start_values = (activity_points - segment_points[0]) @ start_unit
        crossed, before_idx, after_idx = _plane_crossing_indices(
            start_values, required_span
        )
        crosses_start = crossed
        if crossed and before_idx is not None and after_idx is not None:
            start_entry_idx = max(after_idx, before_idx + 1)

    # Reverse the finish direction so pre-finish samples project negative.
    end_vec = segment_points[-2] - segment_points[-1]
    end_norm = np.linalg.norm(end_vec)
    if end_norm > 0.0:
        end_unit = end_vec / end_norm
        end_values = (activity_points - segment_points[-1]) @ end_unit
        crossed, before_idx, after_idx = _plane_crossing_indices(
            end_values, required_span, prefer_last=True
        )
        crosses_end = crossed
        if crossed and before_idx is not None and after_idx is not None:
            candidate = after_idx - 1 if after_idx > 0 else before_idx
            end_exit_idx = max(before_idx, candidate)

    return crosses_start, crosses_end, start_entry_idx, end_exit_idx


def _prepare_refined_inputs(
    coverage: "CoverageResult",
    segment_points: np.ndarray,
    max_offset_threshold: float,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Return projections, offsets, indices and length for refinement."""

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

    mask = finite & (offsets <= max_offset_threshold)
    if np.count_nonzero(mask) < 2:
        relaxed = max(max_offset_threshold * 1.25, max_offset_threshold + 25.0)
        mask = finite & (offsets <= relaxed)
        if np.count_nonzero(mask) < 2:
            return None

    total_length = _polyline_length(segment_points)
    if total_length <= 0.0:
        return None

    indices = np.nonzero(mask)[0]
    if indices.size < 2:
        return None

    return projections, offsets, indices, total_length


def _select_timing_indices(
    chunk_indices: np.ndarray,
    chunk_projections: np.ndarray,
) -> tuple[int, int]:
    """Return indices of the earliest and latest on-route samples."""

    if chunk_indices.size < 2:
        return 0, 0

    projections = np.asarray(chunk_projections, dtype=float)
    if projections.shape[0] != chunk_indices.shape[0]:
        return 0, 0

    finite_mask = np.isfinite(projections)
    if np.count_nonzero(finite_mask) < 2:
        return 0, 0

    finite_indices = np.nonzero(finite_mask)[0]
    finite_values = projections[finite_mask]
    entry_pos = int(finite_indices[np.argmin(finite_values)])
    exit_pos = int(finite_indices[np.argmax(finite_values)])

    entry_idx = int(chunk_indices[entry_pos])
    exit_idx = int(chunk_indices[exit_pos])
    if exit_idx <= entry_idx:
        # When projections wrap around the segment start, fall back to the
        # chronological slice to keep timing indices strictly increasing.
        first_idx = int(chunk_indices[0])
        last_idx = int(chunk_indices[-1])
        if last_idx > first_idx:
            return first_idx, last_idx
        return 0, 0

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

    crosses_start = coverage_diag.get("crosses_start")
    crosses_end = coverage_diag.get("crosses_end")
    if crosses_start is not True or crosses_end is not True:
        diagnostics["failure_reason"] = "segment_gate_not_crossed"
        _log_match_outcome(segment_id, activity_id, cache_hit, False, diagnostics)
        return MatchResult(
            False,
            coverage_ratio=coverage.coverage_ratio,
            diagnostics=diagnostics,
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

    prepared = _prepare_refined_inputs(
        coverage,
        segment_points,
        max_offset_threshold,
    )
    if prepared is None:
        return None
    projections, offsets, indices, total_length = prepared

    chunk_indices, span_value = _best_chunk_slice(projections, indices, total_length)
    if chunk_indices.size == 0 or span_value <= 0.0:
        return None

    bounds = _projection_bounds(projections, chunk_indices)
    if bounds is None:
        return None
    raw_start, raw_end, chunk_proj = bounds

    sample_count = chunk_indices.size
    if sample_count < 3:
        return None

    guard_span = max(raw_end - raw_start, 0.0)
    segment_stride = total_length / max(segment_points.shape[0] - 1, 1)
    max_stride = max(segment_stride * 10.0, 25.0)
    mean_stride = guard_span / max(sample_count - 1, 1)
    # Reject slices that stay too sparse after removing large offsets.
    if mean_stride > max_stride:
        return None

    entry_idx, exit_idx = _select_timing_indices(chunk_indices, chunk_proj)
    if entry_idx == 0 and exit_idx == 0:
        return None

    ratio = float(min(max(span_value / total_length, 0.0), 1.0))
    max_offset = _max_offset_value(offsets[chunk_indices])

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
    crosses_start, crosses_end, start_entry_idx, end_exit_idx = _gate_crossings(
        prepared_activity.metric_points,
        prepared_segment.metric_points,
        start_tolerance_m,
    )
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
        clamped_entry, clamped_exit = _clamp_timing_indices(
            refined.entry_index,
            refined.exit_index,
            start_entry_idx,
            end_exit_idx,
        )
        timing_indices = (clamped_entry, clamped_exit)
    coverage_diag: Dict[str, object] = {
        "ratio": coverage.coverage_ratio,
        "bounds_m": coverage.coverage_bounds,
        "max_offset_m": coverage.max_offset_m,
        "crosses_start": crosses_start,
        "crosses_end": crosses_end,
    }
    if start_entry_idx is not None:
        coverage_diag["start_entry_index"] = start_entry_idx
    if end_exit_idx is not None:
        coverage_diag["end_exit_index"] = end_exit_idx
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
        adaptive_cap = max_offset_threshold * 5.0
        # Widen tolerance for well-aligned tracks that still diverge along the route.
        similarity_threshold = max(
            similarity_threshold,
            min(trimmed_max_offset * 5.0, adaptive_cap),
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

    return matched, frechet_distance, dtw_distance, similarity_threshold, diag


__all__ = ["match_activity_to_segment", "MatchResult", "Tolerances"]
