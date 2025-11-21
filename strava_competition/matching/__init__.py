"""Public entry points for the GPS-based segment matching package."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

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
from .timing import (
    GateCrossingHint,
    GateTimingHints,
    SegmentTimingEstimate,
    estimate_segment_time,
)
from .validation import CoverageResult, check_direction, compute_coverage


# Fallback tuning for gate-clipped tracks that otherwise fail similarity thresholds.
_GATE_FALLBACK_MIN_POINTS = 200
_GATE_FALLBACK_MAX_OFFSET_M = 25.0
_GATE_FALLBACK_FRECHET_MULTIPLIER = 12.0


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


def _plane_crossing_candidates(
    values: np.ndarray,
    min_span_m: float,
) -> list[tuple[int, int]]:
    """Return all sample pairs that straddle a plane crossing."""

    if values.size < 2:
        return []

    classifications = np.zeros(values.shape[0], dtype=int)
    classifications[values >= min_span_m] = 1
    classifications[values <= -min_span_m] = -1
    signed_indices = np.nonzero(classifications != 0)[0]

    candidates: list[tuple[int, int]] = []
    if signed_indices.size >= 2:
        prev_idx = int(signed_indices[0])
        prev_state = int(classifications[prev_idx])
        for idx in map(int, signed_indices[1:]):
            curr_state = int(classifications[idx])
            if curr_state == prev_state:
                prev_idx = idx
                continue
            before_idx = min(prev_idx, idx)
            after_idx = max(prev_idx, idx)
            candidates.append((before_idx, after_idx))
            prev_idx = idx
            prev_state = curr_state

    fallback_pairs: list[tuple[int, int]] = []
    for idx in range(values.shape[0] - 1):
        lhs = float(values[idx])
        rhs = float(values[idx + 1])
        if not (np.isfinite(lhs) and np.isfinite(rhs)):
            continue
        if lhs == rhs:
            continue
        if lhs <= 0.0 <= rhs or lhs >= 0.0 >= rhs:
            fallback_pairs.append((idx, idx + 1))

    if candidates:
        # Ensure any near-plane options participate in downstream ranking.
        for pair in fallback_pairs:
            if pair not in candidates:
                candidates.append(pair)
        return candidates

    return fallback_pairs


def _plane_crossing_indices(
    values: np.ndarray, min_span_m: float, *, prefer_last: bool = False
) -> tuple[bool, Optional[int], Optional[int]]:
    """Return crossing status and bracketing sample indices for a plane."""

    candidates = _plane_crossing_candidates(values, min_span_m)
    if not candidates:
        return False, None, None

    before_idx, after_idx = candidates[-1] if prefer_last else candidates[0]
    return True, before_idx, after_idx


def _has_plane_crossing(values: np.ndarray, min_span_m: float) -> bool:
    crossed, _, _ = _plane_crossing_indices(values, min_span_m)
    return crossed


def _start_gate_values(
    activity_points: np.ndarray,
    segment_points: np.ndarray,
) -> Optional[np.ndarray]:
    """Return signed distances to the start gate plane for each activity sample."""

    if segment_points.shape[0] < 2:
        return None

    start_vec = segment_points[1] - segment_points[0]
    start_norm = np.linalg.norm(start_vec)
    if start_norm <= 0.0:
        return None

    start_unit = start_vec / start_norm
    return (activity_points - segment_points[0]) @ start_unit


def _start_orientation_matches(
    start_values: np.ndarray,
    before_idx: int,
    after_idx: int,
) -> bool:
    """Return True when the candidate crosses the plane in the forward direction."""

    before_val = float(start_values[before_idx])
    after_val = float(start_values[after_idx])
    return before_val <= 0.0 <= after_val


def _build_start_hint_from_candidate(
    start_values: np.ndarray,
    candidate: tuple[int, int],
    required_span: float,
) -> tuple[int, GateCrossingHint]:
    """Return entry index and hint derived from a start gate candidate pair."""

    before_idx, after_idx = candidate
    before_idx, after_idx = _refine_crossing_indices(
        start_values,
        before_idx,
        after_idx,
        required_span,
        prefer_start=True,
    )
    entry_idx = after_idx
    hint = GateCrossingHint(
        before_index=before_idx,
        after_index=after_idx,
        before_value=float(start_values[before_idx]),
        after_value=float(start_values[after_idx]),
    )
    return entry_idx, hint


def _select_start_crossing(
    values: np.ndarray,
    candidates: list[tuple[int, int]],
) -> Optional[tuple[int, int]]:
    """Return the earliest start-gate crossing with expected orientation."""

    for before_idx, after_idx in candidates:
        before_val = float(values[before_idx])
        after_val = float(values[after_idx])
        if before_val <= 0.0 <= after_val:
            return before_idx, after_idx
    return candidates[0] if candidates else None


def _select_end_crossing(
    values: np.ndarray,
    candidates: list[tuple[int, int]],
    offset: int,
) -> Optional[tuple[int, int]]:
    """Return the finish-gate crossing closest to the plane."""

    if not candidates:
        return None

    records: list[tuple[int, int, int, int, float, bool]] = []
    for before_local, after_local in candidates:
        global_before = before_local + offset
        global_after = after_local + offset
        before_val = float(values[global_before])
        after_val = float(values[global_after])
        cost = max(abs(before_val), abs(after_val))
        orientation_ok = before_val >= 0.0 and after_val <= 0.0
        records.append(
            (
                before_local,
                after_local,
                global_before,
                global_after,
                cost,
                orientation_ok,
            )
        )

    earliest = min(
        records,
        key=lambda item: (item[3], item[4]),
    )
    oriented_candidates = [item for item in records if item[5]]
    if not oriented_candidates:
        return earliest[0], earliest[1]

    best_oriented = min(
        oriented_candidates,
        key=lambda item: (
            item[4],
            item[3],
        ),
    )

    if earliest[5]:
        return earliest[0], earliest[1]

    # Allow an orientation-mismatch only when the best oriented option is
    # significantly later and much less aligned with the gate plane.
    oriented_gap = best_oriented[3] - earliest[3]
    min_gap = 120
    if oriented_gap > min_gap:
        oriented_cost = best_oriented[4]
        mismatch_limit = max(3.0, oriented_cost * 0.5)
        if earliest[4] <= mismatch_limit and oriented_cost >= 10.0:
            return earliest[0], earliest[1]

    return best_oriented[0], best_oriented[1]


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


def _refine_crossing_indices(
    values: np.ndarray,
    before_idx: int,
    after_idx: int,
    threshold: float,
    *,
    prefer_start: bool,
) -> tuple[int, int]:
    """Return adjusted crossing indices favouring samples near the gate plane."""

    if after_idx <= before_idx:
        return before_idx, after_idx

    best_pair: Optional[tuple[int, int]] = None
    best_abs = float("inf")

    for idx in range(before_idx, after_idx):
        lhs = float(values[idx])
        rhs = float(values[idx + 1])
        if abs(lhs) <= threshold or abs(rhs) <= threshold:
            return idx, idx + 1
        if lhs <= 0.0 <= rhs or lhs >= 0.0 >= rhs:
            return idx, idx + 1

        local_abs = min(abs(lhs), abs(rhs))
        if best_pair is None or local_abs < best_abs:
            best_pair = (idx, idx + 1)
            best_abs = local_abs

        if prefer_start and local_abs <= threshold:
            return idx, idx + 1

    if best_pair is not None:
        return best_pair

    return before_idx, after_idx


def _compute_start_hint(
    activity_points: np.ndarray,
    segment_points: np.ndarray,
    required_span: float,
    *,
    search_start_idx: Optional[int] = None,
) -> tuple[bool, Optional[int], Optional[GateCrossingHint]]:
    """Return start gate crossing status, entry index and hint."""

    start_values = _start_gate_values(activity_points, segment_points)
    if start_values is None:
        return False, None, None
    start_candidates = _plane_crossing_candidates(start_values, required_span)
    if search_start_idx is not None:
        start_candidates = [
            candidate
            for candidate in start_candidates
            if candidate[1] >= max(search_start_idx, 0)
        ]
    if not start_candidates:
        return False, None, None

    selected = _select_start_crossing(start_values, start_candidates)
    if selected is None:
        return False, None, None
    entry_idx, hint = _build_start_hint_from_candidate(
        start_values,
        selected,
        required_span,
    )
    return True, entry_idx, hint


def _segment_point_and_direction(
    segment_points: np.ndarray,
    distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a point and backward-facing direction at ``distance_m`` along the polyline."""

    if segment_points.shape[0] < 2:
        raise ValueError("segment_points must contain at least two samples")

    cumulative = _cumulative_distances_np(segment_points)
    total_length = float(cumulative[-1]) if cumulative.size else 0.0
    if total_length <= 0.0:
        return segment_points[-1], segment_points[-2] - segment_points[-1]

    clamped = float(np.clip(distance_m, 0.0, total_length))
    upper_idx = int(np.searchsorted(cumulative, clamped, side="right"))
    idx = max(0, min(upper_idx - 1, segment_points.shape[0] - 2))

    span = float(cumulative[idx + 1] - cumulative[idx])
    seg_start = segment_points[idx]
    seg_end = segment_points[idx + 1]
    if span <= 0.0:
        point = seg_start.astype(float, copy=True)
    else:
        ratio = (clamped - float(cumulative[idx])) / span
        point = seg_start + (seg_end - seg_start) * ratio

    direction = seg_start - seg_end
    if np.linalg.norm(direction) <= 0.0:
        fallback_idx = max(0, min(idx - 1, segment_points.shape[0] - 2))
        direction = segment_points[fallback_idx] - segment_points[fallback_idx + 1]
    return point, direction


def _compute_end_hint(
    activity_points: np.ndarray,
    segment_points: np.ndarray,
    required_span: float,
    start_entry_idx: Optional[int],
    *,
    finish_point: Optional[np.ndarray] = None,
    finish_direction: Optional[np.ndarray] = None,
) -> tuple[bool, Optional[int], Optional[GateCrossingHint]]:
    """Return finish gate crossing status, exit index and hint."""

    if finish_point is None or finish_direction is None:
        finish_point = segment_points[-1]
        finish_direction = segment_points[-2] - segment_points[-1]

    end_norm = np.linalg.norm(finish_direction)
    if end_norm <= 0.0:
        return False, None, None

    end_unit = finish_direction / end_norm
    end_values = (activity_points - finish_point) @ end_unit
    finish_search_start = start_entry_idx if start_entry_idx is not None else 0
    sliced_end_values = end_values[finish_search_start:]
    if sliced_end_values.size < 2:
        return False, None, None

    end_candidates = _plane_crossing_candidates(
        sliced_end_values,
        required_span,
    )
    if not end_candidates:
        return False, None, None

    selected = _select_end_crossing(end_values, end_candidates, finish_search_start)
    if selected is None:
        return False, None, None

    before_idx = selected[0] + finish_search_start
    after_idx = selected[1] + finish_search_start
    before_idx, after_idx = _refine_crossing_indices(
        end_values,
        before_idx,
        after_idx,
        required_span,
        prefer_start=False,
    )
    exit_idx = before_idx
    hint = GateCrossingHint(
        before_index=before_idx,
        after_index=after_idx,
        before_value=float(end_values[before_idx]),
        after_value=float(end_values[after_idx]),
    )
    return True, exit_idx, hint


def _gate_crossings(
    activity_points: np.ndarray,
    segment_points: np.ndarray,
    start_tolerance_m: float,
    *,
    finish_distance_m: Optional[float] = None,
    timestamps: Optional[np.ndarray] = None,
    projections: Optional[np.ndarray] = None,
) -> tuple[bool, bool, Optional[int], Optional[int], GateTimingHints]:
    """Determine whether the activity crosses the synthetic start and finish planes."""

    required_span = min(3.0, max(0.5, start_tolerance_m * 0.02))
    if activity_points.shape[0] < 2 or segment_points.shape[0] < 2:
        return False, False, None, None, GateTimingHints()

    finish_point: Optional[np.ndarray] = None
    finish_direction: Optional[np.ndarray] = None
    if finish_distance_m is not None and np.isfinite(finish_distance_m):
        try:
            finish_point, finish_direction = _segment_point_and_direction(
                segment_points,
                float(finish_distance_m),
            )
        except Exception:  # noqa: BLE001 - fallback to default behaviour
            finish_point = None
            finish_direction = None

    start_values = _start_gate_values(activity_points, segment_points)
    if start_values is None:
        return False, False, None, None, GateTimingHints()

    start_candidates = _plane_crossing_candidates(start_values, required_span)
    selected_start = _select_start_crossing(start_values, start_candidates)
    if selected_start is None:
        return False, False, None, None, GateTimingHints()

    start_entry_idx, start_hint = _build_start_hint_from_candidate(
        start_values,
        selected_start,
        required_span,
    )
    crosses_start = True

    crosses_end, end_exit_idx, end_hint = _compute_end_hint(
        activity_points,
        segment_points,
        required_span,
        start_entry_idx,
        finish_point=finish_point,
        finish_direction=finish_direction,
    )

    gate_hints = GateTimingHints(start=start_hint, end=end_hint)

    if timestamps is not None and start_candidates:
        best_window = _select_fastest_gate_window(
            activity_points,
            segment_points,
            start_values,
            start_candidates,
            required_span,
            timestamps,
            projections,
            finish_point=finish_point,
            finish_direction=finish_direction,
            finish_distance_m=finish_distance_m,
        )
        if best_window is not None:
            (
                start_entry_idx,
                end_exit_idx,
                best_start_hint,
                best_end_hint,
            ) = best_window
            gate_hints = GateTimingHints(start=best_start_hint, end=best_end_hint)
            crosses_start = True
            crosses_end = True

    return (
        crosses_start,
        crosses_end,
        start_entry_idx,
        end_exit_idx,
        gate_hints,
    )


def _candidate_elapsed_seconds(
    timestamps: np.ndarray,
    start_hint: GateCrossingHint,
    end_hint: GateCrossingHint,
) -> Optional[float]:
    """Return approximate elapsed seconds between two gate hints."""

    if timestamps.size == 0:
        return None

    start_idx = int(min(max(start_hint.after_index, 0), timestamps.shape[0] - 1))
    end_idx = int(min(max(end_hint.before_index, 0), timestamps.shape[0] - 1))
    if end_idx <= start_idx:
        return None

    return float(timestamps[end_idx] - timestamps[start_idx])


def _candidate_span_ratio(
    projections: Optional[np.ndarray],
    entry_idx: int,
    exit_idx: int,
    finish_distance_m: Optional[float],
) -> Optional[float]:
    """Return the projection span ratio covered between two indices."""

    if projections is None or projections.size == 0:
        return None

    point_count = projections.shape[0]
    if point_count == 0:
        return None

    start = max(0, min(entry_idx, point_count - 1))
    stop = max(start, min(exit_idx, point_count - 1))
    if stop - start < 2:
        return None

    window = projections[start : stop + 1]
    finite = window[np.isfinite(window)]
    if finite.size < 2:
        return None

    span = float(np.max(finite) - np.min(finite))
    if span <= 0.0:
        return None

    target = finish_distance_m
    if target is None or not np.isfinite(target) or target <= 0.0:
        target = span
    if target <= 0.0:
        return None
    return min(max(span / target, 0.0), 1.5)


def _score_gate_candidate(
    activity_points: np.ndarray,
    segment_points: np.ndarray,
    required_span: float,
    timestamps: np.ndarray,
    projections: Optional[np.ndarray],
    finish_point: Optional[np.ndarray],
    finish_direction: Optional[np.ndarray],
    finish_distance_m: Optional[float],
    start_values: np.ndarray,
    candidate: tuple[int, int],
) -> Optional[tuple[int, float, float, int, int, GateCrossingHint, GateCrossingHint]]:
    """Return a scored record for a single gate candidate."""

    entry_idx, start_hint = _build_start_hint_from_candidate(
        start_values,
        candidate,
        required_span,
    )
    crosses_end, exit_idx, end_hint = _compute_end_hint(
        activity_points,
        segment_points,
        required_span,
        entry_idx,
        finish_point=finish_point,
        finish_direction=finish_direction,
    )
    if not crosses_end or exit_idx is None or end_hint is None:
        return None

    elapsed = _candidate_elapsed_seconds(timestamps, start_hint, end_hint)
    if elapsed is None or elapsed <= 0.0:
        return None

    span_ratio = _candidate_span_ratio(
        projections,
        entry_idx,
        exit_idx,
        finish_distance_m,
    )
    if span_ratio is not None and span_ratio < 0.65:
        return None

    orientation_penalty = (
        0
        if _start_orientation_matches(
            start_values,
            candidate[0],
            candidate[1],
        )
        else 1
    )
    span_score = -span_ratio if span_ratio is not None else 0.0
    return (
        orientation_penalty,
        elapsed,
        span_score,
        entry_idx,
        exit_idx,
        start_hint,
        end_hint,
    )


def _select_fastest_gate_window(
    activity_points: np.ndarray,
    segment_points: np.ndarray,
    start_values: np.ndarray,
    start_candidates: list[tuple[int, int]],
    required_span: float,
    timestamps: np.ndarray,
    projections: Optional[np.ndarray],
    *,
    finish_point: Optional[np.ndarray],
    finish_direction: Optional[np.ndarray],
    finish_distance_m: Optional[float],
) -> Optional[tuple[int, int, GateCrossingHint, GateCrossingHint]]:
    """Return the start/finish pair representing the fastest complete lap."""

    if timestamps.size < 2 or not start_candidates:
        return None

    scored: list[
        tuple[int, float, float, int, int, GateCrossingHint, GateCrossingHint]
    ] = []
    for candidate in start_candidates:
        record = _score_gate_candidate(
            activity_points,
            segment_points,
            required_span,
            timestamps,
            projections,
            finish_point,
            finish_direction,
            finish_distance_m,
            start_values,
            candidate,
        )
        if record is not None:
            scored.append(record)

    if not scored:
        return None

    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    best = scored[0]
    return best[3], best[4], best[5], best[6]


def _resolve_gate_context(
    activity_points: np.ndarray,
    full_activity_points: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return the gate clipping context and matching indices."""

    if full_activity_points is None or full_activity_points.shape[0] < 2:
        indices = np.arange(activity_points.shape[0], dtype=int)
        return activity_points, indices, False

    subset_indices = _map_subset_indices(activity_points, full_activity_points)
    if subset_indices is None or subset_indices.shape[0] != activity_points.shape[0]:
        indices = np.arange(activity_points.shape[0], dtype=int)
        return activity_points, indices, False

    return full_activity_points, subset_indices, True


def _prepare_refined_inputs(
    coverage: "CoverageResult",
    segment_points: np.ndarray,
    max_offset_threshold: float,
    gate_slice: Optional[Tuple[int, int]] = None,
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

    if gate_slice is not None:
        start_idx, end_idx = gate_slice
        gate_mask = np.zeros_like(mask)
        gate_mask[start_idx : end_idx + 1] = True
        mask &= gate_mask
        if np.count_nonzero(mask) < 2:
            return None

    total_length = _polyline_length(segment_points)
    if total_length <= 0.0:
        return None

    indices = np.nonzero(mask)[0]
    if indices.size < 2:
        return None

    return projections, offsets, indices, total_length


def _gate_slice_from_hints(
    hints: Optional[GateTimingHints],
    point_count: int,
) -> Optional[Tuple[int, int]]:
    """Return inclusive index bounds representing the gate-trimmed span."""

    if hints is None or point_count < 2:
        return None

    start_idx = 0
    end_idx = point_count - 1
    has_hint = False

    if hints.start is not None:
        start_idx = int(min(max(hints.start.after_index, 0), end_idx))
        has_hint = True
    if hints.end is not None:
        end_idx = int(max(start_idx, min(hints.end.before_index, end_idx)))
        has_hint = True

    if not has_hint or end_idx - start_idx < 1:
        return None
    return start_idx, end_idx


def _max_offset_within_slice(
    offsets: Optional[np.ndarray],
    gate_slice: Optional[Tuple[int, int]],
) -> Optional[float]:
    """Return the maximum offset restricted to an inclusive gate slice."""

    if offsets is None or gate_slice is None:
        return None

    start_idx, end_idx = gate_slice
    if start_idx < 0 or end_idx >= offsets.shape[0] or end_idx < start_idx:
        return None

    subset = offsets[start_idx : end_idx + 1]
    if subset.size == 0:
        return None
    return _max_offset_value(subset)


def _prune_gate_hints(
    hints: GateTimingHints,
    timestamps: np.ndarray,
    *,
    entry_idx: Optional[int],
    exit_idx: Optional[int],
    max_start_lead_s: float = 180.0,
    max_start_lag_s: float = 60.0,
    max_finish_lead_s: float = 180.0,
    max_finish_lag_s: float = 600.0,
) -> GateTimingHints:
    """Return gate hints with timestamps far from the refined window removed."""

    if timestamps.size == 0:
        return hints

    def _timestamp_or_none(index: int) -> Optional[float]:
        if index < 0 or index >= timestamps.shape[0]:
            return None
        return float(timestamps[index])

    def _prune_single_hint(
        hint: Optional[GateCrossingHint],
        target_time: Optional[float],
        index_getter: Callable[[GateCrossingHint], int],
        max_lead: float,
        max_lag: float,
    ) -> Optional[GateCrossingHint]:
        if hint is None or target_time is None:
            return hint
        hint_time = _timestamp_or_none(int(index_getter(hint)))
        if hint_time is None:
            return None
        delta = target_time - hint_time
        if delta > max_lead or delta < -max_lag:
            return None
        return hint

    start_time = _timestamp_or_none(int(entry_idx)) if entry_idx is not None else None
    end_time = _timestamp_or_none(int(exit_idx)) if exit_idx is not None else None

    start_hint = _prune_single_hint(
        hints.start,
        start_time,
        lambda hint: hint.after_index,
        max_start_lead_s,
        max_start_lag_s,
    )
    end_hint = _prune_single_hint(
        hints.end,
        end_time,
        lambda hint: hint.before_index,
        max_finish_lead_s,
        max_finish_lag_s,
    )

    return GateTimingHints(start=start_hint, end=end_hint)


def _restore_start_hint_from_refined_entry(
    hints: GateTimingHints,
    activity_points: np.ndarray,
    segment_points: np.ndarray,
    refined_entry_idx: Optional[int],
    required_span: float,
    current_start_idx: Optional[int],
) -> Tuple[GateTimingHints, Optional[int]]:
    """Recreate a start hint near the refined entry when the original hint was pruned."""

    if hints.start is not None or refined_entry_idx is None:
        return hints, current_start_idx

    search_idx = max(int(refined_entry_idx) - 2, 0)
    start_crossed, entry_idx, new_hint = _compute_start_hint(
        activity_points,
        segment_points,
        required_span,
        search_start_idx=search_idx,
    )
    if not start_crossed or entry_idx is None or new_hint is None:
        return hints, current_start_idx

    return GateTimingHints(start=new_hint, end=hints.end), entry_idx


def _apply_gate_indices(
    start_entry_idx: Optional[int],
    end_exit_idx: Optional[int],
    gate_hints: GateTimingHints,
    *,
    fallback_start: Optional[int] = None,
    fallback_end: Optional[int] = None,
) -> tuple[Optional[int], Optional[int]]:
    """Return gate indices adjusted with explicit hint indices when present."""

    if gate_hints.start is not None:
        start_entry_idx = gate_hints.start.after_index
    elif fallback_start is not None:
        start_entry_idx = fallback_start
    if gate_hints.end is not None:
        end_exit_idx = gate_hints.end.before_index
    elif fallback_end is not None:
        end_exit_idx = fallback_end
    return start_entry_idx, end_exit_idx


def _build_coverage_diag(
    coverage: CoverageResult,
    crosses_start: bool,
    crosses_end: bool,
    start_entry_idx: Optional[int],
    end_exit_idx: Optional[int],
    raw_bounds: Optional[Tuple[float, float]],
    raw_max_offset: Optional[float],
    refined: Optional[_RefinedCoverage],
    trimmed_max: Optional[float],
    gate_slice: Optional[Tuple[int, int]],
    gate_hints: GateTimingHints,
    timing_indices: Optional[Tuple[int, int]],
) -> Dict[str, object]:
    """Return a diagnostics mapping summarising coverage evaluation."""

    diag: Dict[str, object] = {
        "ratio": coverage.coverage_ratio,
        "bounds_m": coverage.coverage_bounds,
        "max_offset_m": coverage.max_offset_m,
        "crosses_start": crosses_start,
        "crosses_end": crosses_end,
    }
    if start_entry_idx is not None:
        diag["start_entry_index"] = start_entry_idx
    if end_exit_idx is not None:
        diag["end_exit_index"] = end_exit_idx
    if trimmed_max is not None:
        diag["gate_trimmed_max_offset_m"] = trimmed_max
    if gate_slice is not None:
        diag["gate_slice_indices"] = gate_slice
    if gate_hints.start is not None:
        diag["start_crossing_samples"] = (
            gate_hints.start.before_index,
            gate_hints.start.after_index,
        )
        diag["start_crossing_offsets_m"] = (
            gate_hints.start.before_value,
            gate_hints.start.after_value,
        )
    if gate_hints.end is not None:
        diag["end_crossing_samples"] = (
            gate_hints.end.before_index,
            gate_hints.end.after_index,
        )
        diag["end_crossing_offsets_m"] = (
            gate_hints.end.before_value,
            gate_hints.end.after_value,
        )
        diag["end_crossing_span_m"] = (
            gate_hints.end.after_value - gate_hints.end.before_value
        )
    if refined is not None:
        diag["raw_bounds_m"] = refined.raw_bounds
        if raw_max_offset is not None:
            diag["raw_max_offset_m"] = raw_max_offset
    else:
        if raw_bounds is not None:
            diag["raw_bounds_m"] = raw_bounds
        if raw_max_offset is not None:
            diag["raw_max_offset_m"] = raw_max_offset
    if timing_indices is not None:
        diag["timing_indices"] = timing_indices
    return diag


def _entry_position_near_start(
    projections: np.ndarray,
    finite_indices: np.ndarray,
    finite_values: np.ndarray,
    lower_bound: float,
    span: float,
) -> int:
    """Return the earliest finite position whose projection hugs the lower bound."""

    if finite_indices.size == 0:
        return 0

    entry_band = max(3.0, min(span * 0.005, 20.0))
    for pos in map(int, finite_indices):
        value = projections[pos]
        if not np.isfinite(value):
            continue
        if value - lower_bound <= entry_band:
            return pos
    # Fall back to the absolute minimum when no near-start projections exist.
    return int(finite_indices[np.argmin(finite_values)])


def _refined_exit_index(
    chunk_indices: np.ndarray,
    projections: np.ndarray,
    entry_pos: int,
    entry_idx: int,
    raw_end: float,
    span: float,
    initial_exit_idx: int,
) -> int:
    """Return an exit index aligned with the first projection that reaches the finish."""

    tolerance = max(2.0, span * 0.02)
    target = float(raw_end) - tolerance
    exit_idx = initial_exit_idx

    for pos in range(entry_pos, chunk_indices.shape[0]):
        value = projections[pos]
        if not np.isfinite(value):
            continue
        if value >= target:
            candidate_idx = int(chunk_indices[pos])
            if candidate_idx > entry_idx:
                return candidate_idx

    return exit_idx


def _select_timing_indices(
    chunk_indices: np.ndarray,
    chunk_projections: np.ndarray,
    raw_start: float,
    raw_end: float,
) -> tuple[int, int]:
    """Return entry index and first finish crossing matching the coverage span."""

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
    span = max(float(raw_end) - float(raw_start), 0.0)
    lower_bound = float(raw_start)

    entry_pos = _entry_position_near_start(
        projections,
        finite_indices,
        finite_values,
        lower_bound,
        span,
    )
    entry_idx = int(chunk_indices[entry_pos])

    exit_pos = int(finite_indices[np.argmax(finite_values)])
    initial_exit_idx = int(chunk_indices[exit_pos])
    exit_idx = _refined_exit_index(
        chunk_indices,
        projections,
        entry_pos,
        entry_idx,
        raw_end,
        span,
        initial_exit_idx,
    )

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
        gate_hints,
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
            gate_hints=gate_hints,
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


def _match_subset_indices(
    subset: np.ndarray,
    full: np.ndarray,
    decimals: int,
) -> Optional[np.ndarray]:
    """Return positions of ``subset`` samples inside ``full`` using rounding."""

    if subset.shape[0] == 0:
        return np.empty(0, dtype=int)
    if full.shape[0] == 0:
        return None

    rounded_full = np.round(full.astype(float, copy=False), decimals=decimals)
    rounded_subset = np.round(subset.astype(float, copy=False), decimals=decimals)

    lookup: Dict[tuple[float, float], list[int]] = {}
    for idx, point in enumerate(rounded_full):
        key = (float(point[0]), float(point[1]))
        lookup.setdefault(key, []).append(idx)

    indices = np.empty(subset.shape[0], dtype=int)
    prev_idx = -1
    for pos, point in enumerate(rounded_subset):
        key = (float(point[0]), float(point[1]))
        candidates = lookup.get(key)
        if not candidates:
            return None
        chosen = None
        for candidate in candidates:
            if candidate >= prev_idx:
                chosen = candidate
                break
        if chosen is None:
            chosen = candidates[-1]
        if chosen < prev_idx:
            return None
        indices[pos] = chosen
        prev_idx = chosen
    return indices


def _map_subset_indices(subset: np.ndarray, full: np.ndarray) -> Optional[np.ndarray]:
    """Return positions of ``subset`` samples inside ``full`` with relaxed tolerance."""

    for decimals in (6, 5, 4):
        indices = _match_subset_indices(subset, full, decimals)
        if indices is not None:
            return indices
    return None


def _resolve_gate_start_index(
    subset_indices: np.ndarray,
    crosses_start: bool,
    start_entry_idx: Optional[int],
    start_hint: Optional[GateCrossingHint],
) -> Optional[int]:
    """Return the sliced start index derived from the full gate crossings."""

    if not crosses_start and start_hint is None:
        return 0
    entry_target = start_entry_idx
    if entry_target is None and start_hint is not None:
        entry_target = start_hint.after_index
    if entry_target is None:
        return 0
    candidates = np.nonzero(subset_indices >= entry_target)[0]
    if candidates.size == 0:
        return None
    return int(candidates[0])


def _resolve_gate_end_index(
    subset_indices: np.ndarray,
    crosses_end: bool,
    end_exit_idx: Optional[int],
    end_hint: Optional[GateCrossingHint],
) -> Optional[int]:
    """Return the sliced end index derived from the full gate crossings."""

    if not crosses_end and end_hint is None:
        return subset_indices.shape[0] - 1
    exit_target = end_exit_idx
    if exit_target is None and end_hint is not None:
        exit_target = end_hint.before_index
    if exit_target is None:
        return subset_indices.shape[0] - 1
    end_limit = exit_target + 1
    candidates = np.nonzero(subset_indices <= end_limit)[0]
    if candidates.size == 0:
        return None
    return int(candidates[-1])


def _clip_activity_to_gate_window(
    activity_points: np.ndarray,
    segment_points: np.ndarray,
    start_tolerance_m: float,
    *,
    full_activity_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Clip activity samples to stay within the detected gate crossings."""

    if activity_points.shape[0] < 2 or segment_points.shape[0] < 2:
        return activity_points

    gate_context, subset_indices, used_full_context = _resolve_gate_context(
        activity_points,
        full_activity_points,
    )

    crossings = _gate_crossings(gate_context, segment_points, start_tolerance_m)
    (
        crosses_start,
        crosses_end,
        start_entry_idx,
        end_exit_idx,
        gate_hints,
    ) = crossings

    start_idx = _resolve_gate_start_index(
        subset_indices,
        crosses_start,
        start_entry_idx,
        gate_hints.start,
    )
    if start_idx is None:
        if full_activity_points is not None and gate_context is not activity_points:
            return _clip_activity_to_gate_window(
                activity_points,
                segment_points,
                start_tolerance_m,
            )
        return activity_points
    end_idx = _resolve_gate_end_index(
        subset_indices,
        crosses_end,
        end_exit_idx,
        gate_hints.end,
    )
    invalid_window = (
        start_idx is None
        or end_idx is None
        or end_idx <= start_idx
        or (start_idx == 0 and end_idx == activity_points.shape[0] - 1)
    )
    if invalid_window:
        if used_full_context:
            return _clip_activity_to_gate_window(
                activity_points,
                segment_points,
                start_tolerance_m,
            )
        return activity_points

    clipped = activity_points[start_idx : end_idx + 1]
    if clipped.shape[0] < 2:
        return activity_points
    return clipped


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
    gate_slice: Optional[Tuple[int, int]] = None,
) -> Optional[_RefinedCoverage]:
    """Derive an overlap window by filtering projections with acceptable offsets."""

    prepared = _prepare_refined_inputs(
        coverage,
        segment_points,
        max_offset_threshold,
        gate_slice,
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

    chunk_span = max(raw_end - raw_start, 0.0)
    if chunk_span <= 0.0:
        return None

    raw_bounds_full = getattr(coverage, "coverage_bounds", None)
    if raw_bounds_full is not None:
        baseline_span = max(raw_bounds_full[1] - raw_bounds_full[0], 0.0)
        if baseline_span > 0.0 and chunk_span < baseline_span * 0.7:
            return None
    if total_length > 0.0 and chunk_span < total_length * 0.7:
        return None

    sample_count = chunk_indices.size
    if sample_count < 3:
        return None

    guard_span = chunk_span
    segment_stride = total_length / max(segment_points.shape[0] - 1, 1)
    max_stride = max(segment_stride * 10.0, 25.0)
    mean_stride = guard_span / max(sample_count - 1, 1)
    # Reject slices that stay too sparse after removing large offsets.
    if mean_stride > max_stride:
        return None

    entry_idx, exit_idx = _select_timing_indices(
        chunk_indices,
        chunk_proj,
        raw_start,
        raw_end,
    )
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
    GateTimingHints,
]:
    """Evaluate coverage metrics and associated diagnostics for an activity."""

    coverage = compute_coverage(
        prepared_activity.metric_points,
        prepared_segment.resampled_points,
    )
    timestamps = np.asarray(prepared_activity.activity.timestamps_s, dtype=float)
    raw_bounds = coverage.coverage_bounds
    raw_max_offset = coverage.max_offset_m
    finish_distance = None
    if isinstance(raw_bounds, tuple) and len(raw_bounds) == 2:
        finish_distance = raw_bounds[1]
    required_span = min(3.0, max(0.5, start_tolerance_m * 0.02))
    (
        crosses_start,
        crosses_end,
        start_entry_idx,
        end_exit_idx,
        gate_hints,
    ) = _gate_crossings(
        prepared_activity.metric_points,
        prepared_segment.resampled_points,
        start_tolerance_m,
        finish_distance_m=finish_distance,
        timestamps=timestamps,
        projections=coverage.projections,
    )

    offsets = getattr(coverage, "offsets", None)
    gate_slice = _gate_slice_from_hints(
        gate_hints,
        offsets.shape[0] if isinstance(offsets, np.ndarray) else 0,
    )
    trimmed_max = _max_offset_within_slice(offsets, gate_slice)
    if trimmed_max is not None:
        coverage.max_offset_m = trimmed_max

    refined = _refine_coverage_window(
        coverage,
        prepared_segment.resampled_points,
        max_offset_threshold=max_offset_threshold,
        start_tolerance_m=start_tolerance_m,
        gate_slice=gate_slice,
    )
    timing_bounds: Optional[Tuple[float, float]] = raw_bounds
    timing_indices: Optional[Tuple[int, int]] = None
    refined_entry_idx: Optional[int] = None
    refined_exit_idx: Optional[int] = None
    if refined is not None:
        coverage.coverage_bounds = refined.expanded_bounds
        coverage.coverage_ratio = refined.ratio
        coverage.max_offset_m = refined.max_offset
        refined_entry_idx = refined.entry_index
        refined_exit_idx = refined.exit_index

    gate_hints = _prune_gate_hints(
        gate_hints,
        timestamps,
        entry_idx=refined_entry_idx
        if refined_entry_idx is not None
        else start_entry_idx,
        exit_idx=refined_exit_idx if refined_exit_idx is not None else end_exit_idx,
    )

    gate_hints, start_entry_idx = _restore_start_hint_from_refined_entry(
        gate_hints,
        prepared_activity.metric_points,
        prepared_segment.resampled_points,
        refined_entry_idx,
        required_span,
        start_entry_idx,
    )

    start_entry_idx, end_exit_idx = _apply_gate_indices(
        start_entry_idx,
        end_exit_idx,
        gate_hints,
        fallback_start=refined_entry_idx,
        fallback_end=refined_exit_idx,
    )

    if refined_entry_idx is not None and refined_exit_idx is not None:
        clamped_entry, clamped_exit = _clamp_timing_indices(
            refined_entry_idx,
            refined_exit_idx,
            start_entry_idx,
            end_exit_idx,
        )
        timing_indices = (clamped_entry, clamped_exit)
    elif start_entry_idx is not None and end_exit_idx is not None:
        if end_exit_idx > start_entry_idx:
            timing_indices = (start_entry_idx, end_exit_idx)

    coverage_diag = _build_coverage_diag(
        coverage,
        crosses_start,
        crosses_end,
        start_entry_idx,
        end_exit_idx,
        raw_bounds,
        raw_max_offset,
        refined,
        trimmed_max,
        gate_slice,
        gate_hints,
        timing_indices,
    )

    return coverage, coverage_diag, timing_bounds, timing_indices, gate_hints


def _should_accept_gate_trimmed_fallback(
    coverage: CoverageResult,
    trimmed_max_offset: Optional[float],
    trimmed_point_count: int,
    gate_clipped: bool,
    frechet_distance: float,
) -> Optional[Dict[str, float]]:
    """Return diagnostic payload when gate-clipped fallback should apply."""

    if not gate_clipped or trimmed_point_count < _GATE_FALLBACK_MIN_POINTS:
        return None
    if trimmed_max_offset is None or not np.isfinite(trimmed_max_offset):
        return None
    if trimmed_max_offset > _GATE_FALLBACK_MAX_OFFSET_M:
        return None
    coverage_ratio = getattr(coverage, "coverage_ratio", 0.0) or 0.0
    if coverage_ratio < max(0.995, MATCHING_COVERAGE_THRESHOLD):
        return None
    max_offset = getattr(coverage, "max_offset_m", None)
    if max_offset is not None and np.isfinite(max_offset) and max_offset > 40.0:
        return None
    fallback_threshold = trimmed_max_offset * _GATE_FALLBACK_FRECHET_MULTIPLIER
    if frechet_distance > fallback_threshold:
        return None
    return {
        "threshold_m": float(fallback_threshold),
        "coverage_ratio": float(coverage_ratio),
        "trimmed_max_offset_m": float(trimmed_max_offset),
    }


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

    gate_clipped = False
    clipped_activity = _clip_activity_to_gate_window(
        trimmed_activity,
        segment_window,
        start_tolerance_m,
        full_activity_points=prepared_activity.resampled_points,
    )
    if clipped_activity.shape[0] >= 2:
        gate_clipped = clipped_activity.shape[0] != trimmed_activity.shape[0]
        trimmed_activity = clipped_activity

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
    if gate_clipped:
        diag["similarity_window"]["gate_trimmed"] = True
    if dtw_distance is not None:
        diag["dtw_distance_m"] = dtw_distance

    if not matched:
        fallback_note = _should_accept_gate_trimmed_fallback(
            coverage,
            trimmed_max_offset,
            int(trimmed_activity.shape[0]),
            gate_clipped,
            frechet_distance,
        )
        if fallback_note is not None:
            diag["gate_trimmed_fallback"] = fallback_note
            matched = True

    return matched, frechet_distance, dtw_distance, similarity_threshold, diag


__all__ = ["match_activity_to_segment", "MatchResult", "Tolerances"]
