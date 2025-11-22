"""Utilities for estimating elapsed time across a matched segment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .preprocessing import PreparedActivityTrack, PreparedSegmentGeometry

MetricArray = NDArray[np.float64]
# Strava elapsed times snap to discrete stream samples even when the
# interpolated gate crossing differs by nearly two seconds. Use a larger
# delta so our fallback matcher mirrors the official timing harness.
_GATE_TIMING_SNAP_DELTA_S = 2.5


@dataclass(slots=True)
class SegmentTimingEstimate:
    """Detailed timing information for a matched segment effort."""

    elapsed_time_s: float
    entry_index: Optional[int]
    exit_index: Optional[int]
    entry_time_s: Optional[float]
    exit_time_s: Optional[float]


@dataclass(slots=True)
class GateCrossingHint:
    """Bracketed samples around a start/finish plane crossing."""

    before_index: int
    after_index: int
    before_value: float
    after_value: float


@dataclass(slots=True)
class GateTimingHints:
    """Optional gate crossing hints derived from plane intersection checks."""

    start: Optional[GateCrossingHint] = None
    end: Optional[GateCrossingHint] = None


def estimate_segment_time(
    activity_track: PreparedActivityTrack,
    segment_geometry: PreparedSegmentGeometry,
    coverage_range: Tuple[float, float],
    projections: Optional[MetricArray] = None,
    sample_indices: Optional[Tuple[int, int]] = None,
    gate_hints: Optional[GateTimingHints] = None,
) -> SegmentTimingEstimate:
    """Estimate elapsed time for the portion of an activity covering the segment.

    Args:
        activity_track: Metric projections and timestamps for the activity.
        segment_geometry: Prepared geometry representing the target segment.
        coverage_range: Tuple of start and end distances (in metres) that the
            activity is expected to cover along the segment polyline.
        projections: Optional precomputed projections of ``activity_track`` onto
            the segment polyline. When provided, the array must align with the
            activity timestamps; otherwise a fresh projection is computed.
        sample_indices: Optional tuple containing the indices of the first and
            last samples whose offsets fall within the refined coverage window.
            When provided, the timing logic expands to the immediate neighbour
            before the entry sample and the neighbour after the exit sample (if
            available) so durations do not underestimate the true effort.
        gate_hints: Optional start/finish plane crossing metadata used to
            interpolate precise entry/exit timestamps when gate crossings are
            detected. When provided, entry and exit indices map to the first
            in-gate sample and the final in-gate sample respectively.

    Returns:
        SegmentTimingEstimate capturing the elapsed time (seconds) plus entry and
        exit indices/timestamps for downstream diagnostics.

    Raises:
        ValueError: If timestamp and coordinate counts do not match or coverage
            bounds are malformed.
    """

    start_m, end_m = coverage_range
    if not np.isfinite(start_m) or not np.isfinite(end_m):
        raise ValueError("Coverage range must contain finite values")
    if start_m > end_m:
        start_m, end_m = end_m, start_m
    if start_m == end_m:
        return SegmentTimingEstimate(0.0, None, None, None, None)

    metric_points = activity_track.metric_points
    timestamps = np.asarray(activity_track.activity.timestamps_s, dtype=float)
    if metric_points.shape[0] != timestamps.shape[0]:
        raise ValueError("Activity points and timestamps must align")
    if metric_points.shape[0] < 2:
        return SegmentTimingEstimate(0.0, None, None, None, None)

    segment_points = segment_geometry.resampled_points
    if segment_points.shape[0] < 2:
        raise ValueError("Segment geometry must contain at least two points")

    if projections is not None:
        projection_array = np.asarray(projections, dtype=float)
        if projection_array.shape[0] != metric_points.shape[0]:
            raise ValueError("Projection count must match activity points")
    else:
        projection_array = _project_onto_polyline(metric_points, segment_points)

    resolved = _resolve_entry_exit(
        projection_array,
        timestamps,
        start_m,
        end_m,
        sample_indices,
    )
    if resolved is None:
        return SegmentTimingEstimate(0.0, None, None, None, None)
    entry_idx, entry_time, exit_idx, exit_time = _apply_gate_hints(
        resolved,
        gate_hints,
        timestamps,
    )

    if exit_time < entry_time:
        return SegmentTimingEstimate(0.0, entry_idx, exit_idx, entry_time, exit_time)

    elapsed = float(exit_time - entry_time)
    return SegmentTimingEstimate(elapsed, entry_idx, exit_idx, entry_time, exit_time)


def _apply_gate_hints(
    resolved: Tuple[int, float, int, float],
    gate_hints: Optional[GateTimingHints],
    timestamps: MetricArray,
) -> Tuple[int, float, int, float]:
    """Return entry/exit indices adjusted with optional gate hints."""

    entry_idx, entry_time, exit_idx, exit_time = resolved
    if gate_hints is None:
        return entry_idx, entry_time, exit_idx, exit_time

    adjusted_entry = _apply_start_hint(gate_hints.start, timestamps)
    if adjusted_entry is not None:
        entry_idx, entry_time = adjusted_entry

    adjusted_exit = _apply_end_hint(gate_hints.end, timestamps)
    if adjusted_exit is not None:
        exit_idx, exit_time = adjusted_exit

    if exit_time < entry_time:
        exit_idx = entry_idx
        exit_time = entry_time

    return entry_idx, entry_time, exit_idx, exit_time


def _apply_start_hint(
    hint: Optional[GateCrossingHint],
    timestamps: MetricArray,
) -> Optional[Tuple[int, float]]:
    """Return interpolated entry index/time using a gate crossing hint."""

    if hint is None:
        return None
    interpolated = _interpolate_gate_crossing(hint, timestamps, prefer_after=True)
    if interpolated is None:
        return None
    index, time_s = interpolated
    snapped = _snap_time_to_sample(index, time_s, timestamps)
    return index, snapped


def _apply_end_hint(
    hint: Optional[GateCrossingHint],
    timestamps: MetricArray,
) -> Optional[Tuple[int, float]]:
    """Return interpolated exit index/time using a gate crossing hint."""

    if hint is None:
        return None
    interpolated = _interpolate_gate_crossing(hint, timestamps, prefer_after=False)
    if interpolated is None:
        return None
    index, time_s = interpolated
    snapped = _snap_time_to_sample(index, time_s, timestamps)
    return index, snapped


def _find_entry_event(
    projections: MetricArray,
    timestamps: MetricArray,
    start_m: float,
) -> Optional[Tuple[int, float]]:
    """Return the closest sample strictly before the coverage interval starts."""

    if projections.size == 0:
        return None

    tolerance = max(1e-6, abs(start_m) * 1e-6)
    lower_threshold = start_m - tolerance

    before_mask = projections < lower_threshold
    if np.any(before_mask):
        last_before_idx = int(np.flatnonzero(before_mask)[-1])
        return last_before_idx, float(timestamps[last_before_idx])

    # Fallback: activity begins directly on or after the segment start.
    return 0, float(timestamps[0])


def _find_exit_event(
    projections: MetricArray,
    timestamps: MetricArray,
    end_m: float,
) -> Optional[Tuple[int, float]]:
    """Return the first index/time where the activity reaches the segment end."""

    if projections.size == 0:
        return None

    tolerance = max(1e-6, abs(end_m) * 1e-6)
    threshold = end_m - tolerance

    for idx, distance in enumerate(projections):
        if distance >= threshold:
            if idx == 0:
                return idx, float(timestamps[idx])
            prev_distance = projections[idx - 1]
            if prev_distance >= threshold:
                return idx, float(timestamps[idx])
            interpolated = _interpolate_time(
                prev_distance,
                distance,
                float(timestamps[idx - 1]),
                float(timestamps[idx]),
                end_m,
            )
            return idx, interpolated

    if projections.size and (end_m - projections[-1]) <= tolerance:
        last_idx = projections.shape[0] - 1
        return last_idx, float(timestamps[last_idx])
    return None


def _align_span_to_base(
    base_value: float,
    start_m: float,
    end_m: float,
    span: float,
    half_span: float,
) -> Tuple[float, float]:
    """Return coverage bounds shifted close to the local projection base."""

    adjusted_start = float(start_m)
    adjusted_end = float(end_m)
    while adjusted_start - base_value > half_span:
        adjusted_start -= span
        adjusted_end -= span
    while base_value - adjusted_start > half_span:
        adjusted_start += span
        adjusted_end += span
    return adjusted_start, adjusted_end


def _unwrap_projection_window(
    window_proj: MetricArray,
    start_m: float,
    end_m: float,
) -> Tuple[MetricArray, float, float]:
    """Return an adjusted projection window that stays close to monotonic."""

    span = max(abs(end_m - start_m), 1.0)
    half_span = span * 0.5
    tolerance = max(1e-6, span * 1e-6)
    unwrapped = np.asarray(window_proj, dtype=float).copy()
    offset = 0.0
    for idx in range(1, unwrapped.size):
        prev = unwrapped[idx - 1]
        curr = unwrapped[idx]
        if not (np.isfinite(prev) and np.isfinite(curr)):
            unwrapped[idx] = curr + offset
            continue
        delta = curr - prev
        if delta < -half_span - tolerance:
            offset += span
        elif delta > half_span + tolerance:
            offset -= span
        unwrapped[idx] = curr + offset
    return unwrapped, span, half_span


def _resolve_with_sample_hints(
    projections: MetricArray,
    timestamps: MetricArray,
    start_m: float,
    end_m: float,
    entry_hint: int,
    exit_hint: int,
) -> Optional[Tuple[int, float, int, float]]:
    """Resolve timing using the refined sample indices as hints."""

    point_count = timestamps.shape[0]
    search_start = max(0, entry_hint - 1)
    search_stop = min(point_count, exit_hint + 2)
    if search_stop - search_start < 2:
        return None

    window_proj = np.asarray(
        projections[search_start:search_stop], dtype=float, order="C"
    )
    window_time = timestamps[search_start:search_stop]

    unwrapped, span, half_span = _unwrap_projection_window(window_proj, start_m, end_m)
    finite = np.nonzero(np.isfinite(unwrapped))[0]
    if finite.size == 0:
        return None

    base_value = float(unwrapped[int(finite[0])])
    adjusted_start, adjusted_end = _align_span_to_base(
        base_value, start_m, end_m, span, half_span
    )

    entry = _find_entry_event(unwrapped, window_time, adjusted_start)
    exit_event = _find_exit_event(unwrapped, window_time, adjusted_end)
    if entry is None or exit_event is None:
        return None

    entry_idx_local, entry_time = entry
    exit_idx_local, exit_time = exit_event
    entry_idx = int(search_start + entry_idx_local)
    exit_idx = int(search_start + exit_idx_local)
    if exit_idx < entry_idx:
        exit_idx = entry_idx
        exit_time = entry_time
    return entry_idx, entry_time, exit_idx, exit_time


def _resolve_entry_exit(
    projections: MetricArray,
    timestamps: MetricArray,
    start_m: float,
    end_m: float,
    sample_indices: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, float, int, float]]:
    """Resolve entry/exit indices and timestamps with optional coverage hints."""

    point_count = timestamps.shape[0]
    if point_count == 0:
        return None

    if sample_indices is not None:
        entry_hint, exit_hint = sample_indices
        entry_hint = int(max(0, min(entry_hint, point_count - 1)))
        exit_hint = int(max(entry_hint, min(exit_hint, point_count - 1)))
        resolved = _resolve_with_sample_hints(
            projections, timestamps, start_m, end_m, entry_hint, exit_hint
        )
        if resolved is not None:
            return resolved

        entry_idx = entry_hint
        exit_idx = exit_hint
        entry_time = float(timestamps[entry_idx])
        exit_time = float(timestamps[exit_idx])
        if exit_idx < entry_idx:
            exit_idx = entry_idx
            exit_time = entry_time
        return entry_idx, entry_time, exit_idx, exit_time

    entry = _find_entry_event(projections, timestamps, start_m)
    exit_event = _find_exit_event(projections, timestamps, end_m)
    if entry is None or exit_event is None:
        return None
    entry_idx, entry_time = entry
    exit_idx, exit_time = exit_event
    return entry_idx, entry_time, exit_idx, exit_time


def _interpolate_time(
    distance_a: float,
    distance_b: float,
    time_a: float,
    time_b: float,
    target_distance: float,
) -> float:
    """Linearly interpolate the timestamp where distance crosses the target."""

    delta = distance_b - distance_a
    if delta == 0:
        return time_b
    ratio = (target_distance - distance_a) / delta
    ratio = min(max(ratio, 0.0), 1.0)
    return time_a + ratio * (time_b - time_a)


def _closest_gate_sample(
    before_idx: int,
    after_idx: int,
    time_before: float,
    time_after: float,
    value_before: float,
    value_after: float,
) -> Tuple[int, float]:
    """Return the sample closest to the gate plane using signed offsets."""

    if abs(value_after) <= abs(value_before):
        return after_idx, time_after
    return before_idx, time_before


def _interpolate_gate_crossing(
    hint: GateCrossingHint,
    timestamps: MetricArray,
    *,
    prefer_after: bool,
) -> Optional[Tuple[int, float]]:
    """Return interpolated timestamp at a plane crossing using hint samples."""

    point_count = timestamps.shape[0]
    before = int(hint.before_index)
    after = int(hint.after_index)
    if point_count == 0:
        return None
    if before < 0 or before >= point_count or after < 0 or after >= point_count:
        return None

    target_index = after if prefer_after else before
    time_before = float(timestamps[before])
    time_after = float(timestamps[after])
    value_before = float(hint.before_value)
    value_after = float(hint.after_value)

    delta = value_after - value_before
    epsilon = max(1e-6, abs(delta) * 1e-6)

    if not np.isfinite(delta) or abs(delta) < 1e-9:
        return _closest_gate_sample(
            before, after, time_before, time_after, value_before, value_after
        )

    if abs(value_before) <= epsilon:
        return before, time_before
    if abs(value_after) <= epsilon:
        return after, time_after

    if value_before * value_after > 0.0:
        return _closest_gate_sample(
            before, after, time_before, time_after, value_before, value_after
        )

    ratio = -value_before / delta
    ratio = float(min(max(ratio, 0.0), 1.0))
    interpolated = time_before + ratio * (time_after - time_before)

    return target_index, interpolated


def _snap_time_to_sample(
    index: int,
    candidate_time: float,
    timestamps: MetricArray,
) -> float:
    """Return gate timestamp snapped to the discrete sample when nearby."""

    if timestamps.size == 0:
        return candidate_time
    if index < 0 or index >= timestamps.shape[0]:
        return candidate_time
    sample_time = float(timestamps[index])
    if abs(sample_time - candidate_time) <= _GATE_TIMING_SNAP_DELTA_S:
        return sample_time
    return candidate_time


def _project_onto_polyline(
    points: MetricArray,
    polyline: MetricArray,
) -> MetricArray:
    """Project each activity point onto the segment polyline in metric space."""

    if points.shape[0] == 0:
        return np.zeros(0, dtype=float)

    segments = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    if segments.shape[0] == 0:
        return np.zeros(points.shape[0], dtype=float)

    segment_len_sq = np.where(segment_lengths > 0.0, segment_lengths**2, 1.0)
    segment_starts = polyline[:-1]
    cumulative = _cumulative_distances(polyline)

    # Broadcast all point-to-segment projections in a single vectorised pass.
    vectors = points[:, None, :] - segment_starts[None, :, :]
    t = np.einsum("nid,id->ni", vectors, segments) / segment_len_sq
    valid_lengths = segment_lengths > 0.0
    t = np.where(valid_lengths[None, :], t, np.nan)
    valid_t = valid_lengths[None, :] & np.greater_equal(t, 0.0) & np.less_equal(t, 1.0)

    nearest = segment_starts[None, :, :] + t[..., None] * segments[None, :, :]
    offsets_sq = np.sum((nearest - points[:, None, :]) ** 2, axis=2)
    offsets_sq = np.where(valid_t, offsets_sq, np.inf)

    best_segment_idx = np.argmin(offsets_sq, axis=1)
    best_offset_sq = offsets_sq[np.arange(points.shape[0]), best_segment_idx]
    best_start = cumulative[best_segment_idx]
    result = np.empty(points.shape[0], dtype=float)
    result.fill(np.nan)

    has_valid = np.isfinite(best_offset_sq)
    if np.any(has_valid):
        best_t = np.clip(t[np.arange(points.shape[0]), best_segment_idx], 0.0, 1.0)
        best_lengths = segment_lengths[best_segment_idx]
        result[has_valid] = (
            best_start[has_valid] + best_t[has_valid] * best_lengths[has_valid]
        )

    if np.any(~has_valid):
        subset = points[~has_valid]
        vertex_diff = polyline[None, :, :] - subset[:, None, :]
        vertex_dist_sq = np.sum(vertex_diff**2, axis=2)
        nearest_vertex_idx = np.argmin(vertex_dist_sq, axis=1)
        result[~has_valid] = cumulative[nearest_vertex_idx]

    return result


def _cumulative_distances(points: MetricArray) -> MetricArray:
    """Return cumulative distances along a metric polyline."""

    if points.shape[0] == 0:
        return np.zeros(1, dtype=float)
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return cumulative
