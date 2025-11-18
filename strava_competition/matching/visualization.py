"""Utilities for visualising activity and segment geometry on a map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import folium  # Using folium to build an interactive Leaflet map.
import numpy as np

from .preprocessing import PreparedActivityTrack, PreparedSegmentGeometry
from .validation import CoverageResult

LatLon = Tuple[float, float]
PathLike = Union[str, Path]

_ACTIVITY_COLOR = "#2c7bb6"
_SEGMENT_COLOR = "#1a9641"
_DIVERGENCE_COLOR = "#d73027"


@dataclass(slots=True)
class DeviationSummary:
    """Summary describing the worst deviation observed in an activity."""

    index: int
    offset_m: float
    coordinate: LatLon


def _contiguous_runs(indices: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive index ranges representing contiguous slices."""

    if indices.size == 0:
        return []
    slices: List[Tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for value in map(int, indices[1:]):
        if value != previous + 1:
            slices.append((start, previous))
            start = value
        previous = value
    slices.append((start, previous))
    return slices


def _extract_divergence_indices(
    offsets: Sequence[float],
    gate_slice: Optional[Tuple[int, int]],
    threshold_m: float,
) -> np.ndarray:
    """Return indices where the perpendicular offset exceeds the threshold."""

    values = np.asarray(offsets, dtype=float)
    if values.size == 0:
        return np.array([], dtype=int)
    finite = np.isfinite(values)
    if gate_slice is not None:
        gate_mask = np.zeros_like(finite)
        start, end = gate_slice
        gate_mask[start : end + 1] = True
        finite &= gate_mask
    exceeding = finite & (values >= threshold_m)
    return np.nonzero(exceeding)[0]


def _build_deviation_summary(
    offsets: Sequence[float],
    coordinates: Sequence[LatLon],
    gate_slice: Optional[Tuple[int, int]],
) -> Optional[DeviationSummary]:
    """Return information about the largest deviation inside the gate window."""

    values = np.asarray(offsets, dtype=float)
    if values.size == 0:
        return None
    if gate_slice is not None:
        start, end = gate_slice
        mask = np.zeros(values.shape[0], dtype=bool)
        mask[start : end + 1] = True
        masked_values = np.where(mask, values, -np.inf)
    else:
        masked_values = values
    if not np.isfinite(masked_values).any():
        return None
    index = int(np.argmax(masked_values))
    offset = float(masked_values[index])
    if not np.isfinite(offset):
        return None
    return DeviationSummary(index=index, offset_m=offset, coordinate=coordinates[index])


def _slice_activity_points(
    activity_points: Sequence[LatLon],
    slices: Iterable[Tuple[int, int]],
) -> List[List[LatLon]]:
    """Convert index slices into lists of lat/lon coordinates."""

    segments: List[List[LatLon]] = []
    for start, end in slices:
        if start < 0 or end >= len(activity_points) or end < start:
            continue
        segments.append(list(activity_points[start : end + 1]))
    return segments


def create_deviation_map(
    prepared_activity: PreparedActivityTrack,
    prepared_segment: PreparedSegmentGeometry,
    coverage: CoverageResult,
    *,
    threshold_m: float = 50.0,
    gate_slice: Optional[Tuple[int, int]] = None,
    output_html_path: Optional[PathLike] = None,
) -> folium.Map:
    """Create an interactive map that highlights large activity deviations.

    Args:
        prepared_activity: Activity artefacts produced by :func:`prepare_activity`.
        prepared_segment: Segment artefacts produced by :func:`prepare_geometry`.
        coverage: Coverage metrics returned by :func:`compute_coverage`.
        threshold_m: Minimum offset in metres to treat as a divergence.
        gate_slice: Optional inclusive index range restricting the highlighted span.
        output_html_path: Optional path to persist the resulting map as an HTML file.

    Returns:
        A :class:`folium.Map` instance containing the overlay.

    Raises:
        ValueError: If coverage offsets are unavailable or cannot be aligned with the
            activity coordinates.
    """

    offsets = coverage.offsets
    if offsets is None:
        raise ValueError("Coverage offsets are required to build the map overlay")
    if len(offsets) != len(prepared_activity.latlon_points):
        raise ValueError("Coverage offsets do not align with activity track points")

    divergence_indices = _extract_divergence_indices(offsets, gate_slice, threshold_m)
    slices = _contiguous_runs(divergence_indices)
    divergent_segments = _slice_activity_points(prepared_activity.latlon_points, slices)
    worst = _build_deviation_summary(
        offsets, prepared_activity.latlon_points, gate_slice
    )

    map_center: LatLon
    if worst is not None:
        map_center = worst.coordinate
    else:
        map_center = prepared_segment.latlon_points[0]

    folium_map = folium.Map(location=map_center, zoom_start=15, control_scale=True)
    folium.PolyLine(
        prepared_segment.latlon_points,
        color=_SEGMENT_COLOR,
        weight=4,
        opacity=0.8,
        tooltip="Segment geometry",
    ).add_to(folium_map)
    folium.PolyLine(
        prepared_activity.latlon_points,
        color=_ACTIVITY_COLOR,
        weight=4,
        opacity=0.5,
        tooltip="Activity track",
    ).add_to(folium_map)

    for segment in divergent_segments:
        if len(segment) < 2:
            continue
        folium.PolyLine(
            segment,
            color=_DIVERGENCE_COLOR,
            weight=6,
            opacity=0.9,
            tooltip="Divergent section",
        ).add_to(folium_map)

    if worst is not None:
        popup = folium.Popup(
            html=(
                f"<strong>Max deviation:</strong> {worst.offset_m:.1f} m "
                f"(sample {worst.index})"
            ),
            max_width=300,
        )
        folium.CircleMarker(
            location=worst.coordinate,
            radius=7,
            color=_DIVERGENCE_COLOR,
            fill=True,
            fill_color=_DIVERGENCE_COLOR,
            tooltip="Highest deviation",
            popup=popup,
        ).add_to(folium_map)

    if output_html_path is not None:
        output_path = Path(output_html_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        folium_map.save(str(output_path))

    return folium_map


__all__ = ["DeviationSummary", "create_deviation_map"]
