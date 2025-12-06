"""Preprocessing utilities for segment and activity geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from polyline import decode as polyline_decode
from pyproj import CRS, Transformer
from shapely.geometry import LineString

from ...config import GEOMETRY_MAX_RESAMPLED_POINTS, GEOMETRY_MAX_SIMPLIFIED_POINTS
from .models import ActivityTrack, LatLon, SegmentGeometry

MetricArray = NDArray[np.float64]


@dataclass(slots=True)
class PreparedSegmentGeometry:
    """Container holding reusable metric representations of a segment."""

    segment: SegmentGeometry
    latlon_points: List[LatLon]
    metric_points: MetricArray
    simplified_points: MetricArray
    resampled_points: MetricArray
    transformer: Transformer
    simplification_tolerance_m: float
    resample_interval_m: float
    simplification_capped: bool
    resample_capped: bool


@dataclass(slots=True)
class PreparedActivityTrack:
    """Container holding metric representations of an activity track."""

    activity: ActivityTrack
    latlon_points: List[LatLon]
    metric_points: MetricArray
    simplified_points: MetricArray
    resampled_points: MetricArray
    transformer: Transformer
    simplification_tolerance_m: float
    resample_interval_m: float
    simplification_capped: bool
    resample_capped: bool


def decode_polyline(encoded: str) -> List[LatLon]:
    """Decode an encoded polyline string into a list of (lat, lon) tuples."""

    if not encoded:
        return []
    try:
        decoded = polyline_decode(encoded)
    except (ValueError, TypeError) as exc:
        raise ValueError("Unable to decode polyline") from exc
    return [(float(lat), float(lon)) for lat, lon in decoded]


def reproject_to_local_crs(
    points: Sequence[LatLon],
) -> Tuple[MetricArray, Transformer]:
    """Project lat/lon points into a local metric coordinate system."""

    if not points:
        raise ValueError("Cannot reproject an empty point collection")
    transformer = _build_local_transformer(points)
    metric = _project_points(points, transformer)
    return metric, transformer


def simplify_points(
    points: Iterable[Sequence[float]], tolerance_m: float
) -> MetricArray:
    """Simplify metric coordinates while preserving endpoints and overall shape."""

    array = _as_metric_array(points)
    if len(array) < 3 or tolerance_m <= 0:
        return array
    line = LineString(array)
    simplified = line.simplify(tolerance_m, preserve_topology=False)
    return _as_metric_array(simplified.coords)


def resample_by_distance(
    points: Iterable[Sequence[float]], interval_m: float
) -> MetricArray:
    """Resample coordinates so successive points are spaced by ``interval_m`` meters."""

    if interval_m <= 0:
        raise ValueError("interval_m must be greater than zero")
    array = _as_metric_array(points)
    count = len(array)
    if count == 0:
        return array
    if count == 1:
        return array.copy()
    deltas = np.diff(array, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative[-1]
    if total_length == 0:
        return np.repeat(array[:1], repeats=max(int(np.ceil(1)), 1), axis=0)
    target = _build_target_distances(total_length, interval_m)
    x = np.interp(target, cumulative, array[:, 0])
    y = np.interp(target, cumulative, array[:, 1])
    return np.column_stack((x, y))


def prepare_geometry(
    segment: SegmentGeometry,
    *,
    simplification_tolerance_m: float,
    resample_interval_m: float,
) -> PreparedSegmentGeometry:
    """Return cached geometry artefacts for a segment."""

    latlon = list(segment.points) if segment.points else []
    if not latlon and segment.polyline:
        latlon = decode_polyline(segment.polyline)
    if not latlon:
        raise ValueError("Segment geometry has no coordinate data")
    metric_points, transformer = reproject_to_local_crs(latlon)
    simplified, effective_simplification, simpl_capped = _simplify_with_budget(
        metric_points,
        simplification_tolerance_m,
        GEOMETRY_MAX_SIMPLIFIED_POINTS,
    )
    resampled, effective_resample, resample_capped = _resample_with_budget(
        simplified,
        resample_interval_m,
        GEOMETRY_MAX_RESAMPLED_POINTS,
    )
    return PreparedSegmentGeometry(
        segment=segment,
        latlon_points=latlon,
        metric_points=metric_points,
        simplified_points=simplified,
        resampled_points=resampled,
        transformer=transformer,
        simplification_tolerance_m=effective_simplification,
        resample_interval_m=effective_resample,
        simplification_capped=simpl_capped,
        resample_capped=resample_capped,
    )


def prepare_activity(
    activity: ActivityTrack,
    transformer: Optional[Transformer] = None,
    *,
    simplification_tolerance_m: float,
    resample_interval_m: float,
) -> PreparedActivityTrack:
    """Return cached geometry artefacts for an activity track."""

    if len(activity.points) != len(activity.timestamps_s):
        raise ValueError("Activity points and timestamps must be the same length")
    latlon = list(activity.points)
    if not latlon:
        raise ValueError("Activity track has no GPS points")
    if transformer is None:
        metric_points, transformer = reproject_to_local_crs(latlon)
    else:
        metric_points = _project_points(latlon, transformer)
    simplified, effective_simplification, simpl_capped = _simplify_with_budget(
        metric_points,
        simplification_tolerance_m,
        GEOMETRY_MAX_SIMPLIFIED_POINTS,
    )
    resampled, effective_resample, resample_capped = _resample_with_budget(
        simplified,
        resample_interval_m,
        GEOMETRY_MAX_RESAMPLED_POINTS,
    )
    return PreparedActivityTrack(
        activity=activity,
        latlon_points=latlon,
        metric_points=metric_points,
        simplified_points=simplified,
        resampled_points=resampled,
        transformer=transformer,
        simplification_tolerance_m=effective_simplification,
        resample_interval_m=effective_resample,
        simplification_capped=simpl_capped,
        resample_capped=resample_capped,
    )


def _simplify_with_budget(
    points: MetricArray,
    tolerance_m: float,
    max_points: int,
) -> Tuple[MetricArray, float, bool]:
    """Simplify points while capping the output cardinality."""

    effective_tolerance = max(tolerance_m, 0.0)
    simplified = simplify_points(points, effective_tolerance)
    adjusted = False
    if len(simplified) <= max_points:
        return simplified, effective_tolerance, adjusted

    # Increase tolerance iteratively to reduce the point count before decimating.
    attempts = 0
    while len(simplified) > max_points and attempts < 5:
        effective_tolerance = (
            effective_tolerance * 1.5 if effective_tolerance > 0 else 1.0
        )
        simplified = simplify_points(points, effective_tolerance)
        attempts += 1
        adjusted = True

    if len(simplified) > max_points:
        simplified = _decimate_points(simplified, max_points)
        adjusted = True

    return simplified, effective_tolerance, adjusted


def _resample_with_budget(
    points: MetricArray,
    interval_m: float,
    max_points: int,
) -> Tuple[MetricArray, float, bool]:
    """Resample points while enforcing an upper bound on the output length."""

    effective_interval = max(interval_m, 0.001)
    resampled = resample_by_distance(points, effective_interval)
    adjusted = False
    if len(resampled) <= max_points:
        return resampled, effective_interval, adjusted

    # Increase the interval enough to bring the count under budget if possible.
    ratio = len(resampled) / float(max_points)
    if ratio > 1.0:
        effective_interval = effective_interval * max(2.0, math.ceil(ratio))
        resampled = resample_by_distance(points, effective_interval)
        adjusted = True

    if len(resampled) > max_points:
        resampled = _decimate_points(resampled, max_points)
        adjusted = True

    return resampled, effective_interval, adjusted


def _decimate_points(points: MetricArray, max_points: int) -> MetricArray:
    """Down-sample a point array while preserving the endpoints."""

    max_points = max(2, max_points)
    count = points.shape[0]
    if count <= max_points:
        return points
    indices = np.linspace(0, count - 1, num=max_points, dtype=int)
    return points[indices]


def _build_target_distances(
    total_length: float, interval_m: float
) -> NDArray[np.float64]:
    """Return monotonically increasing sample distances that include the end point."""

    distances = [0.0]
    current = interval_m
    while current < total_length:
        distances.append(current)
        current += interval_m
    distances.append(total_length)
    return np.asarray(distances, dtype=float)


def _build_local_transformer(points: Sequence[LatLon]) -> Transformer:
    """Build a local UTM transformer centred on the provided coordinates."""

    lats = [pt[0] for pt in points]
    lons = [pt[1] for pt in points]
    mean_lat = float(np.mean(lats))
    mean_lon = float(np.mean(lons))
    zone = int((mean_lon + 180.0) // 6.0) + 1
    zone = max(1, min(zone, 60))
    if mean_lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    try:
        target_crs = CRS.from_epsg(epsg)
    except Exception:
        target_crs = CRS.from_epsg(3857)
    return Transformer.from_crs(CRS.from_epsg(4326), target_crs, always_xy=True)


def _project_points(points: Sequence[LatLon], transformer: Transformer) -> MetricArray:
    """Project lat/lon pairs through an existing transformer."""

    if not points:
        return np.empty((0, 2), dtype=float)
    lats = np.asarray([pt[0] for pt in points], dtype=float)
    lons = np.asarray([pt[1] for pt in points], dtype=float)
    xs, ys = transformer.transform(lons, lats)
    return np.column_stack((xs, ys)).astype(float, copy=False)


def _as_metric_array(points: Iterable[Sequence[float]]) -> MetricArray:
    """Convert an arbitrary iterable of 2D coordinates into a float64 array."""

    array = np.asarray(list(points), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Expected a sequence of 2D coordinates")
    return array
