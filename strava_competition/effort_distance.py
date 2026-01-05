"""Helpers for deriving effort distances from raw GPS streams."""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Sequence

from .errors import StravaAPIError, StravaStreamEmptyError
from .models import Runner
from .tools.geometry.fetchers import fetch_activity_stream
from .tools.geometry.models import LatLon

_LOG = logging.getLogger(__name__)
_EARTH_RADIUS_M = 6_371_000.0


def derive_effort_distance_m(
    runner: Runner,
    effort: Mapping[str, Any],
    *,
    allow_stream: bool = True,
) -> float | None:
    """Return the best-effort distance for an effort payload.

    When ``allow_stream`` is True (default) the function attempts the more
    accurate GPS stream measurement before falling back to any ``distance``
    field present on the payload. Callers can disable the stream lookup to
    avoid additional API fetches when only the Strava-provided value is needed.
    """

    if allow_stream and _has_stream_indices(effort):
        distance = compute_effort_distance_from_payload(runner, effort)
        if distance is not None:
            return distance
    return _coerce_float(effort.get("distance"))


def compute_effort_distance_from_payload(
    runner: Runner,
    effort: Mapping[str, Any],
) -> float | None:
    """Resolve indices/activity from a segment effort payload and measure distance."""

    activity_id = _extract_activity_id(effort)
    if activity_id is None:
        return None
    start_index = effort.get("start_index")
    end_index = effort.get("end_index")
    return compute_effort_distance(runner, activity_id, start_index, end_index)


def compute_effort_distance(
    runner: Runner,
    activity_id: int | str,
    start_index: int | None,
    end_index: int | None,
) -> float | None:
    """Return the travelled distance between two stream indices for an effort.

    Parameters:
        runner: Runner whose access token should be used for the API fetch.
        activity_id: Strava activity identifier that produced the effort.
        start_index: Inclusive index of the effort's first lat/lon sample.
        end_index: Inclusive index of the effort's last lat/lon sample.

    Returns:
        The summed haversine distance in metres, or ``None`` when the stream
        is unavailable or the provided indices cannot be sliced safely.
    """

    start_idx = _coerce_index(start_index)
    end_idx = _coerce_index(end_index)
    if start_idx is None or end_idx is None:
        return None
    if start_idx < 0 or end_idx < 0:
        return None
    if end_idx < start_idx:
        return None
    try:
        track = fetch_activity_stream(runner, int(activity_id))
    except (StravaStreamEmptyError, StravaAPIError):
        _LOG.debug(
            "Unable to fetch activity stream for runner=%s activity=%s",
            runner.name,
            activity_id,
            exc_info=True,
        )
        return None

    points = track.points
    total_points = len(points)
    if total_points == 0:
        return None
    if start_idx >= total_points:
        return None
    end_idx = min(end_idx, total_points - 1)
    if start_idx == end_idx:
        return 0.0

    segment_points = points[start_idx : end_idx + 1]
    return _sum_distances(segment_points)


def _coerce_index(value: int | str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extract_activity_id(effort: Mapping[str, Any]) -> int | None:
    activity_id = effort.get("activity_id")
    coerced = _coerce_index(activity_id)
    if coerced is not None:
        return coerced
    activity = effort.get("activity")
    if isinstance(activity, Mapping):
        return _coerce_index(activity.get("id"))
    return None


def _has_stream_indices(effort: Mapping[str, Any]) -> bool:
    return effort.get("start_index") is not None and effort.get("end_index") is not None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum_distances(points: Sequence[LatLon]) -> float:
    total = 0.0
    previous = points[0]
    for current in points[1:]:
        total += _haversine(previous, current)
        previous = current
    return total


def _haversine(first: LatLon, second: LatLon) -> float:
    lat1, lon1 = first
    lat2, lon2 = second
    sin = math.sin
    cos = math.cos
    radians = math.radians
    atan2 = math.atan2
    sqrt = math.sqrt
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = radians(lon2 - lon1)
    sin_half_lat = sin(delta_lat / 2.0)
    sin_half_lon = sin(delta_lon / 2.0)
    a = sin_half_lat**2 + cos(lat1_rad) * cos(lat2_rad) * sin_half_lon**2
    c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a))
    return _EARTH_RADIUS_M * c


__all__ = [
    "compute_effort_distance",
    "compute_effort_distance_from_payload",
    "derive_effort_distance_m",
]
