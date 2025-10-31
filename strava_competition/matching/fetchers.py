"""Helpers for retrieving Strava data used by the segment matcher."""

from __future__ import annotations

from typing import Sequence

from ..models import Runner
from ..strava_api import (
    fetch_activity_stream as api_fetch_activity_stream,
    fetch_segment_geometry as api_fetch_segment_geometry,
)
from .models import ActivityTrack, LatLon, SegmentGeometry


def fetch_segment_geometry(runner: Runner, segment_id: int) -> SegmentGeometry:
    """Fetch and normalise segment geometry details from the Strava API."""

    payload = api_fetch_segment_geometry(runner, segment_id)
    polyline = payload.get("polyline")
    distance = float(payload.get("distance", 0.0))
    metadata = {
        "name": payload.get("name"),
        "start_latlng": payload.get("start_latlng"),
        "end_latlng": payload.get("end_latlng"),
        "elevation_high": payload.get("elevation_high"),
        "elevation_low": payload.get("elevation_low"),
        "map": payload.get("map"),
        "raw": payload.get("raw"),
    }
    geometry = SegmentGeometry(
        segment_id=int(payload.get("segment_id", segment_id)),
        points=[],
        distance_m=distance,
        polyline=polyline,
        metadata=metadata,
    )
    return geometry


def fetch_activity_stream(runner: Runner, activity_id: int) -> ActivityTrack:
    """Fetch activity stream data (lat/lon + timestamps) from the Strava API."""

    payload = api_fetch_activity_stream(runner, activity_id)
    raw_points: Sequence[Sequence[float]] = payload.get("latlng", [])
    raw_times: Sequence[float] = payload.get("time", [])
    points = [_normalize_point(pair) for pair in raw_points]
    timestamps = [float(value) for value in raw_times]
    metadata = payload.get("metadata", {})
    track = ActivityTrack(
        activity_id=int(payload.get("activity_id", activity_id)),
        points=points,
        timestamps_s=timestamps,
        metadata=metadata,
    )
    return track


def _normalize_point(point: Sequence[float]) -> LatLon:
    """Convert a raw lat/lon pair to a typed tuple."""

    if len(point) != 2:
        raise ValueError("Expected lat/lon pair from Strava stream")
    lat, lon = point
    return float(lat), float(lon)
