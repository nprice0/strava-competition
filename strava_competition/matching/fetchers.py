"""Helpers for retrieving Strava data used by the segment matcher."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Sequence, Tuple

from ..models import Runner
from ..strava_api import (
    fetch_activity_stream as api_fetch_activity_stream,
    fetch_segment_geometry as api_fetch_segment_geometry,
)
from ..config import MATCHING_ACTIVITY_STREAM_CACHE_SIZE
from .models import ActivityTrack, LatLon, SegmentGeometry


_ActivityCacheKey = Tuple[str, int]


class _ActivityStreamCache:
    """Simple LRU cache for activity streams scoped to the current process."""

    def __init__(self, max_entries: int) -> None:
        self._data: "OrderedDict[_ActivityCacheKey, ActivityTrack]" = OrderedDict()
        self._max_entries = max(0, max_entries)
        self._lock = RLock()

    def get(self, key: _ActivityCacheKey) -> ActivityTrack | None:
        with self._lock:
            if key not in self._data:
                return None
            value = self._data.pop(key)
            self._data[key] = value
            return value

    def put(self, key: _ActivityCacheKey, value: ActivityTrack) -> None:
        if self._max_entries <= 0:
            return
        with self._lock:
            if key in self._data:
                self._data.pop(key)
            elif len(self._data) >= self._max_entries:
                self._data.popitem(last=False)
            self._data[key] = value


_activity_stream_cache = _ActivityStreamCache(MATCHING_ACTIVITY_STREAM_CACHE_SIZE)


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

    cache_key: _ActivityCacheKey = (runner.strava_id, int(activity_id))
    cached = _activity_stream_cache.get(cache_key)
    if cached is not None:
        return cached

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
    _activity_stream_cache.put(cache_key, track)
    return track


def _normalize_point(point: Sequence[float]) -> LatLon:
    """Convert a raw lat/lon pair to a typed tuple."""

    if len(point) != 2:
        raise ValueError("Expected lat/lon pair from Strava stream")
    lat, lon = point
    return float(lat), float(lon)
