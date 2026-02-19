"""Strava API helpers for segment efforts, activities, and data streams."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import requests

from .config import (
    ACTIVITY_SCAN_CACHE_INCLUDE_ALL_EFFORTS,
    REQUEST_TIMEOUT,
    STRAVA_BASE_URL,
)
from .errors import (
    StravaAPIError,
    StravaStreamEmptyError,
)
from .models import Runner
from .strava_client.activities import ActivitiesAPI
from .strava_client.base import ensure_runner_token as _ensure_runner_token
from .strava_client.rate_limiter import RateLimiter
from .strava_client.resources import ResourceAPI
from .strava_client.session import get_default_session

DEFAULT_TIMEOUT: int = REQUEST_TIMEOUT

# Shared limiter instance (thread-safe by design)
_limiter = RateLimiter()


def set_rate_limiter(max_concurrent: Optional[int] = None) -> None:
    """Adjust concurrency limit at runtime.

    Passing ``max_concurrent`` calls ``RateLimiter.resize``; omitted / None
    leaves the current limit unchanged.
    """
    get_default_client().set_rate_limiter(max_concurrent)


def _get_activities_impl(
    runner: Runner,
    start_date: datetime,
    end_date: datetime,
    *,
    activity_types: Optional[Iterable[str]] = ("Run",),
    max_pages: Optional[int] = None,
    session: Optional[requests.Session] = None,
    limiter: Optional[RateLimiter] = None,
    activities_api: Optional[ActivitiesAPI] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Delegate activity fetching to ``strava_client.activities.ActivitiesAPI``."""

    api = activities_api or ActivitiesAPI(
        session=session or get_default_session(),
        limiter=limiter or _limiter,
    )
    return api.get_activities(
        runner,
        start_date,
        end_date,
        activity_types=activity_types,
        max_pages=max_pages,
    )


def get_activity_with_efforts(
    runner: Runner,
    activity_id: int,
    *,
    include_all_efforts: bool = True,
) -> Dict[str, Any]:
    """Return ``/activities/{id}`` payload via the shared Strava client."""

    return get_default_client().get_activity_with_efforts(
        runner,
        activity_id,
        include_all_efforts=include_all_efforts,
    )


def fetch_segment_geometry(
    runner: Runner,
    segment_id: int,
) -> Dict[str, Any]:
    """Return high-resolution geometry details for a Strava segment."""

    return get_default_client().fetch_segment_geometry(runner, segment_id)


def fetch_activity_stream(
    runner: Runner,
    activity_id: int,
    *,
    stream_types: tuple[str, ...] = ("latlng", "time"),
    resolution: str = "high",
    series_type: str = "time",
) -> Dict[str, Any]:
    """Return GPS stream data for an activity (lat/lon + timestamps)."""

    return get_default_client().fetch_activity_stream(
        runner,
        activity_id,
        stream_types=stream_types,
        resolution=resolution,
        series_type=series_type,
    )


class StravaClient:
    """Thin façade that wires dependencies for Strava API operations."""

    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        limiter: Optional["RateLimiter"] = None,
        activities_api: Optional[ActivitiesAPI] = None,
        resource_api: Optional[ResourceAPI] = None,
    ) -> None:
        self._session = session or get_default_session()
        self._limiter = limiter or _limiter
        self._activities = activities_api or ActivitiesAPI(
            session=self._session,
            limiter=self._limiter,
        )
        self._resources = resource_api or ResourceAPI(
            session=self._session,
            limiter=self._limiter,
            timeout=DEFAULT_TIMEOUT,
        )

    def ensure_runner_token(self, runner: Runner) -> None:
        """Ensure ``runner`` has a valid access token via shared helper."""

        _ensure_runner_token(runner)

    def get_activities(
        self,
        runner: Runner,
        start_date: datetime,
        end_date: datetime,
        *,
        activity_types: Optional[Iterable[str]] = ("Run",),
        max_pages: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        return _get_activities_impl(
            runner,
            start_date,
            end_date,
            activity_types=activity_types,
            max_pages=max_pages,
            session=self._session,
            limiter=self._limiter,
            activities_api=self._activities,
        )

    def get_activity_with_efforts(
        self,
        runner: Runner,
        activity_id: int,
        *,
        include_all_efforts: bool = True,
    ) -> Dict[str, Any]:
        params = {"include_all_efforts": "true"} if include_all_efforts else None
        url = f"{STRAVA_BASE_URL}/activities/{activity_id}"
        context = "activity_detail"
        if include_all_efforts and ACTIVITY_SCAN_CACHE_INCLUDE_ALL_EFFORTS:
            payload = self._resources.fetch_with_capture(runner, url, params, context)
        else:
            payload = self._resources.fetch_json(runner, url, params, context)
        if isinstance(payload, dict):
            return payload
        raise StravaAPIError(
            f"{context} returned non-object payload for runner {runner.name} activity={activity_id}"
        )

    def fetch_segment_geometry(
        self,
        runner: Runner,
        segment_id: int,
    ) -> Dict[str, Any]:
        url = f"{STRAVA_BASE_URL}/segments/{segment_id}"
        context = f"segment:{segment_id}"
        data = self._resources.fetch_with_capture(
            runner, url, params=None, context=context
        )
        if not isinstance(data, dict):
            message = f"{context} payload had unexpected type {type(data).__name__}"
            logging.error(message)
            raise StravaAPIError(message)
        map_info = data.get("map") or {}
        polyline = map_info.get("polyline") or map_info.get("summary_polyline")
        if not polyline:
            message = f"{context} missing polyline data for runner {runner.name}"
            logging.error(message)
            raise StravaAPIError(message)
        distance_val = data.get("distance")
        try:
            distance_m = float(distance_val) if distance_val is not None else 0.0
        except (TypeError, ValueError):
            distance_m = 0.0
        return {
            "segment_id": int(data.get("id", segment_id)),
            "name": data.get("name"),
            "distance": distance_m,
            "polyline": polyline,
            "start_latlng": data.get("start_latlng"),
            "end_latlng": data.get("end_latlng"),
            "elevation_high": data.get("elevation_high"),
            "elevation_low": data.get("elevation_low"),
            "map": map_info,
            "raw": data,
        }

    def fetch_activity_stream(
        self,
        runner: Runner,
        activity_id: int,
        *,
        stream_types: tuple[str, ...] = ("latlng", "time"),
        resolution: str = "high",
        series_type: str = "time",
    ) -> Dict[str, Any]:
        keys = ",".join(stream_types)
        url = f"{STRAVA_BASE_URL}/activities/{activity_id}/streams"
        context = f"activity_stream:{activity_id}"
        params = {
            "keys": keys,
            "key_by_type": "true",
            "resolution": resolution,
            "series_type": series_type,
        }
        data = self._resources.fetch_with_capture(
            runner, url, params=params, context=context
        )
        if not isinstance(data, dict):
            message = f"{context} payload had unexpected type {type(data).__name__}"
            logging.error(message)
            raise StravaAPIError(message)
        latlng_stream = data.get("latlng")
        time_stream = data.get("time")
        if not isinstance(latlng_stream, dict) or not isinstance(time_stream, dict):
            message = f"{context} missing required stream metadata"
            logging.warning(message)
            raise StravaStreamEmptyError(message)
        latlng_data = latlng_stream.get("data")
        time_data = time_stream.get("data")
        if not latlng_data or not time_data:
            message = f"{context} missing latlng or time samples"
            logging.warning(message)
            raise StravaStreamEmptyError(message)
        if len(latlng_data) != len(time_data):
            message = (
                f"{context} stream length mismatch latlng={len(latlng_data)}"
                f" time={len(time_data)}"
            )
            logging.warning(message)
            raise StravaStreamEmptyError(message)
        metadata = {
            "latlng": {k: v for k, v in latlng_stream.items() if k != "data"},
            "time": {k: v for k, v in time_stream.items() if k != "data"},
            "additional_streams": {
                key: value
                for key, value in data.items()
                if key not in {"latlng", "time"}
            },
        }
        return {
            "activity_id": activity_id,
            "latlng": latlng_data,
            "time": time_data,
            "metadata": metadata,
        }

    def set_rate_limiter(self, max_concurrent: Optional[int]) -> None:
        if max_concurrent is None:
            return
        self._limiter.resize(max_concurrent)


# Lazy-initialized default client to avoid configuration issues during testing
_default_client: Optional[StravaClient] = None


def get_default_client() -> StravaClient:
    """Get or create the default StravaClient instance."""
    global _default_client
    if _default_client is None:
        _default_client = StravaClient()
    return _default_client


def get_activities(
    runner: Runner,
    start_date: datetime,
    end_date: datetime,
    *,
    activity_types: Optional[Iterable[str]] = ("Run",),
    max_pages: Optional[int] = None,
) -> Optional[List[Dict[str, Any]]]:
    return get_default_client().get_activities(
        runner,
        start_date,
        end_date,
        activity_types=activity_types,
        max_pages=max_pages,
    )
