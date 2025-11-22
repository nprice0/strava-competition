"""Scanner that infers segment attempts from runner activities."""

from __future__ import annotations

import logging
from datetime import datetime
from threading import Event
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from ..config import ACTIVITY_SCAN_MAX_ACTIVITY_PAGES
from ..errors import StravaAPIError
from ..models import Runner, Segment
from ..strava_api import get_activities, get_activity_with_efforts
from .cache import ActivityDetailCache
from .models import ActivityScanResult

ActivityProvider = Callable[[Runner, datetime, datetime], Sequence[Dict[str, object]]]


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except Exception:
        return None


class ActivityEffortScanner:
    """Activity-based fallback scanner for runners without Strava segment access."""

    def __init__(
        self,
        *,
        activity_provider: ActivityProvider | None = None,
        activity_types: Iterable[str] | None = None,
        detail_cache_size: int = 128,
        max_activity_pages: Optional[int] = None,
    ) -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        self._activity_provider = activity_provider
        self._detail_cache = ActivityDetailCache(detail_cache_size)
        if activity_types is None:
            activity_types = ("Run",)
        self._activity_types = tuple(activity_types)
        if max_activity_pages is None:
            max_activity_pages = ACTIVITY_SCAN_MAX_ACTIVITY_PAGES
        self._max_activity_pages = max_activity_pages

    def scan_segment(  # noqa: C901 - loops + filtering logic
        self,
        runner: Runner,
        segment: Segment,
        *,
        cancel_event: Event | None = None,
    ) -> ActivityScanResult | None:
        if cancel_event and cancel_event.is_set():
            return None
        activities = self._get_activities(runner, segment, cancel_event)
        if not activities:
            return None

        attempts = 0
        effort_ids: List[int | str] = []
        inspected: List[Dict[str, object]] = []
        best_elapsed: float | None = None
        best_effort: Dict[str, object] | None = None
        best_activity_id: int | None = None
        best_moving_time: float | None = None
        best_start: datetime | None = None

        for summary in activities:
            if cancel_event and cancel_event.is_set():
                break
            activity_id = self._coerce_int(summary.get("id"))
            if activity_id is None:
                continue
            inspected.append({"id": activity_id, "name": summary.get("name")})
            detail = self._get_activity_detail(runner, activity_id, cancel_event)
            if not detail:
                continue
            efforts = detail.get("segment_efforts") or []
            for effort in efforts:
                segment_obj = effort.get("segment") or {}
                if segment_obj.get("id") != segment.id:
                    continue
                elapsed = self._coerce_float(effort.get("elapsed_time"))
                if elapsed is None or elapsed <= 0:
                    continue
                attempts += 1
                effort_id = effort.get("id")
                if effort_id is not None:
                    effort_ids.append(effort_id)
                if best_elapsed is None or elapsed < best_elapsed:
                    best_elapsed = elapsed
                    best_effort = effort
                    best_activity_id = activity_id
                    best_moving_time = self._coerce_float(effort.get("moving_time"))
                    best_start = _parse_iso_datetime(
                        effort.get("start_date_local") or effort.get("start_date")
                    )
        if attempts == 0 or best_elapsed is None:
            return None
        fastest_effort_id = best_effort.get("id") if best_effort else None
        return ActivityScanResult(
            segment_id=segment.id,
            attempts=attempts,
            fastest_elapsed=best_elapsed,
            fastest_effort_id=fastest_effort_id,
            fastest_activity_id=best_activity_id,
            fastest_start_date=best_start,
            moving_time=best_moving_time,
            effort_ids=effort_ids,
            inspected_activities=inspected,
        )

    def _get_activities(
        self,
        runner: Runner,
        segment: Segment,
        cancel_event: Event | None,
    ) -> List[Dict[str, object]]:
        if cancel_event and cancel_event.is_set():
            return []
        provider = self._activity_provider or self._default_activity_provider
        try:
            data = provider(runner, segment.start_date, segment.end_date)
        except Exception:
            self._log.warning(
                "Activity provider failed runner=%s segment=%s",
                runner.name,
                segment.id,
                exc_info=True,
            )
            return []
        return list(data or [])

    def _default_activity_provider(
        self,
        runner: Runner,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[Dict[str, object]]:
        return (
            get_activities(
                runner,
                start_date,
                end_date,
                activity_types=self._activity_types,
                max_pages=self._max_activity_pages,
            )
            or []
        )

    def _get_activity_detail(
        self,
        runner: Runner,
        activity_id: int,
        cancel_event: Event | None,
    ) -> Dict[str, object] | None:
        if cancel_event and cancel_event.is_set():
            return None
        key = (runner.strava_id, activity_id)
        cached = self._detail_cache.get(key)
        if cached is not None:
            return cached
        try:
            detail = get_activity_with_efforts(
                runner,
                activity_id,
                include_all_efforts=True,
            )
        except StravaAPIError:
            self._log.warning(
                "StravaAPIError while fetching include_all_efforts runner=%s activity=%s",
                runner.name,
                activity_id,
                exc_info=True,
            )
            raise
        except Exception:
            self._log.debug(
                "Failed to fetch include_all_efforts runner=%s activity=%s",
                runner.name,
                activity_id,
                exc_info=True,
            )
            return None
        self._detail_cache.put(key, detail)
        return detail

    @staticmethod
    def _coerce_int(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
