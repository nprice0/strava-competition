"""Scanner that infers segment attempts from runner activities.

This module scans runner activities for segment efforts embedded
in the activity detail payloads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event, RLock
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from cachetools import TTLCache

from ..config import ACTIVITY_SCAN_MAX_ACTIVITY_PAGES
from ..effort_distance import derive_effort_distance_m
from ..errors import StravaAPIError
from ..models import Runner, Segment
from ..strava_api import get_activities, get_activity_with_efforts
from ..utils import parse_iso_datetime
from .models import ActivityScanResult

# Type aliases for dependency injection
ActivityProvider = Callable[[Runner, datetime, datetime], Sequence[Dict[str, Any]]]
ElapsedAdjuster = Callable[
    [Runner, Segment, float, datetime | None], tuple[float, bool]
]

_DetailCacheKey = tuple[int | str, int]
_DEFAULT_ACTIVITY_TYPES = ("Run",)
_DEFAULT_DETAIL_CACHE_SIZE = 128
_DEFAULT_DETAIL_CACHE_TTL_SECONDS = 3600


@dataclass
class _BestEffortTracker:
    """Tracks the best (fastest) effort found during a scan."""

    adjusted_time: float | None = None
    raw_elapsed: float | None = None
    effort: Dict[str, Any] | None = None
    activity_id: int | None = None
    moving_time: float | None = None
    start_date: datetime | None = None
    bonus_applied: bool = False
    distance_m: float | None = None

    def is_faster(self, adjusted_time: float) -> bool:
        """Return True if the given time beats the current best."""
        return self.adjusted_time is None or adjusted_time < self.adjusted_time

    def update(
        self,
        *,
        adjusted_time: float,
        raw_elapsed: float,
        effort: Dict[str, Any],
        activity_id: int,
        start_date: datetime | None,
        bonus_applied: bool,
        distance_m: float | None,
    ) -> None:
        """Replace current best with a new faster effort."""
        self.adjusted_time = adjusted_time
        self.raw_elapsed = raw_elapsed
        self.effort = effort
        self.activity_id = activity_id
        self.moving_time = _coerce_float(effort.get("moving_time"))
        self.start_date = start_date
        self.bonus_applied = bonus_applied
        self.distance_m = distance_m


@dataclass
class _ScanAccumulator:
    """Accumulates statistics and effort IDs during a segment scan."""

    attempts: int = 0
    effort_ids: List[int | str] = field(default_factory=list)
    inspected_activities: List[Dict[str, object]] = field(default_factory=list)
    filtered_by_distance: int = 0
    best: _BestEffortTracker = field(default_factory=_BestEffortTracker)

    def record_attempt(self, effort_id: int | str | None) -> None:
        """Record a valid attempt, optionally tracking the effort ID."""
        self.attempts += 1
        if effort_id is not None:
            self.effort_ids.append(effort_id)

    def record_inspected_activity(self, activity_id: int, name: str | None) -> None:
        """Record that an activity was inspected."""
        self.inspected_activities.append({"id": activity_id, "name": name})

    def record_distance_filter(self) -> None:
        """Record that an effort was filtered out due to insufficient distance."""
        self.filtered_by_distance += 1


class ActivityEffortScanner:
    """Scanner that finds segment efforts within runner activities.

    Fetches a runner's activities within a date range and inspects
    each activity's embedded segment efforts to find attempts on a target segment.

    Attributes:
        activity_provider: Optional callable to fetch activities (for testing).
        activity_types: Strava activity types to include (default: "Run").
        elapsed_adjuster: Optional callable to adjust elapsed times (e.g., birthday bonus).
    """

    def __init__(
        self,
        *,
        activity_provider: ActivityProvider | None = None,
        activity_types: Iterable[str] | None = None,
        detail_cache_size: int = _DEFAULT_DETAIL_CACHE_SIZE,
        max_activity_pages: int | None = None,
        elapsed_adjuster: ElapsedAdjuster | None = None,
    ) -> None:
        """Initialize the scanner.

        Args:
            activity_provider: Custom activity fetcher (defaults to Strava API).
            activity_types: Activity types to scan (defaults to "Run").
            detail_cache_size: Max entries in the activity detail cache.
            max_activity_pages: Max pages of activities to fetch.
            elapsed_adjuster: Callable to adjust elapsed time (e.g., birthday bonus).
        """
        self._log = logging.getLogger(self.__class__.__name__)
        self._activity_provider = activity_provider
        self._detail_cache: TTLCache[_DetailCacheKey, Dict[str, Any]] = TTLCache(
            maxsize=max(1, detail_cache_size),
            ttl=_DEFAULT_DETAIL_CACHE_TTL_SECONDS,
        )
        self._detail_cache_lock = RLock()
        self._activity_types = tuple(activity_types or _DEFAULT_ACTIVITY_TYPES)
        self._max_activity_pages = (
            max_activity_pages
            if max_activity_pages is not None
            else ACTIVITY_SCAN_MAX_ACTIVITY_PAGES
        )
        self._elapsed_adjuster = elapsed_adjuster

    def scan_segment(
        self,
        runner: Runner,
        segment: Segment,
        *,
        cancel_event: Event | None = None,
    ) -> ActivityScanResult | None:
        """Scan runner's activities for efforts on a segment.

        Args:
            runner: The runner whose activities to scan.
            segment: The target segment to find efforts for.
            cancel_event: Optional threading event to signal cancellation.

        Returns:
            ActivityScanResult with the fastest effort found, or None if no
            valid efforts exist.
        """
        if _is_cancelled(cancel_event):
            return None

        activities = self._fetch_activities(runner, segment, cancel_event)
        if not activities:
            return None

        accumulator = _ScanAccumulator()
        min_distance = _effective_min_distance(segment)

        for activity_summary in activities:
            if _is_cancelled(cancel_event):
                break
            self._process_activity(
                runner,
                segment,
                activity_summary,
                min_distance,
                accumulator,
                cancel_event,
            )

        return self._build_result(runner, segment, min_distance, accumulator)

    def _process_activity(
        self,
        runner: Runner,
        segment: Segment,
        activity_summary: Dict[str, Any],
        min_distance: float,
        accumulator: _ScanAccumulator,
        cancel_event: Event | None,
    ) -> None:
        """Process a single activity, extracting matching segment efforts."""
        activity_id = _coerce_int(activity_summary.get("id"))
        if activity_id is None:
            return

        accumulator.record_inspected_activity(activity_id, activity_summary.get("name"))

        detail = self._fetch_activity_detail(runner, activity_id, cancel_event)
        if not detail:
            return

        efforts_payload = detail.get("segment_efforts")
        if not isinstance(efforts_payload, list):
            return

        for effort in efforts_payload:
            self._process_effort(
                runner, segment, effort, activity_id, min_distance, accumulator
            )

    def _process_effort(
        self,
        runner: Runner,
        segment: Segment,
        effort: Any,
        activity_id: int,
        min_distance: float,
        accumulator: _ScanAccumulator,
    ) -> None:
        """Validate and process a single segment effort."""
        if not isinstance(effort, dict):
            return

        # Check segment ID matches
        segment_obj = effort.get("segment")
        if not isinstance(segment_obj, Mapping) or segment_obj.get("id") != segment.id:
            return

        # Validate elapsed time
        elapsed = _coerce_float(effort.get("elapsed_time"))
        if elapsed is None or elapsed <= 0:
            return

        # Parse and validate effort date
        start_dt = self._parse_effort_date(effort)
        if not self._effort_within_window(start_dt, segment):
            return

        # Check minimum distance requirement
        distance = derive_effort_distance_m(
            runner, effort, allow_stream=min_distance > 0
        )
        if min_distance > 0 and (distance is None or distance < min_distance):
            accumulator.record_distance_filter()
            return

        # Valid attempt - record it
        effort_id = _coerce_effort_id(effort.get("id"))
        accumulator.record_attempt(effort_id)

        # Apply elapsed time adjustments (e.g., birthday bonus)
        adjusted, bonus_applied = self._apply_elapsed_adjuster(
            runner, segment, elapsed, start_dt
        )

        # Update best if this is faster
        if accumulator.best.is_faster(adjusted):
            accumulator.best.update(
                adjusted_time=adjusted,
                raw_elapsed=elapsed,
                effort=effort,
                activity_id=activity_id,
                start_date=start_dt,
                bonus_applied=bonus_applied,
                distance_m=distance,
            )

    def _parse_effort_date(self, effort: Dict[str, Any]) -> datetime | None:
        """Extract the effort's start datetime from the payload."""
        return parse_iso_datetime(
            effort.get("start_date_local") or effort.get("start_date")
        )

    def _effort_within_window(
        self, effort_date: datetime | None, segment: Segment
    ) -> bool:
        """Check if an effort date falls within the segment's date window."""
        if effort_date is None:
            # No date available - allow the effort (conservative approach)
            return True

        effort_naive = _to_naive(effort_date)
        window_start = _to_naive(segment.start_date)
        window_end = _to_naive(segment.end_date)

        return window_start <= effort_naive <= window_end

    def _build_result(
        self,
        runner: Runner,
        segment: Segment,
        min_distance: float,
        accumulator: _ScanAccumulator,
    ) -> ActivityScanResult | None:
        """Build the final scan result from accumulated data."""
        best = accumulator.best

        if accumulator.attempts == 0 or best.adjusted_time is None:
            return None

        # Try to get precise distance for the best effort if not filtered
        final_distance = best.distance_m
        if min_distance <= 0 and best.effort is not None:
            precise = derive_effort_distance_m(runner, best.effort, allow_stream=True)
            if precise is not None:
                final_distance = precise

        effort_id = _coerce_effort_id(best.effort.get("id")) if best.effort else None

        return ActivityScanResult(
            segment_id=segment.id,
            attempts=accumulator.attempts,
            fastest_elapsed=best.adjusted_time,
            fastest_effort_id=effort_id,
            fastest_activity_id=best.activity_id,
            fastest_start_date=best.start_date,
            moving_time=best.moving_time,
            fastest_distance_m=final_distance,
            effort_ids=accumulator.effort_ids,
            inspected_activities=accumulator.inspected_activities,
            birthday_bonus_applied=best.bonus_applied,
            filtered_efforts_below_distance=accumulator.filtered_by_distance,
        )

    # -------------------------------------------------------------------------
    # Activity fetching
    # -------------------------------------------------------------------------

    def _fetch_activities(
        self,
        runner: Runner,
        segment: Segment,
        cancel_event: Event | None,
    ) -> List[Dict[str, Any]]:
        """Fetch activities for the runner within the segment's date range."""
        if _is_cancelled(cancel_event):
            return []

        provider = self._activity_provider or self._default_activity_provider
        try:
            data = provider(runner, segment.start_date, segment.end_date)
        except Exception as exc:
            self._log.warning(
                "Activity provider failed runner=%s segment=%s: %s",
                runner.name,
                segment.id,
                exc,
                exc_info=True,
            )
            return []

        if data is None:
            return []

        if not isinstance(data, Sequence):
            self._log.warning(
                "Activity provider returned unexpected type %s for runner=%s segment=%s",
                type(data).__name__,
                runner.name,
                segment.id,
            )
            return []

        return list(data)

    def _default_activity_provider(
        self,
        runner: Runner,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[Dict[str, Any]]:
        """Default provider that fetches from Strava API."""
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

    def _fetch_activity_detail(
        self,
        runner: Runner,
        activity_id: int,
        cancel_event: Event | None,
    ) -> Dict[str, Any] | None:
        """Fetch activity detail with segment efforts, using cache when available."""
        if _is_cancelled(cancel_event):
            return None

        key: _DetailCacheKey = (runner.strava_id, activity_id)

        # Check cache first
        with self._detail_cache_lock:
            cached = self._detail_cache.get(key)
        if cached is not None:
            return cached

        # Fetch from API
        try:
            detail = get_activity_with_efforts(
                runner, activity_id, include_all_efforts=True
            )
        except StravaAPIError:
            self._log.warning(
                "StravaAPIError fetching activity detail runner=%s activity=%s",
                runner.name,
                activity_id,
                exc_info=True,
            )
            raise
        except Exception:
            self._log.debug(
                "Failed to fetch activity detail runner=%s activity=%s",
                runner.name,
                activity_id,
                exc_info=True,
            )
            return None

        # Cache the result
        with self._detail_cache_lock:
            self._detail_cache[key] = detail

        return detail

    def _apply_elapsed_adjuster(
        self,
        runner: Runner,
        segment: Segment,
        elapsed: float,
        start_date: datetime | None,
    ) -> tuple[float, bool]:
        """Apply elapsed time adjustment (e.g., birthday bonus).

        Returns:
            Tuple of (adjusted_elapsed, was_adjusted).
        """
        if self._elapsed_adjuster is None:
            return float(elapsed), False

        try:
            adjusted, applied = self._elapsed_adjuster(
                runner, segment, elapsed, start_date
            )
            return float(adjusted), bool(applied)
        except Exception:
            self._log.debug(
                "Elapsed adjuster failed runner=%s segment=%s",
                runner.name,
                segment.id,
                exc_info=True,
            )
            return float(elapsed), False


# -----------------------------------------------------------------------------
# Module-level helper functions
# -----------------------------------------------------------------------------


def _is_cancelled(event: Event | None) -> bool:
    """Check if the cancellation event is set."""
    return event is not None and event.is_set()


def _to_naive(dt: datetime | Any) -> datetime:
    """Convert a datetime to naive (strip timezone info).

    Handles both standard datetime and pandas Timestamp objects.
    """
    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


def _coerce_int(value: Any) -> int | None:
    """Attempt to coerce a value to int, returning None on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    """Attempt to coerce a value to float, returning None on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_effort_id(value: Any) -> int | str | None:
    """Extract effort ID if it's a valid type."""
    if isinstance(value, (int, str)):
        return value
    return None


def _effective_min_distance(segment: Segment) -> float:
    """Extract the effective minimum distance threshold from a segment."""
    raw_value = getattr(segment, "min_distance_meters", None)
    try:
        numeric = float(raw_value) if raw_value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return numeric if numeric > 0 else 0.0
