"""Segment competition service (application layer).

Encapsulates orchestration of fetching efforts and aggregating results so
higher-level code (main, CLI, etc.) depends on a stable service API rather
than a free function implementation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Sequence, Tuple, Set

from cachetools import TTLCache

from ..activity_scan import ActivityEffortScanner
from ..config import (
    FORCE_ACTIVITY_SCAN_FALLBACK,
    RUNNER_ACTIVITY_CACHE_SIZE,
    USE_ACTIVITY_SCAN_FALLBACK,
    MAX_WORKERS,
    SEGMENT_SPLIT_WINDOWS_ENABLED,
)
from ..effort_distance import derive_effort_distance_m
from ..errors import StravaAPIError
from ..models import Segment, Runner, SegmentResult, SegmentGroup, SegmentWindow
from ..strava_api import get_segment_efforts, get_activities
from ..utils import parse_iso_datetime

ResultsMapping = Dict[str, Dict[str, List[SegmentResult]]]

_ActivityCacheKey = Tuple[int | str, datetime, datetime]

# Cache TTL for runner activities (1 hour)
_ACTIVITY_CACHE_TTL_SECONDS = 3600


@dataclass(frozen=True, slots=True)
class _ValidatedEffort:
    """Intermediate representation of a validated segment effort."""

    adjusted_elapsed: float
    raw_elapsed: float
    start_date: datetime | None
    effort: dict
    birthday_bonus_applied: bool
    distance_m: float | None


class SegmentService:
    """Orchestrates fetching segment efforts and aggregating results."""

    def __init__(self, max_workers: int | None = None):
        """Initialize the service with thread pool size and caches."""
        self.max_workers = max_workers or MAX_WORKERS
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._log = logging.getLogger(self.__class__.__name__)
        self._activity_cache: TTLCache[_ActivityCacheKey, List[Dict[str, Any]]] = (
            TTLCache(
                maxsize=max(1, RUNNER_ACTIVITY_CACHE_SIZE),
                ttl=_ACTIVITY_CACHE_TTL_SECONDS,
            )
        )
        self._activity_cache_lock = threading.RLock()
        self._activity_scanner = ActivityEffortScanner(
            activity_provider=self._get_runner_activities,
            elapsed_adjuster=self._adjust_elapsed_for_birthday,
        )

    def process(
        self,
        segments: Sequence[Segment],
        runners: Sequence[Runner],
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> ResultsMapping:
        """Process all segments for all runners, returning aggregated results.

        Iterates through each segment, fetches efforts concurrently, and
        applies fallback strategies for runners without direct API access.
        """
        results: ResultsMapping = {}
        total_segments = len(segments)
        try:
            for seg_index, segment in enumerate(segments, start=1):
                if segment.start_date > segment.end_date:
                    self._log.warning(
                        "Skipping segment with inverted date range: %s (start=%s end=%s)",
                        segment.name,
                        segment.start_date,
                        segment.end_date,
                    )
                    continue
                segment_results = self._process_segment(
                    segment,
                    runners,
                    seg_index,
                    total_segments,
                    cancel_event,
                    progress,
                )
                for team_results in segment_results.values():
                    team_results.sort(key=lambda r: r.fastest_time)
                results[segment.name] = segment_results
        finally:
            self._clear_runner_activity_cache()
        return results

    def process_groups(
        self,
        segment_groups: Sequence[SegmentGroup],
        runners: Sequence[Runner],
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> ResultsMapping:
        """Process segment groups for all runners, returning aggregated results.

        When SEGMENT_SPLIT_WINDOWS_ENABLED is True, each runner's best time
        across all windows in a group is selected. When disabled, each window
        is processed as a separate segment.
        """
        results: ResultsMapping = {}
        total_groups = len(segment_groups)
        try:
            for group_index, group in enumerate(segment_groups, start=1):
                if cancel_event and cancel_event.is_set():
                    self._log.info("Cancellation requested; aborting processing.")
                    break

                if SEGMENT_SPLIT_WINDOWS_ENABLED:
                    # Aggregate best time across all windows
                    group_results = self._process_segment_group(
                        group,
                        runners,
                        group_index,
                        total_groups,
                        cancel_event,
                        progress,
                    )
                    for team_results in group_results.values():
                        team_results.sort(key=lambda r: r.fastest_time)
                    results[group.name] = group_results
                else:
                    # Disabled mode: process each window as separate segment
                    for window in group.windows:
                        sheet_name = self._get_window_sheet_name(group, window)
                        # Create a temporary Segment for compatibility
                        temp_segment = self._segment_from_group_window(group, window)
                        window_results = self._process_segment(
                            temp_segment,
                            runners,
                            group_index,
                            total_groups,
                            cancel_event,
                            progress,
                        )
                        for team_results in window_results.values():
                            team_results.sort(key=lambda r: r.fastest_time)
                        results[sheet_name] = window_results
        finally:
            self._clear_runner_activity_cache()
        return results

    def _get_window_sheet_name(self, group: SegmentGroup, window: SegmentWindow) -> str:
        """Generate sheet name for a window when split windows is disabled."""
        if len(group.windows) == 1:
            return group.name
        if window.label:
            return f"{group.name} - {window.label}"
        start_str = window.start_date.strftime("%Y-%m-%d")
        end_str = window.end_date.strftime("%Y-%m-%d")
        return f"{group.name} - {start_str} to {end_str}"

    def _segment_from_group_window(
        self, group: SegmentGroup, window: SegmentWindow
    ) -> Segment:
        """Create a temporary Segment from a SegmentGroup and SegmentWindow."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return Segment(
                id=group.id,
                name=group.name,
                start_date=window.start_date,
                end_date=window.end_date,
                default_time_seconds=group.default_time_seconds,
                min_distance_meters=group.min_distance_meters,
                birthday_bonus_seconds=window.birthday_bonus_seconds,
            )

    def _process_segment_group(
        self,
        group: SegmentGroup,
        runners: Sequence[Runner],
        group_index: int,
        total_groups: int,
        cancel_event: threading.Event | None,
        progress: Callable[[str, int, int], None] | None,
    ) -> Dict[str, List[SegmentResult]]:
        """Process a segment group with multiple windows, selecting best time."""
        segment_results: Dict[str, List[SegmentResult]] = {}
        eligible_runners = [r for r in runners if r.segment_team]
        total_runners = len(eligible_runners)

        if total_runners == 0:
            self._log.info("No eligible runners for segment group %s", group.name)
            return segment_results

        self._log.debug(
            "Processing segment group %s (%d/%d) with %d windows and %d runners",
            group.name,
            group_index,
            total_groups,
            len(group.windows),
            total_runners,
        )

        if cancel_event and cancel_event.is_set():
            return segment_results

        # Compute union date range for caching
        union_start = min(w.start_date for w in group.windows)
        union_end = max(w.end_date for w in group.windows)

        completed = 0

        def notify_progress(count: int) -> None:
            if progress is None:
                return
            try:
                progress(group.name, count, total_runners)
            except Exception:
                self._log.debug(
                    "Progress callback failed for group %s", group.name, exc_info=True
                )

        # Process each runner across all windows
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_runner: Dict[Future, Runner] = {}
            for runner in eligible_runners:
                if cancel_event and cancel_event.is_set():
                    break
                future = executor.submit(
                    self._process_runner_across_windows,
                    runner,
                    group,
                    union_start,
                    union_end,
                    cancel_event,
                )
                future_to_runner[future] = runner

            for fut in as_completed(future_to_runner):
                if cancel_event and cancel_event.is_set():
                    for pending in future_to_runner:
                        pending.cancel()
                    break
                runner = future_to_runner[fut]
                try:
                    result = fut.result()
                except Exception:  # noqa: BLE001
                    self._log.debug(
                        "Failed processing runner %s for group %s",
                        runner.name,
                        group.name,
                        exc_info=True,
                    )
                    result = None

                if result:
                    team = runner.segment_team
                    if team:
                        bucket = segment_results.setdefault(team, [])
                        bucket.append(result)

                completed += 1
                notify_progress(completed)

        # Inject default times for runners without results
        self._inject_default_group_results(group, eligible_runners, segment_results)
        return segment_results

    def _process_runner_across_windows(
        self,
        runner: Runner,
        group: SegmentGroup,
        union_start: datetime,
        union_end: datetime,
        cancel_event: threading.Event | None,
    ) -> SegmentResult | None:
        """Process a runner across all windows in a group, returning best result."""
        all_validated: List[Tuple[SegmentWindow, _ValidatedEffort]] = []
        total_attempts = 0

        for window in group.windows:
            if cancel_event and cancel_event.is_set():
                return None

            # Fetch efforts for this window
            # Try Strava API first if not payment_required
            efforts: List[dict] | None = None
            if not getattr(runner, "payment_required", False):
                try:
                    efforts = get_segment_efforts(
                        runner,
                        group.id,
                        window.start_date,
                        window.end_date,
                    )
                except Exception:  # noqa: BLE001
                    efforts = None

            # Process efforts for this window
            window_validated = self._validate_efforts_for_window(
                runner, group, window, efforts
            )
            total_attempts += len(window_validated)
            for v in window_validated:
                all_validated.append((window, v))

            # If no efforts from API, try activity scan fallback
            if not window_validated and USE_ACTIVITY_SCAN_FALLBACK:
                temp_segment = self._segment_from_group_window(group, window)
                scan_result = self._result_from_activity_scan(
                    runner, temp_segment, cancel_event
                )
                if scan_result:
                    # Convert scan result to validated effort for comparison
                    scan_validated = _ValidatedEffort(
                        adjusted_elapsed=scan_result.fastest_time,
                        raw_elapsed=scan_result.fastest_time,
                        start_date=scan_result.fastest_date,
                        effort={"id": scan_result.diagnostics.get("fastest_effort_id")},
                        birthday_bonus_applied=scan_result.birthday_bonus_applied,
                        distance_m=scan_result.fastest_distance_m,
                    )
                    all_validated.append((window, scan_validated))
                    total_attempts += scan_result.attempts

        if not all_validated:
            return None

        # Find the fastest across all windows
        all_validated.sort(key=lambda x: x[1].adjusted_elapsed)
        best_window, best_effort = all_validated[0]

        team = runner.segment_team or ""
        diagnostics: Dict[str, object] = {
            "source": "strava",
            "windows_processed": len(group.windows),
            "best_window_label": best_window.label,
            "best_window_start": best_window.start_date.isoformat(),
            "best_window_end": best_window.end_date.isoformat(),
            "best_effort_id": best_effort.effort.get("id"),
            "birthday_bonus_applied": best_effort.birthday_bonus_applied,
        }
        if best_effort.distance_m is not None:
            diagnostics["fastest_distance_m"] = best_effort.distance_m

        return SegmentResult(
            runner=runner.name,
            team=team,
            segment=group.name,
            attempts=total_attempts,
            fastest_time=best_effort.adjusted_elapsed,
            fastest_date=best_effort.start_date,
            birthday_bonus_applied=best_effort.birthday_bonus_applied,
            source="strava",
            diagnostics=diagnostics,
            fastest_distance_m=best_effort.distance_m,
        )

    def _validate_efforts_for_window(
        self,
        runner: Runner,
        group: SegmentGroup,
        window: SegmentWindow,
        efforts: List[dict] | None,
    ) -> List[_ValidatedEffort]:
        """Validate and filter efforts for a specific window."""
        if FORCE_ACTIVITY_SCAN_FALLBACK:
            return []

        if not efforts:
            return []

        min_distance = self._coerce_float(group.min_distance_meters) or 0.0
        valid: List[_ValidatedEffort] = []

        for effort in efforts:
            if not isinstance(effort, dict):
                continue
            elapsed = effort.get("elapsed_time")
            elapsed_f = self._coerce_float(elapsed)
            if elapsed_f is None or elapsed_f <= 0:
                continue

            # Check effort is within window bounds
            start_dt = parse_iso_datetime(
                effort.get("start_date_local") or effort.get("start_date")
            )
            if start_dt:
                # Normalize to naive datetime for comparison (window dates are naive)
                start_dt_naive = (
                    start_dt.replace(tzinfo=None) if start_dt.tzinfo else start_dt
                )
                window_start_naive = (
                    window.start_date.replace(tzinfo=None)
                    if hasattr(window.start_date, "tzinfo") and window.start_date.tzinfo
                    else window.start_date
                )
                window_end_naive = (
                    window.end_date.replace(tzinfo=None)
                    if hasattr(window.end_date, "tzinfo") and window.end_date.tzinfo
                    else window.end_date
                )
                if (
                    start_dt_naive < window_start_naive
                    or start_dt_naive > window_end_naive
                ):
                    continue

            effort_distance = derive_effort_distance_m(
                runner,
                effort,
                allow_stream=min_distance > 0,
            )
            if min_distance > 0:
                if effort_distance is None or effort_distance < min_distance:
                    continue

            # Apply window-specific birthday bonus
            adjusted_elapsed, bonus_applied = self._adjust_elapsed_for_birthday_window(
                runner,
                window.birthday_bonus_seconds,
                elapsed_f,
                start_dt,
            )
            valid.append(
                _ValidatedEffort(
                    adjusted_elapsed=adjusted_elapsed,
                    raw_elapsed=elapsed_f,
                    start_date=start_dt,
                    effort=effort,
                    birthday_bonus_applied=bonus_applied,
                    distance_m=effort_distance,
                )
            )

        return valid

    def _adjust_elapsed_for_birthday_window(
        self,
        runner: Runner,
        birthday_bonus_seconds: float,
        elapsed_seconds: float,
        effort_date: datetime | None,
    ) -> tuple[float, bool]:
        """Apply birthday bonus deduction using window-specific bonus."""
        if birthday_bonus_seconds <= 0 or not effort_date or not runner.birthday:
            return float(elapsed_seconds), False
        month, day = runner.birthday
        if effort_date.month != month or effort_date.day != day:
            return float(elapsed_seconds), False
        adjusted = max(0.0, float(elapsed_seconds) - birthday_bonus_seconds)
        return adjusted, True

    def _inject_default_group_results(
        self,
        group: SegmentGroup,
        runners: Sequence[Runner],
        segment_results: Dict[str, List[SegmentResult]],
    ) -> None:
        """Add default_time placeholder results for runners without any effort."""
        default_time = group.default_time_seconds
        if default_time is None:
            return
        for runner in runners:
            team = runner.segment_team
            if not team:
                continue
            bucket = segment_results.setdefault(team, [])
            if any(result.runner == runner.name for result in bucket):
                continue
            # Use the first window's start date for default
            default_date = group.windows[0].start_date if group.windows else None
            bucket.append(
                SegmentResult(
                    runner=runner.name,
                    team=team,
                    segment=group.name,
                    attempts=0,
                    fastest_time=default_time,
                    fastest_date=default_date,
                    source="default_time",
                    diagnostics={
                        "reason": "default_time_applied",
                        "segment_id": group.id,
                    },
                    fastest_distance_m=0.0,
                )
            )

    def _process_segment(
        self,
        segment: Segment,
        runners: Sequence[Runner],
        seg_index: int,
        total_segments: int,
        cancel_event: threading.Event | None,
        progress: Callable[[str, int, int], None] | None,
    ) -> Dict[str, List[SegmentResult]]:
        """Process a single segment: fetch efforts, run fallbacks, inject defaults."""
        segment_results: Dict[str, List[SegmentResult]] = {}
        eligible_runners = [r for r in runners if r.segment_team]
        total_runners = len(eligible_runners)
        if total_runners == 0:
            self._log.info("No eligible runners for segment %s", segment.name)
            return segment_results
        self._log.debug(
            "Processing segment %s (%d/%d) with %d runners (max_workers=%d)",
            segment.name,
            seg_index,
            total_segments,
            total_runners,
            self.max_workers,
        )
        if cancel_event and cancel_event.is_set():
            self._log.info(
                "Cancellation requested before segment %s; aborting.", segment.name
            )
            return segment_results
        completed = 0

        def notify_progress(count: int) -> None:
            if progress is None:
                return
            try:
                progress(segment.name, count, total_runners)
            except Exception:
                self._log.debug(
                    "Progress callback failed for segment %s",
                    segment.name,
                    exc_info=True,
                )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_runner, fallback_queue = self._submit_effort_futures(
                executor,
                eligible_runners,
                segment,
                cancel_event,
            )
            completed = self._consume_effort_futures(
                future_to_runner,
                fallback_queue,
                segment,
                segment_results,
                cancel_event,
                completed,
                notify_progress,
            )
        completed = self._process_fallback_queue(
            segment,
            segment_results,
            fallback_queue,
            cancel_event,
            completed,
            notify_progress,
        )
        self._inject_default_segment_results(segment, eligible_runners, segment_results)
        return segment_results

    def _submit_effort_futures(
        self,
        executor: ThreadPoolExecutor,
        runners: Sequence[Runner],
        segment: Segment,
        cancel_event: threading.Event | None,
    ) -> Tuple[Dict[Future, Runner], List[Runner]]:
        """Submit concurrent Strava API calls for each runner's segment efforts.

        Runners marked as payment_required skip the API and go directly to
        the fallback queue for activity-scan processing.
        """
        future_to_runner: Dict[Future, Runner] = {}
        fallback_queue: List[Runner] = []
        for runner in runners:
            if cancel_event and cancel_event.is_set():
                break
            if getattr(runner, "payment_required", False):
                fallback_queue.append(runner)
                continue
            future = executor.submit(
                get_segment_efforts,
                runner,
                segment.id,
                segment.start_date,
                segment.end_date,
            )
            future_to_runner[future] = runner
        return future_to_runner, fallback_queue

    def _consume_effort_futures(
        self,
        future_to_runner: Dict[Future, Runner],
        fallback_queue: List[Runner],
        segment: Segment,
        segment_results: Dict[str, List[SegmentResult]],
        cancel_event: threading.Event | None,
        completed: int,
        notify_progress: Callable[[int], None],
    ) -> int:
        """Collect completed effort futures and process results.

        Runners that fail or return no efforts are added to the fallback
        queue for activity-scan retry. Cancels pending futures if cancelled.
        """
        for fut in as_completed(future_to_runner):
            if cancel_event and cancel_event.is_set():
                self._log.info(
                    "Cancellation requested during segment %s; stopping remaining futures.",
                    segment.name,
                )
                # Cancel any pending futures to stop further Strava requests
                for pending in future_to_runner:
                    pending.cancel()
                break
            runner = future_to_runner[fut]
            try:
                efforts = fut.result()
            except Exception:  # noqa: BLE001
                efforts = None
            seg_result = self._process_runner_results(
                runner,
                segment,
                efforts,
                cancel_event,
            )
            if seg_result:
                team = runner.segment_team
                if team:
                    bucket = segment_results.setdefault(team, [])
                    bucket.append(seg_result)
            if (
                seg_result is None
                and getattr(runner, "payment_required", False)
                and runner not in fallback_queue
            ):
                fallback_queue.append(runner)
            completed += 1
            notify_progress(completed)
        return completed

    def _process_fallback_queue(
        self,
        segment: Segment,
        segment_results: Dict[str, List[SegmentResult]],
        fallback_queue: List[Runner],
        cancel_event: threading.Event | None,
        completed: int,
        notify_progress: Callable[[int], None],
    ) -> int:
        """Process runners via activity-scan fallback when API access failed."""
        unique_runners = self._deduplicate_fallback_runners(fallback_queue)
        if not unique_runners:
            return completed

        if cancel_event and cancel_event.is_set():
            self._log.info(
                "Cancellation requested before fallback execution for segment %s; skipping remaining runners.",
                segment.name,
            )
            return completed

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_runner = self._submit_fallback_tasks(
                executor,
                unique_runners,
                segment,
                cancel_event,
            )
            completed = self._consume_fallback_results(
                future_to_runner,
                segment,
                segment_results,
                cancel_event,
                completed,
                notify_progress,
            )

        return completed

    def _deduplicate_fallback_runners(
        self, fallback_queue: Sequence[Runner]
    ) -> List[Runner]:
        """Remove duplicate runners from fallback queue by strava_id."""
        unique: List[Runner] = []
        seen: Set[int | str] = set()
        for runner in fallback_queue:
            if runner.strava_id in seen:
                continue
            seen.add(runner.strava_id)
            unique.append(runner)
        return unique

    def _submit_fallback_tasks(
        self,
        executor: ThreadPoolExecutor,
        runners: Sequence[Runner],
        segment: Segment,
        cancel_event: threading.Event | None,
    ) -> Dict[Future, Runner]:
        """Submit activity-scan fallback tasks for runners without API results."""
        future_to_runner: Dict[Future, Runner] = {}
        for runner in runners:
            if cancel_event and cancel_event.is_set():
                self._log.info(
                    "Cancellation requested during submission for segment %s; halting fallback queue.",
                    segment.name,
                )
                break
            future = executor.submit(
                self._process_runner_results,
                runner,
                segment,
                None,
                cancel_event,
            )
            future_to_runner[future] = runner
        return future_to_runner

    def _consume_fallback_results(
        self,
        future_to_runner: Dict[Future, Runner],
        segment: Segment,
        segment_results: Dict[str, List[SegmentResult]],
        cancel_event: threading.Event | None,
        completed: int,
        notify_progress: Callable[[int], None],
    ) -> int:
        """Collect fallback results and add to segment_results. Cancels on event."""
        for fut in as_completed(future_to_runner):
            runner = future_to_runner[fut]
            result: SegmentResult | None = None
            try:
                result = fut.result()
            except Exception:  # noqa: BLE001
                self._log.debug(
                    "Fallback matcher failed runner=%s segment=%s",
                    runner.name,
                    segment.id,
                    exc_info=True,
                )
            if result:
                team = runner.segment_team
                if team:
                    bucket = segment_results.setdefault(team, [])
                    bucket.append(result)
            completed += 1
            notify_progress(completed)
            if cancel_event and cancel_event.is_set():
                self._log.info(
                    "Cancellation requested while processing segment %s fallback results; stopping early.",
                    segment.name,
                )
                # Cancel any pending futures to stop further processing
                for pending in future_to_runner:
                    pending.cancel()
                break
        return completed

    def _inject_default_segment_results(
        self,
        segment: Segment,
        runners: Sequence[Runner],
        segment_results: Dict[str, List[SegmentResult]],
    ) -> None:
        """Add default_time placeholder results for runners without any effort."""
        default_time = segment.default_time_seconds
        if default_time is None:
            return
        for runner in runners:
            team = runner.segment_team
            if not team:
                continue
            bucket = segment_results.setdefault(team, [])
            if any(result.runner == runner.name for result in bucket):
                continue
            bucket.append(
                SegmentResult(
                    runner=runner.name,
                    team=team,
                    segment=segment.name,
                    attempts=0,
                    fastest_time=default_time,
                    fastest_date=segment.start_date,
                    source="default_time",
                    diagnostics={
                        "reason": "default_time_applied",
                        "segment_id": segment.id,
                    },
                    fastest_distance_m=0.0,
                )
            )

    def _process_runner_results(
        self,
        runner: Runner,
        segment: Segment,
        efforts: List[dict] | None,
        cancel_event: threading.Event | None = None,
    ) -> SegmentResult | None:
        """Convert raw efforts to SegmentResult, falling back to activity scan."""
        seg_result = self._result_from_efforts(runner, segment, efforts)
        if seg_result is not None:
            return seg_result
        if USE_ACTIVITY_SCAN_FALLBACK:
            scan_result = self._result_from_activity_scan(
                runner,
                segment,
                cancel_event,
            )
            if scan_result is not None:
                return scan_result
        return None

    def _result_from_activity_scan(
        self,
        runner: Runner,
        segment: Segment,
        cancel_event: threading.Event | None,
    ) -> SegmentResult | None:
        """Scan runner's activities for segment efforts when API unavailable."""
        try:
            scan = self._activity_scanner.scan_segment(
                runner,
                segment,
                cancel_event=cancel_event,
            )
        except StravaAPIError as exc:
            self._log.warning(
                "Activity scan failed runner=%s segment=%s: %s",
                runner.name,
                segment.id,
                exc,
            )
            return None
        if scan is None:
            return None
        team = runner.segment_team or ""
        diagnostics: Dict[str, Any] = {
            "source": "activity_scan",
            "effort_ids": scan.effort_ids,
            "inspected_activities": scan.inspected_activities,
            "fastest_activity_id": scan.fastest_activity_id,
            "fastest_effort_id": scan.fastest_effort_id,
            "moving_time": scan.moving_time,
            "birthday_bonus_applied": scan.birthday_bonus_applied,
        }
        if scan.filtered_efforts_below_distance:
            diagnostics["filtered_efforts_below_distance"] = (
                scan.filtered_efforts_below_distance
            )
        attempts = scan.attempts if scan.attempts > 0 else 1
        return SegmentResult(
            runner=runner.name,
            team=team,
            segment=segment.name,
            attempts=attempts,
            fastest_time=scan.fastest_elapsed,
            fastest_date=scan.fastest_start_date,
            birthday_bonus_applied=scan.birthday_bonus_applied,
            source="activity_scan",
            diagnostics=diagnostics,
            fastest_distance_m=scan.fastest_distance_m,
        )

    def _result_from_efforts(
        self,
        runner: Runner,
        segment: Segment,
        efforts: List[dict] | None,
    ) -> SegmentResult | None:
        """Convert Strava segment efforts into a SegmentResult if available."""

        if FORCE_ACTIVITY_SCAN_FALLBACK:
            return None

        if not efforts:
            return None

        min_distance = self._coerce_float(segment.min_distance_meters) or 0.0
        filtered_by_distance = 0
        valid: List[_ValidatedEffort] = []
        for effort in efforts:
            if not isinstance(effort, dict):
                continue
            elapsed = effort.get("elapsed_time")
            elapsed_f = self._coerce_float(elapsed)
            if elapsed_f is None or elapsed_f <= 0:
                continue
            effort_distance = derive_effort_distance_m(
                runner,
                effort,
                allow_stream=min_distance > 0,
            )
            if min_distance > 0:
                if effort_distance is None or effort_distance < min_distance:
                    filtered_by_distance += 1
                    continue
            start_dt = parse_iso_datetime(
                effort.get("start_date_local") or effort.get("start_date")
            )
            adjusted_elapsed, bonus_applied = self._adjust_elapsed_for_birthday(
                runner,
                segment,
                elapsed_f,
                start_dt,
            )
            valid.append(
                _ValidatedEffort(
                    adjusted_elapsed=adjusted_elapsed,
                    raw_elapsed=elapsed_f,
                    start_date=start_dt,
                    effort=effort,
                    birthday_bonus_applied=bonus_applied,
                    distance_m=effort_distance,
                )
            )

        if not valid:
            return None

        valid.sort(key=lambda v: v.adjusted_elapsed)
        fastest = valid[0]

        fastest_distance_m = fastest.distance_m
        if min_distance <= 0:
            precise_distance = derive_effort_distance_m(
                runner,
                fastest.effort,
                allow_stream=True,
            )
            if precise_distance is not None:
                fastest_distance_m = precise_distance

        diagnostics: Dict[str, object] = {
            "source": "strava",
            "effort_ids": [
                v.effort.get("id")
                for v in valid
                if isinstance(v.effort.get("id"), (int, str))
            ],
            "best_effort_id": fastest.effort.get("id"),
            "moving_time": fastest.effort.get("moving_time"),
            "birthday_bonus_applied": fastest.birthday_bonus_applied,
        }
        if fastest_distance_m is not None:
            diagnostics["fastest_distance_m"] = fastest_distance_m
        if filtered_by_distance:
            diagnostics["filtered_efforts_below_distance"] = filtered_by_distance

        attempts = len(valid)
        team = runner.segment_team or ""
        return SegmentResult(
            runner=runner.name,
            team=team,
            segment=segment.name,
            attempts=attempts,
            fastest_time=fastest.adjusted_elapsed,
            fastest_date=fastest.start_date,
            birthday_bonus_applied=fastest.birthday_bonus_applied,
            source="strava",
            diagnostics=diagnostics,
            fastest_distance_m=fastest_distance_m,
        )

    def _get_runner_activities(
        self,
        runner: Runner,
        start_date: datetime,
        end_date: datetime,
        cancel_event: threading.Event | None = None,
    ) -> List[Dict[str, Any]]:
        """Return cached activities for a runner/date window, fetching if needed."""

        if cancel_event and cancel_event.is_set():
            return []
        cache_key = (runner.strava_id, start_date, end_date)
        with self._activity_cache_lock:
            cached: List[Dict[str, Any]] | None = self._activity_cache.get(cache_key)
        if cached is not None:
            self._log.debug(
                "Using cached activities runner=%s window=%s->%s",
                runner.name,
                start_date,
                end_date,
            )
            return cached

        activities = get_activities(runner, start_date, end_date) or []
        with self._activity_cache_lock:
            self._activity_cache[cache_key] = activities
        return activities

    def _clear_runner_activity_cache(self) -> None:
        """Clear the per-runner activity cache between processing batches."""

        with self._activity_cache_lock:
            self._activity_cache.clear()

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        """Safely convert a value to float, returning None on failure."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _adjust_elapsed_for_birthday(
        self,
        runner: Runner,
        segment: Segment,
        elapsed_seconds: float,
        effort_date: datetime | None,
    ) -> tuple[float, bool]:
        """Apply birthday bonus deduction if effort occurred on runner's birthday."""
        bonus = segment.birthday_bonus_seconds or 0.0
        if bonus <= 0 or not effort_date or not runner.birthday:
            return float(elapsed_seconds), False
        month, day = runner.birthday
        if effort_date.month != month or effort_date.day != day:
            return float(elapsed_seconds), False
        adjusted = max(0.0, float(elapsed_seconds) - bonus)
        return adjusted, True


__all__ = ["SegmentService"]
