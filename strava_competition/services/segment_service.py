"""Segment competition service (application layer).

Orchestrates scanning runner activities for segment efforts and
aggregating results so higher-level code (main, CLI, etc.) depends
on a stable service API rather than a free function implementation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
import logging
import math
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Sequence, Tuple

from cachetools import TTLCache

from ..activity_scan import ActivityEffortScanner
from ..config import (
    RUNNER_ACTIVITY_CACHE_SIZE,
    MAX_WORKERS,
    SEGMENT_SPLIT_WINDOWS_ENABLED,
)
from ..errors import StravaAPIError, StravaRateLimitError
from ..models import Segment, Runner, SegmentResult, SegmentGroup, SegmentWindow
from ..strava_api import get_activities
from ..utils import parse_iso_datetime, to_utc_aware

ResultsMapping = Dict[str, Dict[str, List[SegmentResult]]]

_ActivityCacheKey = Tuple[int | str, datetime, datetime]

# Cache TTL for runner activities (1 hour)
_ACTIVITY_CACHE_TTL_SECONDS = 3600

# Max times to retry all rate-limited runners for a segment before giving up
_RATE_LIMIT_RUNNER_RETRIES = 3
# Cooldown multiplier between runner-level retry rounds (seconds)
_RATE_LIMIT_RUNNER_COOLDOWN = 60


@dataclass(slots=True)
class _ScanOutcome:
    """Bookkeeping for one segment/group scan across retry rounds."""

    scanned: set[str] = field(default_factory=set)
    failed: list[str] = field(default_factory=list)
    rate_limited: list[Runner] = field(default_factory=list)
    cancelled: bool = False


def _sort_team_results(segment_results: Dict[str, List[SegmentResult]]) -> None:
    """Sort each team's results by fastest time (missing times last)."""
    for team_results in segment_results.values():
        team_results.sort(
            key=lambda r: r.fastest_time if r.fastest_time is not None else float("inf")
        )


def _activities_in_window(
    activities: Sequence[Dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
) -> List[Dict[str, Any]]:
    """Filter activities to those starting within ``[start_date, end_date]``.

    Mirrors the UTC comparison convention of the activities listing filter
    (prefer ``start_date`` which is true UTC, fall back to
    ``start_date_local``). Activities without a parseable timestamp are
    kept: the scanner's per-effort window check remains authoritative, so
    keeping them can only cost an extra detail inspection, never produce a
    wrong result.
    """
    start_utc = to_utc_aware(start_date)
    end_utc = to_utc_aware(end_date)
    filtered: List[Dict[str, Any]] = []
    for act in activities:
        raw_start = act.get("start_date") or act.get("start_date_local")
        parsed = parse_iso_datetime(raw_start) if isinstance(raw_start, str) else None
        if parsed is not None and not (start_utc <= to_utc_aware(parsed) <= end_utc):
            continue
        filtered.append(act)
    return filtered


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
        # Per-thread marker for activity-listing fetch failures; the scanner
        # swallows provider exceptions, so the failure is re-raised from here.
        self._fetch_failures = threading.local()
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

        Iterates through each segment, scans activities concurrently,
        and aggregates the best effort per runner.
        """
        results: ResultsMapping = {}
        total_segments = len(segments)
        try:
            for seg_index, segment in enumerate(segments, start=1):
                if cancel_event is not None and cancel_event.is_set():
                    self._log.info(
                        "Cancellation requested; aborting segment processing."
                    )
                    break
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
                _sort_team_results(segment_results)
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
                    _sort_team_results(group_results)
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
                        _sort_team_results(window_results)
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
        return Segment(
            id=group.id,
            name=group.name,
            start_date=window.start_date,
            end_date=window.end_date,
            default_time_seconds=group.default_time_seconds,
            min_distance_meters=group.min_distance_meters,
            birthday_bonus_seconds=window.birthday_bonus_seconds,
            time_bonus_seconds=window.time_bonus_seconds,
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

        if cancel_event is not None and cancel_event.is_set():
            return segment_results

        # Union date range: one listing fetch per runner covers every window.
        union_start = min(w.start_date for w in group.windows)
        union_end = max(w.end_date for w in group.windows)
        label = f"group {group.name}"

        def scan_task(runner: Runner) -> SegmentResult | None:
            return self._process_runner_across_windows(
                runner, group, union_start, union_end, cancel_event
            )

        outcome = self._scan_runners_with_retries(
            label,
            eligible_runners,
            scan_task,
            segment_results,
            cancel_event,
            self._make_progress_notifier(progress, group.name, total_runners),
        )
        self._log_scan_failures(label, outcome)
        if outcome.cancelled:
            return segment_results

        # Only assign default_time to runners we successfully scanned:
        # rate-limited, failed, or cancelled runners were never confirmed
        # to have no effort, so penalising them would be unfair.
        scanned_runners = [r for r in eligible_runners if r.name in outcome.scanned]
        self._inject_default_group_results(group, scanned_runners, segment_results)
        return segment_results

    def _process_runner_across_windows(
        self,
        runner: Runner,
        group: SegmentGroup,
        union_start: datetime,
        union_end: datetime,
        cancel_event: threading.Event | None,
    ) -> SegmentResult | None:
        """Process a runner across all windows in a group, returning best result.

        Fetches the runner's activities once for the union window and filters
        them in memory per window, seeding the activity cache so the scanner
        does not refetch each window.

        Raises:
            StravaAPIError: If the union activity fetch fails.
        """
        union_activities = self._get_runner_activities(runner, union_start, union_end)
        best_result: SegmentResult | None = None
        total_attempts = 0

        for window in group.windows:
            if cancel_event is not None and cancel_event.is_set():
                return None

            self._seed_window_activities(runner, window, union_activities)
            temp_segment = self._segment_from_group_window(group, window)
            scan_result = self._result_from_activity_scan(
                runner,
                temp_segment,
                cancel_event,
            )
            if scan_result is None:
                continue
            total_attempts += scan_result.attempts
            if scan_result.fastest_time is None:
                continue

            best_time = (
                best_result.fastest_time
                if best_result is not None and best_result.fastest_time is not None
                else float("inf")
            )
            if best_result is None or scan_result.fastest_time < best_time:
                best_result = self._build_group_result(
                    runner, group, window, scan_result
                )

        if best_result is not None:
            best_result.attempts = total_attempts
        return best_result

    def _seed_window_activities(
        self,
        runner: Runner,
        window: SegmentWindow,
        union_activities: List[Dict[str, Any]],
    ) -> None:
        """Pre-populate the activity cache for a window from the union fetch."""
        window_activities = _activities_in_window(
            union_activities, window.start_date, window.end_date
        )
        cache_key = (runner.strava_id, window.start_date, window.end_date)
        with self._activity_cache_lock:
            self._activity_cache[cache_key] = window_activities

    def _build_group_result(
        self,
        runner: Runner,
        group: SegmentGroup,
        window: SegmentWindow,
        scan_result: SegmentResult,
    ) -> SegmentResult:
        """Build a group-level result from the best window's scan result."""
        diagnostics: Dict[str, Any] = {
            "source": "activity_scan",
            "windows_processed": len(group.windows),
            "best_window_label": window.label,
            "best_window_start": window.start_date.isoformat(),
            "best_window_end": window.end_date.isoformat(),
            "best_effort_id": scan_result.diagnostics.get("fastest_effort_id"),
            "birthday_bonus_applied": scan_result.birthday_bonus_applied,
            "time_bonus_applied": scan_result.time_bonus_applied,
        }
        if scan_result.fastest_distance_m is not None:
            diagnostics["fastest_distance_m"] = scan_result.fastest_distance_m

        return SegmentResult(
            runner=runner.name,
            team=runner.segment_team or "",
            segment=group.name,
            attempts=scan_result.attempts,
            fastest_time=scan_result.fastest_time,
            fastest_date=scan_result.fastest_date,
            birthday_bonus_applied=scan_result.birthday_bonus_applied,
            time_bonus_applied=scan_result.time_bonus_applied,
            source="activity_scan",
            diagnostics=diagnostics,
            fastest_distance_m=scan_result.fastest_distance_m,
        )

    def _apply_time_bonus(
        self,
        time_bonus_seconds: float,
        elapsed_seconds: float,
    ) -> tuple[float, bool]:
        """Apply a time bonus adjustment to an elapsed time.

        Positive values subtract time (reward), negative values add time
        (penalty). Returns (adjusted_time, was_applied).
        """
        if math.isclose(time_bonus_seconds, 0.0, abs_tol=1e-9):
            return elapsed_seconds, False
        adjusted = elapsed_seconds - time_bonus_seconds
        return max(0.0, adjusted), True

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
        """Process a single segment: scan activities, inject defaults."""
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
        if cancel_event is not None and cancel_event.is_set():
            self._log.info(
                "Cancellation requested before segment %s; aborting.", segment.name
            )
            return segment_results

        label = f"segment {segment.name}"

        def scan_task(runner: Runner) -> SegmentResult | None:
            return self._result_from_activity_scan(runner, segment, cancel_event)

        outcome = self._scan_runners_with_retries(
            label,
            eligible_runners,
            scan_task,
            segment_results,
            cancel_event,
            self._make_progress_notifier(progress, segment.name, total_runners),
        )
        self._log_scan_failures(label, outcome)
        if outcome.cancelled:
            return segment_results

        # Only assign default_time to runners we successfully scanned:
        # rate-limited, failed, or cancelled runners were never confirmed
        # to have no effort, so penalising them would be unfair.
        scanned_runners = [r for r in eligible_runners if r.name in outcome.scanned]
        self._inject_default_segment_results(segment, scanned_runners, segment_results)
        return segment_results

    def _make_progress_notifier(
        self,
        progress: Callable[[str, int, int], None] | None,
        name: str,
        total_runners: int,
    ) -> Callable[[int], None]:
        """Wrap the optional progress callback so its failures never break a scan."""

        def notify(count: int) -> None:
            if progress is None:
                return
            try:
                progress(name, count, total_runners)
            except Exception:
                self._log.debug("Progress callback failed for %s", name, exc_info=True)

        return notify

    def _scan_runners_with_retries(
        self,
        label: str,
        runners: Sequence[Runner],
        scan_task: Callable[[Runner], SegmentResult | None],
        segment_results: Dict[str, List[SegmentResult]],
        cancel_event: threading.Event | None,
        notify_progress: Callable[[int], None],
    ) -> _ScanOutcome:
        """Scan runners concurrently, retrying rate-limited ones after cooldowns.

        Returns:
            Bookkeeping of scanned/failed/rate-limited runners; ``cancelled``
            is True when processing stopped early because of the cancel event.
        """
        outcome = _ScanOutcome()
        pending = list(runners)
        completed = 0
        for retry_round in range(_RATE_LIMIT_RUNNER_RETRIES + 1):
            if not pending or outcome.cancelled:
                break
            if cancel_event is not None and cancel_event.is_set():
                outcome.cancelled = True
                break
            if retry_round > 0 and self._cooldown_interrupted(
                retry_round, len(pending), label, cancel_event
            ):
                outcome.cancelled = True
                break
            outcome.rate_limited = []
            completed = self._run_scan_round(
                label,
                pending,
                scan_task,
                segment_results,
                cancel_event,
                notify_progress,
                outcome,
                completed,
            )
            pending = list(outcome.rate_limited)
        outcome.rate_limited = pending
        if outcome.cancelled:
            self._log.info(
                "Cancellation requested while processing %s; stopping early.", label
            )
        return outcome

    def _cooldown_interrupted(
        self,
        retry_round: int,
        pending_count: int,
        label: str,
        cancel_event: threading.Event | None,
    ) -> bool:
        """Wait out the retry cooldown; True when interrupted by cancellation."""
        cooldown = _RATE_LIMIT_RUNNER_COOLDOWN * retry_round
        self._log.warning(
            "Retrying %d rate-limited runners for %s (round %d/%d) after %ds cooldown",
            pending_count,
            label,
            retry_round,
            _RATE_LIMIT_RUNNER_RETRIES,
            cooldown,
        )
        if cancel_event is not None:
            return cancel_event.wait(cooldown)
        time.sleep(cooldown)
        return False

    def _run_scan_round(
        self,
        label: str,
        pending: Sequence[Runner],
        scan_task: Callable[[Runner], SegmentResult | None],
        segment_results: Dict[str, List[SegmentResult]],
        cancel_event: threading.Event | None,
        notify_progress: Callable[[int], None],
        outcome: _ScanOutcome,
        completed: int,
    ) -> int:
        """Run one concurrent scan round, recording per-runner outcomes."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_runner: Dict[Future[SegmentResult | None], Runner] = {}
            for runner in pending:
                if cancel_event is not None and cancel_event.is_set():
                    outcome.cancelled = True
                    break
                future_to_runner[executor.submit(scan_task, runner)] = runner

            for fut in as_completed(future_to_runner):
                runner = future_to_runner[fut]
                result = self._collect_scan_result(fut, runner, label, outcome)
                if result is not None and runner.segment_team:
                    segment_results.setdefault(runner.segment_team, []).append(result)
                completed += 1
                notify_progress(completed)
                if cancel_event is not None and cancel_event.is_set():
                    outcome.cancelled = True
                    for pending_future in future_to_runner:
                        pending_future.cancel()
                    break
        return completed

    def _collect_scan_result(
        self,
        future: Future[SegmentResult | None],
        runner: Runner,
        label: str,
        outcome: _ScanOutcome,
    ) -> SegmentResult | None:
        """Harvest one scan future, classifying rate-limit vs generic failure."""
        try:
            result = future.result()
        except StravaRateLimitError:
            self._log.warning(
                "Rate-limited runner=%s %s; queued for retry", runner.name, label
            )
            outcome.rate_limited.append(runner)
            return None
        except Exception:  # noqa: BLE001
            self._log.warning(
                "Activity scan failed runner=%s %s", runner.name, label, exc_info=True
            )
            outcome.failed.append(runner.name)
            return None
        outcome.scanned.add(runner.name)
        return result

    def _log_scan_failures(self, label: str, outcome: _ScanOutcome) -> None:
        """Emit end-of-scan summaries for rate-limited and failed runners."""
        if outcome.rate_limited:
            self._log.error(
                "Gave up on %d rate-limited runners for %s after %d retries: %s",
                len(outcome.rate_limited),
                label,
                _RATE_LIMIT_RUNNER_RETRIES,
                ", ".join(r.name for r in outcome.rate_limited),
            )
        if outcome.failed:
            self._log.warning(
                "Scan failed for %d runners on %s (no default time applied): %s",
                len(outcome.failed),
                label,
                ", ".join(sorted(outcome.failed)),
            )

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

    def _result_from_activity_scan(
        self,
        runner: Runner,
        segment: Segment,
        cancel_event: threading.Event | None,
    ) -> SegmentResult | None:
        """Scan runner's activities for segment efforts and return the best result.

        Applies the segment's time bonus (if any) to the fastest time so both
        the split-window and single-segment paths score identically.

        Raises:
            StravaRateLimitError: Propagated so callers can retry
                the runner after a cooldown instead of losing data.
            StravaAPIError: When the runner's activity listing could not be
                fetched. The scanner swallows provider exceptions and
                degrades to an empty scan, so the failure is re-raised here
                from the per-thread marker to keep it distinguishable from a
                genuine "no activities" result.
        """
        self._fetch_failures.error = None
        try:
            scan = self._activity_scanner.scan_segment(
                runner,
                segment,
                cancel_event=cancel_event,
            )
        except StravaRateLimitError:
            raise
        except StravaAPIError as exc:
            self._log.warning(
                "Activity scan failed runner=%s segment=%s: %s",
                runner.name,
                segment.id,
                exc,
            )
            return None
        self._raise_pending_fetch_failure()
        if scan is None:
            return None

        fastest_time = scan.fastest_elapsed
        time_bonus_applied = scan.time_bonus_applied
        if fastest_time is not None:
            fastest_time, bonus_applied_now = self._apply_time_bonus(
                segment.time_bonus_seconds, fastest_time
            )
            time_bonus_applied = time_bonus_applied or bonus_applied_now

        team = runner.segment_team or ""
        diagnostics: Dict[str, Any] = {
            "source": "activity_scan",
            "effort_ids": scan.effort_ids,
            "inspected_activities": scan.inspected_activities,
            "fastest_activity_id": scan.fastest_activity_id,
            "fastest_effort_id": scan.fastest_effort_id,
            "moving_time": scan.moving_time,
            "birthday_bonus_applied": scan.birthday_bonus_applied,
            "time_bonus_applied": time_bonus_applied,
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
            fastest_time=fastest_time,
            fastest_date=scan.fastest_start_date,
            birthday_bonus_applied=scan.birthday_bonus_applied,
            time_bonus_applied=time_bonus_applied,
            source="activity_scan",
            diagnostics=diagnostics,
            fastest_distance_m=scan.fastest_distance_m,
        )

    def _raise_pending_fetch_failure(self) -> None:
        """Re-raise an activity fetch failure recorded on this thread.

        Raises:
            StravaAPIError: The failure recorded by ``_get_runner_activities``.
        """
        error = getattr(self._fetch_failures, "error", None)
        if error is not None:
            self._fetch_failures.error = None
            raise error

    def _get_runner_activities(
        self,
        runner: Runner,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Return cached activities for a runner/date window, fetching if needed.

        Raises:
            StravaAPIError: If the fetch failed (``get_activities`` returned
                ``None``). Failures are never cached, and the error is also
                recorded on a per-thread marker so callers that receive the
                scanner's degraded empty result can still surface it.
        """
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

        activities = get_activities(runner, start_date, end_date)
        if activities is None:
            error = StravaAPIError(
                f"Activity fetch failed for runner {runner.name} "
                f"({start_date} -> {end_date})"
            )
            self._fetch_failures.error = error
            raise error
        with self._activity_cache_lock:
            self._activity_cache[cache_key] = activities
        return activities

    def _clear_runner_activity_cache(self) -> None:
        """Clear the per-runner activity cache between processing batches."""

        with self._activity_cache_lock:
            self._activity_cache.clear()

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
