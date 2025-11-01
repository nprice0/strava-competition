"""Segment competition service (application layer).

Encapsulates orchestration of fetching efforts and aggregating results so
higher-level code (main, CLI, etc.) depends on a stable service API rather
than a free function implementation.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import logging
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Sequence, Tuple, Set, Optional, Iterator

from ..models import Segment, Runner, SegmentResult
from ..config import (
    MATCHING_ACTIVITY_MIN_DISTANCE_RATIO,
    MATCHING_ACTIVITY_REQUIRED_TYPES,
    MATCHING_FALLBACK_ENABLED,
    MATCHING_RUNNER_ACTIVITY_CACHE_SIZE,
    MAX_WORKERS,
)
from ..matching.similarity import segment_cache_scope
from ..matching.models import MatchResult
from ..matching.fetchers import fetch_segment_geometry
from ..strava_api import get_segment_efforts, get_activities
from ..matching import match_activity_to_segment

ResultsMapping = Dict[str, Dict[str, List[SegmentResult]]]


class SegmentService:
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or MAX_WORKERS
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._log = logging.getLogger(self.__class__.__name__)
        self._segment_distance_cache: Dict[int, float] = {}
        self._segment_distance_lock = threading.RLock()
        self._activity_cache_max_entries = max(0, MATCHING_RUNNER_ACTIVITY_CACHE_SIZE)
        self._activity_cache: "OrderedDict[tuple[str, datetime, datetime], List[Dict[str, Any]]]" = OrderedDict()
        self._activity_cache_lock = threading.RLock()

    def process(
        self,
        segments: Sequence[Segment],
        runners: Sequence[Runner],
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> ResultsMapping:
        results: ResultsMapping = {}
        total_segments = len(segments)
        try:
            with segment_cache_scope():
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

    def _process_segment(
        self,
        segment: Segment,
        runners: Sequence[Runner],
        seg_index: int,
        total_segments: int,
        cancel_event: threading.Event | None,
        progress: Callable[[str, int, int], None] | None,
    ) -> Dict[str, List[SegmentResult]]:
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
        return segment_results

    def _submit_effort_futures(
        self,
        executor: ThreadPoolExecutor,
        runners: Sequence[Runner],
        segment: Segment,
        cancel_event: threading.Event | None,
    ) -> Tuple[Dict[Future, Runner], List[Runner]]:
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
        for fut in as_completed(future_to_runner):
            if cancel_event and cancel_event.is_set():
                self._log.info(
                    "Cancellation requested during segment %s; stopping remaining futures.",
                    segment.name,
                )
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
                bucket = segment_results.setdefault(runner.segment_team, [])
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
        unique: List[Runner] = []
        seen: Set[int] = set()
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
                bucket = segment_results.setdefault(runner.segment_team, [])
                bucket.append(result)
            completed += 1
            notify_progress(completed)
            if cancel_event and cancel_event.is_set():
                self._log.info(
                    "Cancellation requested while processing segment %s fallback results; stopping early.",
                    segment.name,
                )
                break
        return completed

    def _process_runner_results(
        self,
        runner: Runner,
        segment: Segment,
        efforts: List[dict] | None,
        cancel_event: threading.Event | None = None,
    ) -> SegmentResult | None:
        seg_result = self._result_from_efforts(runner, segment, efforts)
        if seg_result is not None:
            return seg_result
        if not MATCHING_FALLBACK_ENABLED:
            self._log.debug(
                "Matcher fallback disabled; skipping runner=%s segment=%s",
                runner.name,
                segment.id,
            )
            return None
        if not getattr(runner, "payment_required", False):
            self._log.debug(
                "No segment efforts returned; invoking matcher fallback runner=%s segment=%s",
                runner.name,
                segment.id,
            )
        return self._match_runner_segment(runner, segment, cancel_event)

    def _result_from_efforts(
        self,
        runner: Runner,
        segment: Segment,
        efforts: List[dict] | None,
    ) -> SegmentResult | None:
        """Convert Strava segment efforts into a SegmentResult if available."""

        if not efforts:
            return None

        valid: List[tuple[float, dict]] = []
        for effort in efforts:
            if not isinstance(effort, dict):
                continue
            elapsed = effort.get("elapsed_time")
            try:
                elapsed_f = float(elapsed)
            except (TypeError, ValueError):
                continue
            if elapsed_f <= 0:
                continue
            valid.append((elapsed_f, effort))

        if not valid:
            return None

        valid.sort(key=lambda item: item[0])
        fastest_elapsed, fastest_effort = valid[0]
        fastest_date = _parse_iso_datetime(
            fastest_effort.get("start_date_local") or fastest_effort.get("start_date")
        )

        diagnostics: Dict[str, object] = {
            "source": "strava",
            "effort_ids": [
                effort.get("id")
                for _, effort in valid
                if isinstance(effort.get("id"), (int, str))
            ],
            "best_effort_id": fastest_effort.get("id"),
            "moving_time": fastest_effort.get("moving_time"),
        }

        attempts = len(valid)
        team = runner.segment_team or ""
        return SegmentResult(
            runner=runner.name,
            team=team,
            segment=segment.name,
            attempts=attempts,
            fastest_time=fastest_elapsed,
            fastest_date=fastest_date,
            source="strava",
            diagnostics=diagnostics,
        )

    def _match_runner_segment(
        self,
        runner: Runner,
        segment: Segment,
        cancel_event: threading.Event | None = None,
    ) -> SegmentResult | None:
        if cancel_event and cancel_event.is_set():
            return None

        activities = self._get_runner_activities(
            runner,
            segment.start_date,
            segment.end_date,
            cancel_event,
        )
        if not activities:
            return None
        segment_distance_m = self._get_segment_distance(runner, segment.id)
        best_match, best_activity, attempts = self._find_best_match(
            runner,
            segment,
            activities,
            segment_distance_m,
            cancel_event,
        )
        if best_match is None or best_activity is None:
            return None
        fastest_date = _parse_iso_datetime(best_activity.get("start_date_local"))
        diagnostics = {
            "activity_id": best_activity.get("id"),
            "activity_start": best_activity.get("start_date_local"),
            "matcher_diagnostics": best_match.diagnostics,
            "source": "matcher",
        }
        return SegmentResult(
            runner=runner.name,
            team=runner.segment_team,
            segment=segment.name,
            attempts=attempts if attempts > 0 else 1,
            fastest_time=best_match.elapsed_time_s,
            fastest_date=fastest_date,
            source="matcher",
            diagnostics=diagnostics,
        )

    def _find_best_match(
        self,
        runner: Runner,
        segment: Segment,
        activities: Sequence[dict],
        segment_distance_m: Optional[float],
        cancel_event: threading.Event | None,
    ) -> Tuple[Optional[MatchResult], Optional[dict], int]:
        best_match: Optional[MatchResult] = None
        best_activity: Optional[dict] = None
        attempts = 0
        for activity_id, activity in self._iter_candidate_activities(
            activities,
            segment_distance_m,
            cancel_event,
        ):
            if cancel_event and cancel_event.is_set():
                break
            result = self._run_matcher(runner, segment, activity_id)
            if not self._is_successful_match(result):
                continue
            attempts += 1
            if best_match is None or result.elapsed_time_s < best_match.elapsed_time_s:  # type: ignore[union-attr]
                best_match = result
                best_activity = activity
        return best_match, best_activity, attempts

    @staticmethod
    def _is_successful_match(result: Optional[MatchResult]) -> bool:
        return bool(result and result.matched and result.elapsed_time_s is not None)

    def _iter_candidate_activities(
        self,
        activities: Sequence[dict],
        segment_distance_m: Optional[float],
        cancel_event: threading.Event | None = None,
    ) -> Iterator[Tuple[int, dict]]:
        for activity in activities:
            if cancel_event and cancel_event.is_set():
                break
            activity_id = activity.get("id")
            if activity_id is None:
                continue
            if not self._activity_is_candidate(activity):
                continue
            if not self._activity_distance_is_sufficient(activity, segment_distance_m):
                continue
            try:
                yield int(activity_id), activity
            except (TypeError, ValueError):
                continue

    def _run_matcher(
        self,
        runner: Runner,
        segment: Segment,
        activity_id: int,
    ) -> Optional[MatchResult]:
        try:
            return match_activity_to_segment(runner, activity_id, segment.id)
        except Exception:  # noqa: BLE001
            self._log.debug(
                "Matcher failure runner=%s segment=%s activity=%s",
                runner.name,
                segment.id,
                activity_id,
                exc_info=True,
            )
            return None

    def _get_segment_distance(self, runner: Runner, segment_id: int) -> Optional[float]:
        with self._segment_distance_lock:
            cached = self._segment_distance_cache.get(segment_id)
        if cached is not None:
            return cached
        try:
            geometry = fetch_segment_geometry(runner, segment_id)
        except Exception:  # noqa: BLE001
            self._log.debug(
                "Failed to fetch segment geometry for distance check runner=%s segment=%s",
                runner.name,
                segment_id,
                exc_info=True,
            )
            return None
        distance = float(getattr(geometry, "distance_m", 0.0) or 0.0)
        if distance > 0.0:
            with self._segment_distance_lock:
                self._segment_distance_cache[segment_id] = distance
            return distance
        return None

    def _activity_is_candidate(self, activity: dict) -> bool:
        required_types = [value.lower() for value in MATCHING_ACTIVITY_REQUIRED_TYPES]
        if not required_types:
            return True
        activity_type = str(activity.get("type", "")).strip()
        if not activity_type:
            return False
        return activity_type.lower() in required_types

    def _activity_distance_is_sufficient(
        self,
        activity: dict,
        segment_distance_m: Optional[float],
    ) -> bool:
        ratio = MATCHING_ACTIVITY_MIN_DISTANCE_RATIO
        if ratio <= 0.0 or segment_distance_m is None or segment_distance_m <= 0.0:
            return True
        raw_distance = activity.get("distance")
        try:
            activity_distance = float(raw_distance)
        except (TypeError, ValueError):
            return False
        if activity_distance <= 0.0:
            return False
        return activity_distance >= segment_distance_m * ratio

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
            cached = self._activity_cache.get(cache_key)
            if cached is not None:
                self._activity_cache.move_to_end(cache_key)
        if cached is not None:
            self._log.debug(
                "Using cached activities runner=%s window=%s->%s",
                runner.name,
                start_date,
                end_date,
            )
            return cached

        activities = get_activities(runner, start_date, end_date) or []
        if self._activity_cache_max_entries > 0:
            with self._activity_cache_lock:
                if cache_key in self._activity_cache:
                    self._activity_cache.move_to_end(cache_key)
                else:
                    while len(self._activity_cache) >= self._activity_cache_max_entries:
                        self._activity_cache.popitem(last=False)
                self._activity_cache[cache_key] = activities
        return activities

    def _clear_runner_activity_cache(self) -> None:
        """Reset per-runner activity cache before processing a new batch."""

        with self._activity_cache_lock:
            self._activity_cache.clear()


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except Exception:
        return None


__all__ = ["SegmentService"]
