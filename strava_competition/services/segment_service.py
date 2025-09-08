"""Segment competition service (application layer).

Encapsulates orchestration of fetching efforts and aggregating results so
higher-level code (main, CLI, etc.) depends on a stable service API rather
than a free function implementation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import logging
import threading
from typing import Callable, Dict, List, Sequence

from ..models import Segment, Runner, SegmentResult
from ..config import MAX_WORKERS
from ..strava_api import get_segment_efforts

ResultsMapping = Dict[str, Dict[str, List[SegmentResult]]]

class SegmentService:
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or MAX_WORKERS
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self._log = logging.getLogger(self.__class__.__name__)

    def process(
        self,
        segments: Sequence[Segment],
        runners: Sequence[Runner],
        cancel_event: threading.Event | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> ResultsMapping:
        results: ResultsMapping = {}
        total_segments = len(segments)
        for seg_index, segment in enumerate(segments, start=1):
            if segment.start_date > segment.end_date:
                self._log.warning(
                    "Skipping segment with inverted date range: %s (start=%s end=%s)",
                    segment.name,
                    segment.start_date,
                    segment.end_date,
                )
                continue
            results[segment.name] = {}
            eligible_runners = [
                r for r in runners if not getattr(r, "payment_required", False) and r.segment_team
            ]
            total_runners = len(eligible_runners)
            if total_runners == 0:
                self._log.info("No eligible runners for segment %s", segment.name)
                continue
            self._log.debug(
                "Processing segment %s (%d/%d) with %d runners (max_workers=%d)",
                segment.name,
                seg_index,
                total_segments,
                total_runners,
                self.max_workers,
            )
            if cancel_event and cancel_event.is_set():
                self._log.info("Cancellation requested before segment %s; aborting.", segment.name)
                break
            completed = 0
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_runner: Dict[Future, Runner] = {}
                for runner in eligible_runners:
                    if cancel_event and cancel_event.is_set():
                        break
                    fut = executor.submit(
                        get_segment_efforts,
                        runner,
                        segment.id,
                        segment.start_date,
                        segment.end_date,
                    )
                    future_to_runner[fut] = runner
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
                        efforts = None  # errors already logged by API layer
                    if efforts:
                        seg_result = self._result_from_efforts(runner, segment, efforts)
                        if seg_result:
                            team_bucket = results[segment.name].setdefault(runner.segment_team, [])
                            team_bucket.append(seg_result)
                    completed += 1
                    if progress:
                        try:
                            progress(segment.name, completed, total_runners)
                        except Exception:
                            self._log.debug(
                                "Progress callback failed for segment %s", segment.name, exc_info=True
                            )
            for team_results in results[segment.name].values():
                team_results.sort(key=lambda r: r.fastest_time)
        return results

    @staticmethod
    def _result_from_efforts(runner: Runner, segment: Segment, efforts: List[dict] | None) -> SegmentResult | None:
        if not efforts:
            return None
        try:
            fastest = min(efforts, key=lambda e: e["elapsed_time"])
        except Exception:
            return None
        return SegmentResult(
            runner=runner.name,
            team=runner.segment_team,
            segment=segment.name,
            attempts=len(efforts),
            fastest_time=fastest.get("elapsed_time"),
            fastest_date=fastest.get("start_date_local"),
        )

__all__ = ["SegmentService"]