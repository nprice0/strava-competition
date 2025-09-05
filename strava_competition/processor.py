"""Processing orchestration for fetching segment efforts and aggregating results.

Public function:
    process_segments(segments, runners, max_workers=None, cancel_event=None, progress=None)

Responsibilities:
    * Fan out Strava effort fetches per runner per segment using a thread pool.
    * Aggregate raw efforts into SegmentResult objects (fastest effort + attempts).
    * Group results by segment -> team -> list[SegmentResult].

Enterprise-focused additions:
    * Type hints for clarity.
    * Input validation and defensive programming around max_workers.
    * Optional cancellation via threading.Event (best-effort cooperative stop).
    * Optional progress callback: progress(segment_name, completed_runners, total_runners).
    * Structured debug logging (can be enabled by configuring the root logger).

Notes:
    * Errors from API calls are already logged inside the API layer; here we simply
      skip failed runners.
    * Sorting of each team's results is performed for deterministic downstream output.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import logging
import threading
from typing import Callable, Dict, Iterable, List, Sequence

from .models import SegmentResult, Segment, Runner
from .config import MAX_WORKERS
from .strava_api import get_segment_efforts


def _result_from_efforts(runner: Runner, segment: Segment, efforts: List[dict] | None) -> SegmentResult | None:
    """Convert raw efforts list into a SegmentResult selecting the fastest attempt.

    Returns None if there are no efforts.
    """
    if not efforts:
        return None
    attempts = len(efforts)
    try:
        fastest = min(efforts, key=lambda e: e["elapsed_time"])
    except (KeyError, ValueError, TypeError):  # Defensive: malformed effort objects
        logging.debug("Malformed effort data encountered for runner=%s segment=%s", runner.name, segment.name)
        return None
    return SegmentResult(
        runner=runner.name,
        team=runner.team,
        segment=segment.name,
        attempts=attempts,
        fastest_time=fastest.get("elapsed_time"),
        fastest_date=fastest.get("start_date_local"),
    )

def process_segments(
    segments: Sequence[Segment],
    runners: Sequence[Runner],
    max_workers: int | None = None,
    cancel_event: threading.Event | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> Dict[str, Dict[str, List[SegmentResult]]]:
    """Fetch efforts for each runner across the provided segments and aggregate.

    Args:
        segments: Ordered collection of Segment definitions.
        runners: Collection of Runner objects.
        max_workers: Optional override for per-segment thread pool size.
        cancel_event: If set and becomes true, remaining work for the current segment is skipped.
        progress: Optional callback invoked as progress(segment_name, completed_runners, total_runners).

    Returns:
        Nested dict: {segment_name: {team_name: [SegmentResult, ...], ...}, ...}
    """
    if max_workers is None:
        max_workers = MAX_WORKERS
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    results: Dict[str, Dict[str, List[SegmentResult]]] = {}
    total_segments = len(segments)
    log = logging.getLogger(__name__)

    for seg_index, segment in enumerate(segments, start=1):
        if segment.start_date > segment.end_date:
            log.warning(
                "Skipping segment with inverted date range: %s (start=%s end=%s)",
                segment.name,
                segment.start_date,
                segment.end_date,
            )
            continue
        results[segment.name] = {}

        eligible_runners = [r for r in runners if not getattr(r, "payment_required", False)]
        total_runners = len(eligible_runners)
        if total_runners == 0:
            log.info("No eligible runners for segment %s", segment.name)
            continue

        log.debug(
            "Processing segment %s (%d/%d) with %d runners (max_workers=%d)",
            segment.name,
            seg_index,
            total_segments,
            total_runners,
            max_workers,
        )

        completed = 0
        if cancel_event and cancel_event.is_set():  # Early exit before starting
            log.info("Cancellation requested before segment %s; aborting.", segment.name)
            break

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
                    log.info("Cancellation requested during segment %s; stopping remaining futures.", segment.name)
                    break
                runner = future_to_runner[fut]
                try:
                    efforts = fut.result()
                except Exception:  # noqa: BLE001
                    # Errors already logged by API layer; skip.
                    efforts = None
                if efforts:
                    seg_result = _result_from_efforts(runner, segment, efforts)
                    if seg_result:
                        team_bucket = results[segment.name].setdefault(runner.team, [])
                        team_bucket.append(seg_result)
                completed += 1
                if progress:
                    try:
                        progress(segment.name, completed, total_runners)
                    except Exception:  # Non-fatal progress callback errors
                        log.debug("Progress callback failed for segment %s", segment.name, exc_info=True)

        # Sort team lists for the segment
        for team_results in results[segment.name].values():
            team_results.sort(key=lambda r: r.fastest_time)

    return results
