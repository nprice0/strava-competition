from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import SegmentResult
from .config import MAX_WORKERS
from .strava_api import get_segment_efforts


def _result_from_efforts(runner, segment, efforts):
    attempts = len(efforts)
    fastest = min(efforts, key=lambda e: e["elapsed_time"]) if efforts else None
    if not fastest:
        return None
    return SegmentResult(
        runner=runner.name,
        team=runner.team,
        segment=segment.name,
        attempts=attempts,
        fastest_time=fastest["elapsed_time"],
        fastest_date=fastest["start_date_local"],
    )


def process_segments(segments, runners, max_workers: int | None = None):
    if max_workers is None:
        max_workers = MAX_WORKERS
    results: dict[str, dict[str, list[SegmentResult]]] = {}
    for segment in segments:
        results[segment.name] = {}

        # Fetch efforts for all runners in parallel for this segment
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            future_to_runner = {
                exe.submit(
                    get_segment_efforts, runner, segment.id, segment.start_date, segment.end_date
                ): runner
                for runner in runners
                if not getattr(runner, "payment_required", False)
            }
            for fut in as_completed(future_to_runner):
                runner = future_to_runner[fut]
                try:
                    efforts = fut.result()
                except Exception:
                    # Already logged inside API client; skip this runner
                    efforts = None
                if not efforts:
                    continue
                seg_result = _result_from_efforts(runner, segment, efforts)
                if not seg_result:
                    continue
                if runner.team not in results[segment.name]:
                    results[segment.name][runner.team] = []
                results[segment.name][runner.team].append(seg_result)

        # Sort each team's results
        for team in results[segment.name]:
            results[segment.name][team].sort(key=lambda r: r.fastest_time)
    return results
