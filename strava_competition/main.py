from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from typing import Any, Mapping, Sequence

from .auth import TokenError
from .config import (
    INPUT_FILE,
    OUTPUT_FILE,
    OUTPUT_FILE_TIMESTAMP_ENABLED,
    MAX_WORKERS,
)
from .excel_reader import (
    read_runners,
    read_segment_groups,
    read_distance_windows,
    ExcelFormatError,
    workbook_context,
)
from .excel_writer import (
    update_runner_refresh_tokens,
    write_results,
)
from .errors import StravaAPIError
from .models import Runner, SegmentGroup
from .services import SegmentService, DistanceService
from .services.segment_service import ResultsMapping
from .strava_api import get_default_client
from .strava_client import telemetry

DistanceWindow = tuple[datetime, datetime, float | None]
DistanceWindowsResult = list[tuple[str, list[dict[str, Any]]]]


def _setup_logging() -> None:
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        )


def _load_inputs(
    input_file: str,
) -> tuple[
    list[SegmentGroup],
    list[Runner],
    list[DistanceWindow],
    list[Runner],
    list[Runner],
]:
    logging.info("Loading segments, runners and distance windows ...")
    with workbook_context(input_file) as workbook:
        segment_groups = read_segment_groups(input_file, workbook=workbook)
        runners = read_runners(input_file, workbook=workbook)
        distance_windows = read_distance_windows(input_file, workbook=workbook)
    segment_runners = [r for r in runners if r.segment_team]
    distance_runners = [r for r in runners if r.distance_team]
    if distance_windows:
        logging.info("Loaded %s distance windows", len(distance_windows))
    else:
        logging.info("No distance windows defined (sheet optional)")
    return segment_groups, runners, distance_windows, segment_runners, distance_runners


def _ensure_tokens_early(runners: Sequence[Runner], input_file: str) -> None:
    """Refresh all runner tokens up front and best-effort persist rotations.

    Persistence failures are logged loudly but never raised: rotated tokens
    stay on the in-memory ``Runner`` objects and are re-persisted by
    ``_persist_tokens_final`` at shutdown.
    """
    any_token_rotated = False
    for r in runners:
        before = getattr(r, "refresh_token", None)
        try:
            # persist=False: we batch-write all rotated tokens once below,
            # avoiding a full Runners-sheet rewrite per rotated runner.
            get_default_client().ensure_runner_token(r, persist=False)
        except (TokenError, StravaAPIError) as e:
            logging.warning(
                "Initial token ensure failed for runner=%s: %s",
                getattr(r, "name", "?"),
                e,
            )
            continue
        except Exception:
            logging.exception(
                "Unexpected error ensuring token for runner=%s",
                getattr(r, "name", "?"),
            )
            continue
        after = getattr(r, "refresh_token", None)
        if before and after and before != after:
            any_token_rotated = True
    if any_token_rotated:
        try:
            update_runner_refresh_tokens(input_file, runners)
        except (OSError, ExcelFormatError):
            logging.exception(
                "Failed to persist rotated refresh tokens early to '%s'; "
                "tokens remain in memory and will be re-persisted at shutdown",
                input_file,
            )
        else:
            logging.info("Persisted rotated refresh tokens early (pre-processing)")


def _process_segments(
    segment_groups: Sequence[SegmentGroup], segment_runners: Sequence[Runner]
) -> ResultsMapping:
    logging.info(
        "Processing %s segment groups for %s segment runners ...",
        len(segment_groups),
        len(segment_runners),
    )
    segment_service = SegmentService(max_workers=MAX_WORKERS)

    def _progress(seg_name: str, done: int, total: int) -> None:
        if done == 1 or done == total or done % 5 == 0:
            logging.info(
                "Segment %s progress: %d/%d runners fetched", seg_name, done, total
            )

    results = segment_service.process_groups(
        segment_groups, segment_runners, progress=_progress
    )
    logging.info("Finished segment aggregation for %d segments", len(results))
    return results


def _process_distance(
    distance_runners: Sequence[Runner],
    distance_windows: Sequence[DistanceWindow],
) -> DistanceWindowsResult:
    distance_windows_results: DistanceWindowsResult = []
    if distance_windows and distance_runners:
        distance_windows_results = DistanceService().process(
            distance_runners, distance_windows
        )
    return distance_windows_results


def _persist_tokens_final(runners: Sequence[Runner], input_file: str) -> None:
    try:
        # Always write once more defensively (lightweight operation).
        update_runner_refresh_tokens(input_file, runners)
    except OSError as e:
        logging.warning("Failed to persist refresh tokens at shutdown: %s", e)
    except Exception:
        logging.exception("Unexpected error persisting refresh tokens at shutdown")
    else:
        logging.info("Refresh tokens persisted at shutdown.")


def _format_api_usage_summary(
    counters: Mapping[str, int],
    limiter_snapshot: Mapping[str, float | int | None],
) -> str:
    """Format the end-of-run API usage line.

    Args:
        counters: Telemetry counter snapshot (live/cached/refetch counts).
        limiter_snapshot: Rate limiter snapshot; ``short_used``/``short_limit``
            of ``None`` means no rate-limit headers were seen (fully cached
            run) and renders as ``rate limit: n/a``.

    Returns:
        A single log-ready summary line.
    """
    usage = (
        f"API usage: live={counters.get(telemetry.LIVE_CALLS, 0)} "
        f"cached={counters.get(telemetry.CACHE_HITS, 0)} "
        f"validation_refetches={counters.get(telemetry.VALIDATION_REFETCHES, 0)}"
    )
    reset_waits = counters.get(telemetry.RESET_WAITS, 0)
    if reset_waits > 0:
        usage += f" reset_waits={reset_waits}"
    short_used = limiter_snapshot.get("short_used")
    short_limit = limiter_snapshot.get("short_limit")
    if short_used is None or short_limit is None:
        return f"{usage} | rate limit: n/a"
    rate = f"rate limit: {short_used}/{short_limit} (15min)"
    daily_used = limiter_snapshot.get("daily_used")
    daily_limit = limiter_snapshot.get("daily_limit")
    if daily_used is not None and daily_limit is not None:
        rate += f", {daily_used}/{daily_limit} (daily)"
    return f"{usage} | {rate}"


def _log_api_usage_summary() -> None:
    """Log the end-of-run API usage summary (live vs cached vs refetches)."""
    logging.info(
        _format_api_usage_summary(
            telemetry.snapshot(),
            get_default_client().rate_limiter_snapshot(),
        )
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="strava_competition",
        description="Run the Strava competition aggregation.",
    )
    parser.add_argument(
        "--input",
        "-i",
        default=INPUT_FILE,
        help=f"Path to the input Excel workbook (default: {INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=OUTPUT_FILE,
        help=f"Output file base name without extension (default: {OUTPUT_FILE})",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    telemetry.reset()
    args = _parse_args()
    input_file = args.input
    output_base = args.output

    if OUTPUT_FILE_TIMESTAMP_ENABLED:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_base}_{timestamp}.xlsx"
    else:
        output_file = f"{output_base}.xlsx"

    try:
        (
            segment_groups,
            runners,
            distance_windows,
            segment_runners,
            distance_runners,
        ) = _load_inputs(input_file)
    except (ExcelFormatError, FileNotFoundError) as exc:
        logging.exception("Failed to load input workbook '%s'", input_file)
        raise SystemExit(1) from exc

    try:
        # Early token refresh & persistence to avoid losing rotated refresh tokens
        _ensure_tokens_early(runners, input_file)

        results = _process_segments(segment_groups, segment_runners)
        distance_windows_results = _process_distance(distance_runners, distance_windows)

        write_results(
            output_file, results, distance_windows_results=distance_windows_results
        )
        logging.info(
            "Results saved to %s (segment sheets=%s, distance sheets=%s)",
            output_file,
            len(results),
            len(distance_windows_results),
        )
    finally:
        _persist_tokens_final(runners, input_file)
        _log_api_usage_summary()
