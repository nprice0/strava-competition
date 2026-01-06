import argparse
from datetime import datetime, timezone
import logging
from typing import Any, List, Sequence, Tuple

from .auth import TokenError
from .config import (
    INPUT_FILE,
    OUTPUT_FILE,
    OUTPUT_FILE_TIMESTAMP_ENABLED,
    MAX_WORKERS,
)
from .excel_reader import (
    read_runners,
    read_segments,
    read_distance_windows,
    ExcelFormatError,
    workbook_context,
)
from .excel_writer import (
    update_runner_refresh_tokens,
    write_results,
)
from .errors import StravaAPIError
from .models import Runner, Segment
from .services import SegmentService, DistanceService
from .services.segment_service import ResultsMapping
from .strava_api import get_default_client

DistanceWindow = Tuple[datetime, datetime, float | None]
DistanceWindowsResult = List[Tuple[str, List[dict[str, Any]]]]


def _setup_logging() -> None:
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        )


def _load_inputs(
    input_file: str,
) -> Tuple[
    List[Segment],
    List[Runner],
    List[DistanceWindow],
    List[Runner],
    List[Runner],
]:
    logging.info("Loading segments, runners and distance windows ...")
    with workbook_context(input_file) as workbook:
        segments = read_segments(input_file, workbook=workbook)
        runners = read_runners(input_file, workbook=workbook)
        distance_windows = read_distance_windows(input_file, workbook=workbook)
    segment_runners = [r for r in runners if r.segment_team]
    distance_runners = [r for r in runners if r.distance_team]
    if distance_windows:
        logging.info("Loaded %s distance windows", len(distance_windows))
    else:
        logging.info("No distance windows defined (sheet optional)")
    return segments, runners, distance_windows, segment_runners, distance_runners


def _ensure_tokens_early(runners: Sequence[Runner], input_file: str) -> None:
    any_token_rotated = False
    for r in runners:
        before = getattr(r, "refresh_token", None)
        try:
            get_default_client().ensure_runner_token(r)
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
        update_runner_refresh_tokens(input_file, runners)
        logging.info("Persisted rotated refresh tokens early (pre-processing)")


def _process_segments(
    segments: Sequence[Segment], segment_runners: Sequence[Runner]
) -> ResultsMapping:
    logging.info(
        "Processing %s segments for %s segment runners ...",
        len(segments),
        len(segment_runners),
    )
    segment_service = SegmentService(max_workers=MAX_WORKERS)

    def _progress(seg_name: str, done: int, total: int) -> None:
        if done == 1 or done == total or done % 5 == 0:
            logging.info(
                "Segment %s progress: %d/%d runners fetched", seg_name, done, total
            )

    results = segment_service.process(segments, segment_runners, progress=_progress)
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
    rotated = False
    try:
        # Always write once more defensively (lightweight operation).
        update_runner_refresh_tokens(input_file, runners)
        rotated = True
    except (OSError, PermissionError) as e:
        logging.warning("Failed to persist refresh tokens at shutdown: %s", e)
    except Exception:
        logging.exception("Unexpected error persisting refresh tokens at shutdown")
    else:
        if rotated:
            logging.info("Refresh tokens persisted at shutdown.")


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
            segments,
            runners,
            distance_windows,
            segment_runners,
            distance_runners,
        ) = _load_inputs(input_file)
    except (ExcelFormatError, FileNotFoundError) as exc:
        logging.error("Failed to load input workbook '%s': %s", input_file, exc)
        return

    # Early token refresh & persistence to avoid losing rotated refresh tokens
    _ensure_tokens_early(runners, input_file)

    try:
        results = _process_segments(segments, segment_runners)
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
