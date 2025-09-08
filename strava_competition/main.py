from datetime import datetime
import logging
from typing import List, Tuple

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
)
from .excel_writer import (
    update_runner_refresh_tokens,
    write_results,
)
from .services import SegmentService, DistanceService
from .strava_api import get_activities
from .excel_writer import update_runner_refresh_tokens
from .strava_api import _ensure_runner_token as _internal_ensure_token  # type: ignore


def main():
    # Central logging setup (once)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        )
    if OUTPUT_FILE_TIMESTAMP_ENABLED:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{OUTPUT_FILE}_{timestamp}.xlsx"
    else:
        output_file = f"{OUTPUT_FILE}.xlsx"

    logging.info("Loading segments, runners and distance windows ...")
    segments = read_segments(INPUT_FILE)
    runners = read_runners(INPUT_FILE)
    distance_windows = read_distance_windows(INPUT_FILE)
    segment_runners = [r for r in runners if r.segment_team]
    distance_runners = [r for r in runners if r.distance_team]
    if distance_windows:
        logging.info("Loaded %s distance windows", len(distance_windows))
    else:
        logging.info("No distance windows defined (sheet optional)")

    # --- Early token refresh & persistence to avoid losing rotated refresh tokens ---
    any_token_rotated = False
    for r in runners:
        before = getattr(r, "refresh_token", None)
        try:
            _internal_ensure_token(r)  # forces access token / potential refresh rotation
        except Exception as e:
            logging.warning("Initial token ensure failed for runner=%s: %s", getattr(r, 'name', '?'), e)
            continue
        after = getattr(r, "refresh_token", None)
        if before and after and before != after:
            any_token_rotated = True
    if any_token_rotated:
        update_runner_refresh_tokens(INPUT_FILE, runners)
        logging.info("Persisted rotated refresh tokens early (pre-processing)")

    try:
        # Segment competition processing
        logging.info(
            "Processing %s segments for %s segment runners ...", len(segments), len(segment_runners)
        )
        segment_service = SegmentService(max_workers=MAX_WORKERS)

        def _progress(seg_name: str, done: int, total: int) -> None:
            # Log first, every 5th, and completion for visibility at INFO level
            if done == 1 or done == total or done % 5 == 0:
                logging.info("Segment %s progress: %d/%d runners fetched", seg_name, done, total)

        results = segment_service.process(segments, segment_runners, progress=_progress)
        logging.info("Finished segment aggregation for %d segments", len(results))

        # Distance/elevation competition processing via service (deduplicated summary)
        distance_windows_results: List[Tuple[str, list[dict]]] = []
        if distance_windows and distance_runners:
            distance_windows_results = DistanceService().process(distance_runners, distance_windows)

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
        # Final persistence if any token changed during processing
        rotated = False
        try:
            df_before = {r.strava_id: r.refresh_token for r in runners}
            # nothing to compare with except earlier snapshot; we already persisted initial rotation.
            # Always write once more defensively (lightweight operation).
            update_runner_refresh_tokens(INPUT_FILE, runners)
            rotated = True
        except Exception as e:
            logging.warning("Failed to persist refresh tokens at shutdown: %s", e)
        else:
            if rotated:
                logging.info("Refresh tokens persisted at shutdown.")
