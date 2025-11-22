"""Generate interactive deviation overlays for specific Strava efforts."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import folium

from ..config import (
    INPUT_FILE,
    MATCHING_MAX_OFFSET_M,
    MATCHING_RESAMPLE_INTERVAL_M,
    MATCHING_SIMPLIFICATION_TOLERANCE_M,
    MATCHING_START_TOLERANCE_M,
)
from ..excel_reader import ExcelFormatError, read_runners, read_segments
from ..models import Runner, Segment
from ..matching import _compute_coverage_diagnostics
from ..matching.visualization import create_deviation_map
from ..matching.fetchers import fetch_activity_stream, fetch_segment_geometry
from ..matching.models import ActivityTrack, SegmentGeometry
from ..matching.preprocessing import prepare_activity, prepare_geometry
from ..matching.validation import CoverageResult

PathLike = Union[str, Path]
SegmentFetcher = Callable[[Runner, int], SegmentGeometry]
ActivityFetcher = Callable[[Runner, int], ActivityTrack]


def build_deviation_map_for_effort(
    runner: Runner,
    activity_id: int,
    segment_id: int,
    *,
    threshold_m: float = 50.0,
    output_html: Optional[PathLike] = None,
    segment_fetcher: SegmentFetcher = fetch_segment_geometry,
    activity_fetcher: ActivityFetcher = fetch_activity_stream,
) -> Tuple[folium.Map, CoverageResult, dict[str, object]]:
    """Create a deviation map for an activity/segment pairing.

    Args:
        runner: Runner whose Strava credentials are used for API calls.
        activity_id: Identifier of the Strava activity to inspect.
        segment_id: Identifier of the Strava segment defining the course.
        threshold_m: Minimum perpendicular offset (metres) flagged as divergence.
        output_html: Optional path where the rendered HTML map should be saved.
        segment_fetcher: Callable returning :class:`SegmentGeometry` metadata.
        activity_fetcher: Callable returning :class:`ActivityTrack` data.

    Returns:
        Tuple containing the :class:`folium.Map`, the corresponding
        :class:`CoverageResult`, and a diagnostics mapping summarising coverage
        evaluation.

    Raises:
        ValueError: If coverage offsets cannot be aligned with the activity
            samples.
        Exception: Propagates API or preprocessing errors from downstream calls.
    """

    segment = segment_fetcher(runner, segment_id)
    prepared_segment = prepare_geometry(
        segment,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )

    activity_track = activity_fetcher(runner, activity_id)
    prepared_activity = prepare_activity(
        activity_track,
        prepared_segment.transformer,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )

    coverage, diagnostics, _bounds, _indices, _hints = _compute_coverage_diagnostics(
        prepared_activity,
        prepared_segment,
        max_offset_threshold=MATCHING_MAX_OFFSET_M,
        start_tolerance_m=MATCHING_START_TOLERANCE_M,
    )
    gate_slice = diagnostics.get("gate_slice_indices")

    map_object = create_deviation_map(
        prepared_activity,
        prepared_segment,
        coverage,
        threshold_m=threshold_m,
        gate_slice=gate_slice,
        output_html_path=output_html,
    )
    return map_object, coverage, diagnostics


def _select_runner(
    runners: Sequence[Runner],
    *,
    name: Optional[str],
    strava_id: Optional[str],
) -> Runner:
    """Return the workbook runner matching the provided identifier."""

    if strava_id:
        target = str(strava_id).strip()
        for runner in runners:
            if runner.strava_id == target:
                return runner
        raise ValueError(f"No runner found with Strava ID {target}")

    if not name:
        raise ValueError("Runner name or Strava ID must be provided")
    target = name.strip().lower()
    matches = [r for r in runners if r.name.lower() == target]
    if not matches:
        raise ValueError(f"No runner found with name '{name}'")
    if len(matches) > 1:
        raise ValueError(f"Runner name '{name}' is ambiguous; specify ID")
    return matches[0]


def _select_segment(
    segments: Sequence[Segment],
    *,
    segment_id: Optional[int],
    name: Optional[str],
) -> Segment:
    """Return the workbook segment matching the provided identifier."""

    if segment_id is not None:
        for segment in segments:
            if segment.id == segment_id:
                return segment
        raise ValueError(f"No segment found with ID {segment_id}")

    if not name:
        raise ValueError("Segment name or ID must be provided")
    target = name.strip().lower()
    matches = [seg for seg in segments if seg.name.lower() == target]
    if not matches:
        raise ValueError(f"No segment found with name '{name}'")
    if len(matches) > 1:
        raise ValueError(f"Segment name '{name}' is ambiguous; specify ID")
    return matches[0]


def _slugify(value: str) -> str:
    """Return a filesystem-friendly slug."""

    normalized = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized)
    slug = slug.strip("-")
    return slug or "map"


def _default_output_path(runner: Runner, segment: Segment, activity_id: int) -> Path:
    """Return a default HTML output path for the generated map."""

    runner_slug = _slugify(runner.name)
    segment_slug = _slugify(segment.name)
    filename = f"{runner_slug}-{segment_slug}-seg{segment.id}-act{activity_id}.html"
    return Path("maps") / filename


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the deviation map tool."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate an interactive HTML map highlighting large activity"
            " deviations within a segment gate window."
        )
    )
    parser.add_argument("--activity-id", type=int, required=True)
    parser.add_argument("--runner-name")
    parser.add_argument("--runner-id")
    parser.add_argument("--segment-id", type=int)
    parser.add_argument("--segment-name")
    parser.add_argument(
        "--threshold-m",
        type=float,
        default=50.0,
        help="Minimum offset (metres) flagged as divergence (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output HTML path; defaults to maps/<slug>.html",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point used via ``python -m strava_competition.tools.deviation_map``."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    try:
        runners = read_runners(INPUT_FILE)
        segments = read_segments(INPUT_FILE)
    except (ExcelFormatError, FileNotFoundError) as exc:
        logging.error("Failed to load workbook '%s': %s", INPUT_FILE, exc)
        return 1

    try:
        runner = _select_runner(
            runners,
            name=args.runner_name,
            strava_id=args.runner_id,
        )
        segment = _select_segment(
            segments,
            segment_id=args.segment_id,
            name=args.segment_name,
        )
    except ValueError as exc:
        logging.error("%s", exc)
        return 1

    output_path = args.output or _default_output_path(runner, segment, args.activity_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Fetching data for runner='%s' segment='%s' activity=%s",
        runner.name,
        segment.name,
        args.activity_id,
    )
    try:
        _map, coverage, diagnostics = build_deviation_map_for_effort(
            runner,
            args.activity_id,
            segment.id,
            threshold_m=args.threshold_m,
            output_html=output_path,
        )
    except Exception as exc:  # noqa: BLE001
        logging.error("Failed to build deviation map: %s", exc)
        return 1

    coverage_ratio = coverage.coverage_ratio
    trimmed_max = diagnostics.get("gate_trimmed_max_offset_m")
    logging.info("Coverage ratio %.3f", coverage_ratio)
    if trimmed_max is not None:
        logging.info("Gate-trimmed max offset %.1f m", trimmed_max)
    logging.info("Deviation map written to %s", output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
