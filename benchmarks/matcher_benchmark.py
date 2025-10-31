"""Benchmark the GPS segment matcher pipeline with large point counts."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Late imports rely on the adjusted sys.path above.
from strava_competition.config import (  # noqa: E402
    MATCHING_RESAMPLE_INTERVAL_M,
    MATCHING_SIMPLIFICATION_TOLERANCE_M,
)
from strava_competition.matching.models import (  # noqa: E402
    ActivityTrack,
    SegmentGeometry,
)
from strava_competition.matching.preprocessing import (  # noqa: E402
    prepare_activity,
    prepare_geometry,
)
from strava_competition.matching.similarity import (  # noqa: E402
    discrete_frechet_distance,
)
from strava_competition.matching.timing import (  # noqa: E402
    estimate_segment_time,
)
from strava_competition.matching.validation import compute_coverage  # noqa: E402


@dataclass(slots=True)
class StageDurations:
    """Timing measurements (in seconds) for the matcher pipeline."""

    prepare_segment: float
    prepare_activity: float
    frechet: float
    timing: float

    @property
    def total(self) -> float:
        """Return the aggregate duration for this benchmark iteration."""

        return self.prepare_segment + self.prepare_activity + self.frechet + self.timing


@dataclass(slots=True)
class BenchmarkSummary:
    """Aggregated statistics for multiple benchmark iterations."""

    point_count: int
    iterations: int
    mean_prepare_segment_ms: float
    mean_prepare_activity_ms: float
    mean_frechet_ms: float
    mean_timing_ms: float
    mean_total_ms: float
    worst_total_ms: float


def _build_polyline(point_count: int) -> List[tuple[float, float]]:
    """Generate a straight lat/lon polyline with evenly spaced points."""

    base_lat = 37.0
    base_lon = -122.0
    step_deg = 1.2e-5
    return [(base_lat + idx * step_deg, base_lon) for idx in range(point_count)]


def _build_activity(
    points: Iterable[tuple[float, float]],
) -> ActivityTrack:
    """Construct an ActivityTrack using the provided coordinate sequence."""

    latlon = list(points)
    timestamps = [float(idx * 2.0) for idx in range(len(latlon))]
    return ActivityTrack(
        activity_id=999,
        points=latlon,
        timestamps_s=timestamps,
    )


def _run_iteration(
    segment: SegmentGeometry,
) -> StageDurations:
    """Execute one benchmark iteration and capture per-stage timings."""

    start = time.perf_counter()
    prepared_segment = prepare_geometry(
        segment,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )
    prepare_segment = time.perf_counter() - start

    start = time.perf_counter()
    activity = _build_activity(segment.points)
    prepared_activity = prepare_activity(
        activity,
        transformer=prepared_segment.transformer,
        simplification_tolerance_m=MATCHING_SIMPLIFICATION_TOLERANCE_M,
        resample_interval_m=MATCHING_RESAMPLE_INTERVAL_M,
    )
    prepare_activity_dur = time.perf_counter() - start

    start = time.perf_counter()
    score = discrete_frechet_distance(
        prepared_activity.resampled_points,
        prepared_segment.resampled_points,
    )
    _ = score  # guard against optimisation stripping the call
    frechet = time.perf_counter() - start

    coverage = compute_coverage(
        prepared_activity.metric_points,
        prepared_segment.metric_points,
    )
    if coverage.coverage_bounds is None:
        raise RuntimeError("Synthetic activity failed to cover the segment")

    start = time.perf_counter()
    estimate = estimate_segment_time(
        prepared_activity,
        prepared_segment,
        coverage.coverage_bounds,
    )
    _ = estimate
    timing = time.perf_counter() - start

    return StageDurations(
        prepare_segment=prepare_segment,
        prepare_activity=prepare_activity_dur,
        frechet=frechet,
        timing=timing,
    )


def run_benchmark(
    point_count: int,
    iterations: int,
) -> BenchmarkSummary:
    """Benchmark the matcher pipeline and return aggregated timings."""

    if point_count < 10000:
        raise ValueError("point_count must be at least 10,000")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    segment = SegmentGeometry(
        segment_id=1010,
        points=_build_polyline(point_count),
        distance_m=float(point_count * 3.5),
    )

    durations: List[StageDurations] = []
    for _ in range(iterations):
        durations.append(_run_iteration(segment))

    mean_prepare_segment = statistics.fmean(item.prepare_segment for item in durations)
    mean_prepare_activity = statistics.fmean(
        item.prepare_activity for item in durations
    )
    mean_frechet = statistics.fmean(item.frechet for item in durations)
    mean_timing = statistics.fmean(item.timing for item in durations)
    mean_total = statistics.fmean(item.total for item in durations)
    worst_total = max(item.total for item in durations)

    return BenchmarkSummary(
        point_count=point_count,
        iterations=iterations,
        mean_prepare_segment_ms=mean_prepare_segment * 1000.0,
        mean_prepare_activity_ms=mean_prepare_activity * 1000.0,
        mean_frechet_ms=mean_frechet * 1000.0,
        mean_timing_ms=mean_timing * 1000.0,
        mean_total_ms=mean_total * 1000.0,
        worst_total_ms=worst_total * 1000.0,
    )


def _format_summary(summary: BenchmarkSummary) -> Dict[str, float]:
    """Return a JSON-friendly representation of the benchmark summary."""

    return {
        "point_count": summary.point_count,
        "iterations": summary.iterations,
        "mean_prepare_segment_ms": summary.mean_prepare_segment_ms,
        "mean_prepare_activity_ms": summary.mean_prepare_activity_ms,
        "mean_frechet_ms": summary.mean_frechet_ms,
        "mean_timing_ms": summary.mean_timing_ms,
        "mean_total_ms": summary.mean_total_ms,
        "worst_total_ms": summary.worst_total_ms,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark utility."""

    parser = argparse.ArgumentParser(
        description="Benchmark the matcher pipeline with large activities",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=12000,
        help="Number of points in the synthetic segment/activity",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of repetitions for averaging",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point when running the module as a script."""

    args = _parse_args()
    summary = run_benchmark(args.points, args.iterations)
    formatted = _format_summary(summary)
    for key, value in formatted.items():
        if key in {"point_count", "iterations"}:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
