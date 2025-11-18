"""Tests for coverage diagnostics offset handling in ``matching``."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from strava_competition.matching import _compute_coverage_diagnostics
from strava_competition.matching.models import ActivityTrack, SegmentGeometry
from strava_competition.matching.preprocessing import (
    PreparedActivityTrack,
    PreparedSegmentGeometry,
)


def _build_prepared_segment() -> PreparedSegmentGeometry:
    """Create a simple prepared segment with metric coordinates."""

    metric_points = np.column_stack((np.zeros(11), np.linspace(0.0, 100.0, 11)))
    segment = SegmentGeometry(
        segment_id=101,
        points=[(0.0, float(idx)) for idx in np.linspace(0.0, 1.0, 11)],
        distance_m=100.0,
    )
    return PreparedSegmentGeometry(
        segment=segment,
        latlon_points=segment.points,
        metric_points=metric_points,
        simplified_points=metric_points,
        resampled_points=metric_points,
        transformer=cast(Any, object()),
        simplification_tolerance_m=1.0,
        resample_interval_m=5.0,
        simplification_capped=False,
        resample_capped=False,
    )


def _build_prepared_activity() -> PreparedActivityTrack:
    """Create an activity with one large outlier inside the gate span."""

    before_start = np.column_stack((np.full(4, 20.0), np.linspace(-12.0, -3.0, 4)))
    on_segment = np.column_stack((np.full(60, 5.0), np.linspace(0.0, 60.0, 60)))
    outlier = np.array([[45.0, 45.0]])
    after_finish = np.column_stack((np.full(5, 5.0), np.linspace(100.0, 115.0, 5)))
    metric_points = np.vstack((before_start, on_segment, outlier, after_finish))

    activity = ActivityTrack(
        activity_id=202,
        points=[(0.0, 0.0)] * metric_points.shape[0],
        timestamps_s=list(range(metric_points.shape[0])),
    )
    return PreparedActivityTrack(
        activity=activity,
        latlon_points=activity.points,
        metric_points=metric_points,
        simplified_points=metric_points,
        resampled_points=metric_points,
        transformer=cast(Any, object()),
        simplification_tolerance_m=1.0,
        resample_interval_m=5.0,
        simplification_capped=False,
        resample_capped=False,
    )


def test_threshold_filtered_offset_overrides_outlier() -> None:
    """Ensure threshold filtering suppresses extreme offsets within the gate."""

    prepared_segment = _build_prepared_segment()
    prepared_activity = _build_prepared_activity()

    coverage, diagnostics, _timing_bounds, _timing_indices, _gate_hints = (
        _compute_coverage_diagnostics(
            prepared_activity,
            prepared_segment,
            max_offset_threshold=30.0,
            start_tolerance_m=30.0,
        )
    )

    assert diagnostics["raw_max_offset_m"] > 30.0
    assert diagnostics["gate_trimmed_max_offset_m"] > 30.0
    assert diagnostics["threshold_filtered_max_offset_m"] == pytest.approx(
        5.0, abs=1e-6
    )
    assert coverage.max_offset_m == pytest.approx(5.0, abs=1e-6)
