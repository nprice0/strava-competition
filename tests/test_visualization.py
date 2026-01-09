"""Tests for the visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List

import folium
import pytest

from strava_competition.tools.geometry.models import ActivityTrack, SegmentGeometry
from strava_competition.tools.geometry.preprocessing import (
    prepare_activity,
    prepare_geometry,
)
from strava_competition.tools.geometry.visualization import (
    create_deviation_map,
    _DIVERGENCE_COLOR,
)
from strava_competition.tools.geometry.validation import compute_coverage


@pytest.fixture
def simple_segment() -> SegmentGeometry:
    """Return a short synthetic segment for plotting tests."""

    points = [
        (51.4800, -3.1800),
        (51.4805, -3.1800),
        (51.4810, -3.1800),
    ]
    return SegmentGeometry(segment_id=42, points=points, distance_m=400.0)


@pytest.fixture
def deviating_activity() -> ActivityTrack:
    """Return an activity that diverges significantly in the middle."""

    base_points: List[tuple[float, float]] = [
        (51.4800, -3.1800),
        (51.4802, -3.1800),
        (51.4804, -3.1800),
        (51.4806, -3.1815),
        (51.4808, -3.1815),
        (51.4810, -3.1800),
    ]
    timestamps = [float(idx) for idx in range(len(base_points))]
    return ActivityTrack(activity_id=321, points=base_points, timestamps_s=timestamps)


def test_create_deviation_map_highlights_divergence(
    simple_segment: SegmentGeometry, deviating_activity: ActivityTrack, tmp_path: Path
) -> None:
    """Verify that the map overlay includes a highlighted divergent section."""

    prepared_segment = prepare_geometry(
        simple_segment,
        simplification_tolerance_m=5.0,
        resample_interval_m=5.0,
    )
    prepared_activity = prepare_activity(
        deviating_activity,
        prepared_segment.transformer,
        simplification_tolerance_m=5.0,
        resample_interval_m=5.0,
    )
    coverage = compute_coverage(
        prepared_activity.metric_points, prepared_segment.resampled_points
    )

    output_path = tmp_path / "overlay.html"
    map_object = create_deviation_map(
        prepared_activity,
        prepared_segment,
        coverage,
        threshold_m=50.0,
        gate_slice=None,
        output_html_path=output_path,
    )

    assert isinstance(map_object, folium.Map)
    assert output_path.exists(), "Expected the HTML map output to be written"

    polyline_colors = {
        child.options.get("color")
        for child in map_object._children.values()
        if isinstance(child, folium.vector_layers.PolyLine)
    }
    assert _DIVERGENCE_COLOR in polyline_colors, (
        "Divergent section should be highlighted"
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Max deviation" in html
