"""GPS geometry processing utilities for deviation map visualization.

This module provides tools for processing GPS coordinates, computing
coverage metrics, and generating interactive deviation maps.
"""

from .models import ActivityTrack, SegmentGeometry, MatchResult, Tolerances, LatLon
from .preprocessing import (
    PreparedActivityTrack,
    PreparedSegmentGeometry,
    prepare_activity,
    prepare_geometry,
)
from .validation import CoverageResult, compute_coverage
from .fetchers import fetch_activity_stream, fetch_segment_geometry
from .visualization import create_deviation_map, DeviationSummary

__all__ = [
    "ActivityTrack",
    "SegmentGeometry",
    "MatchResult",
    "Tolerances",
    "LatLon",
    "PreparedActivityTrack",
    "PreparedSegmentGeometry",
    "prepare_activity",
    "prepare_geometry",
    "CoverageResult",
    "compute_coverage",
    "fetch_activity_stream",
    "fetch_segment_geometry",
    "create_deviation_map",
    "DeviationSummary",
]
