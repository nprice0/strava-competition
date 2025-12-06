"""Dataclasses describing GPS geometry inputs and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


LatLon = Tuple[float, float]


@dataclass(slots=True)
class SegmentGeometry:
    """Normalized representation of a Strava segment's geometry and metadata."""

    segment_id: int
    points: List[LatLon]
    distance_m: float
    polyline: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActivityTrack:
    """Normalized lat/lon stream with timestamps for a single activity."""

    activity_id: int
    points: List[LatLon]
    timestamps_s: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Tolerances:
    """Configuration for geometric thresholds used during visualization."""

    start_tolerance_m: float = 30.0
    frechet_tolerance_m: float = 20.0
    coverage_threshold: float = 0.95
    simplification_tolerance_m: float = 7.5
    resample_interval_m: float = 5.0


@dataclass(slots=True)
class MatchResult:
    """Aggregate outcome for coverage analysis of an activity against a segment."""

    matched: bool
    score: Optional[float] = None
    max_deviation_m: Optional[float] = None
    coverage_ratio: Optional[float] = None
    elapsed_time_s: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
