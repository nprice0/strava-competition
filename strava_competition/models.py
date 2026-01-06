from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import warnings


@dataclass
class Segment:
    """Deprecated: Use SegmentGroup with SegmentWindow instead."""

    id: int
    name: str
    start_date: datetime
    end_date: datetime
    default_time_seconds: float | None = None
    min_distance_meters: float | None = None
    birthday_bonus_seconds: float | None = None

    def __post_init__(self) -> None:
        warnings.warn(
            "Segment is deprecated; use SegmentGroup with SegmentWindow instead.",
            DeprecationWarning,
            stacklevel=2,
        )


@dataclass
class SegmentWindow:
    """A single date window within a segment group."""

    start_date: datetime
    end_date: datetime
    label: str | None = None
    birthday_bonus_seconds: float = 0.0


@dataclass
class SegmentGroup:
    """A segment with one or more date windows."""

    id: int
    name: str
    windows: List[SegmentWindow]
    default_time_seconds: float | None = None
    min_distance_meters: float | None = None


@dataclass
class Runner:
    name: str
    strava_id: int | str
    refresh_token: str
    # Separate optional teams for segment and distance series
    segment_team: str | None = None
    distance_team: str | None = None
    access_token: str | None = None
    # Set to True after first 402 Payment Required so we can skip further API calls
    payment_required: bool = False
    birthday: tuple[int, int] | None = None


@dataclass
class SegmentResult:
    runner: str
    team: str
    segment: str
    attempts: int
    fastest_time: float
    fastest_date: Optional[datetime]
    birthday_bonus_applied: bool = False
    source: str = "strava"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    fastest_distance_m: float | None = None
