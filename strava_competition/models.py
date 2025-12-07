from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Segment:
    id: int
    name: str
    start_date: datetime
    end_date: datetime
    default_time_seconds: float | None = None


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


@dataclass
class SegmentResult:
    runner: str
    team: str
    segment: str
    attempts: int
    fastest_time: float
    fastest_date: Optional[datetime]
    source: str = "strava"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
