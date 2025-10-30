from dataclasses import dataclass
from datetime import datetime


@dataclass
class Segment:
    id: int
    name: str
    start_date: datetime
    end_date: datetime


@dataclass
class Runner:
    name: str
    strava_id: str
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
    fastest_date: datetime
