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
    strava_id: int
    refresh_token: str
    team: str
    access_token: str = None

@dataclass
class SegmentResult:
    runner: str
    team: str
    segment: str
    attempts: int
    fastest_time: float
    fastest_date: datetime
