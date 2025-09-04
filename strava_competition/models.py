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
