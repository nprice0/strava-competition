"""Dataclasses for the activity-based segment fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass(slots=True)
class ActivityScanResult:
    """Summary of attempts discovered by scanning runner activities."""

    segment_id: int
    attempts: int
    fastest_elapsed: float
    fastest_effort_id: int | str | None
    fastest_activity_id: int | None
    fastest_start_date: datetime | None
    moving_time: float | None
    effort_ids: List[int | str] = field(default_factory=list)
    inspected_activities: List[Dict[str, object]] = field(default_factory=list)
    birthday_bonus_applied: bool = False
