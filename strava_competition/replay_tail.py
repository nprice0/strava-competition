"""Utilities supporting hybrid replay + live refresh for /athlete/activities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class ActivityStats:
    """Aggregate metadata for a cached activity payload."""

    count: int
    latest: datetime | None
    oldest: datetime | None


def parse_activity_timestamp(activity: dict) -> datetime | None:
    """Return the UTC ``start_date`` for an activity, if present."""

    raw = activity.get("start_date")
    if not raw:
        return None
    try:
        if isinstance(raw, str) and raw.endswith("Z"):
            raw = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def summarize_activities(activities: Iterable[dict]) -> ActivityStats:
    """Compute count/oldest/latest timestamps for cached activities."""

    latest: datetime | None = None
    oldest: datetime | None = None
    count = 0
    for act in activities:
        count += 1
        ts = parse_activity_timestamp(act)
        if ts is None:
            continue
        if latest is None or ts > latest:
            latest = ts
        if oldest is None or ts < oldest:
            oldest = ts
    return ActivityStats(count=count, latest=latest, oldest=oldest)


def dedupe_activities(activities: Iterable[dict]) -> List[dict]:
    """Return activities de-duplicated by Strava ID while preserving order."""

    merged: List[dict] = []
    seen: set[int] = set()
    for act in activities:
        act_id = act.get("id")
        try:
            normalized = int(act_id)
        except (TypeError, ValueError):
            normalized = None
        if normalized is not None:
            if normalized in seen:
                continue
            seen.add(normalized)
        merged.append(act)
    return merged


def merge_activity_lists(*lists: Sequence[dict]) -> List[dict]:
    """Concatenate multiple lists giving precedence to earlier arguments."""

    merged: List[dict] = []
    seen: set[int] = set()
    for block in lists:
        for act in block:
            act_id = act.get("id")
            try:
                normalized = int(act_id)
            except (TypeError, ValueError):
                normalized = None
            if normalized is not None and normalized in seen:
                continue
            if normalized is not None:
                seen.add(normalized)
            merged.append(act)
    return merged


def clamp_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """Clamp the requested window to ``now`` and ensure start <= end."""

    if start > end:
        start, end = end, start
    now = datetime.now(timezone.utc)
    if end > now:
        end = now
    return start, end


def chunk_activities(
    activities: Sequence[dict],
    *,
    chunk_size: int = 200,
) -> List[List[dict]]:
    """Split merged activities into Strava-sized pages."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [
        list(activities[i : i + chunk_size])
        for i in range(0, len(activities), chunk_size)
    ]


def cache_is_stale(captured_at: datetime | None, ttl_days: int | None) -> bool:
    """Return True when cache TTL is exceeded (0/None disables TTL)."""

    if ttl_days is None or ttl_days <= 0:
        return False
    if captured_at is None:
        return False
    now = datetime.now(timezone.utc)
    delta = now - captured_at
    return delta.total_seconds() >= ttl_days * 86400


def exceeds_lookback(latest: datetime | None, max_days: int | None) -> bool:
    """Return True when the latest cached activity is beyond the lookback."""

    if max_days is None or max_days <= 0:
        return False
    if latest is None:
        return False
    now = datetime.now(timezone.utc)
    delta = now - latest
    return delta.total_seconds() >= max_days * 86400


__all__ = [
    "ActivityStats",
    "parse_activity_timestamp",
    "summarize_activities",
    "dedupe_activities",
    "merge_activity_lists",
    "clamp_window",
    "chunk_activities",
    "cache_is_stale",
    "exceeds_lookback",
]
