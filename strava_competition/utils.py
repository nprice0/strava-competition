"""General utility helpers shared across modules."""

from __future__ import annotations

import json
import logging
from datetime import datetime, date, timezone
from decimal import Decimal
from typing import Any, Optional


def format_time(seconds: int) -> str:
    """Format seconds into a ``Xm Ys`` string."""

    mins, sec = divmod(seconds, 60)
    return f"{mins}m {sec}s"


def to_utc_aware(dt: datetime) -> datetime:
    """Return a UTC-aware datetime regardless of input.

    Naive datetimes are assumed to be UTC. Aware datetimes are converted
    to UTC. This avoids TypeError when mixing naive & aware datetimes and
    provides deterministic ordering for comparisons.

    Accepts pandas Timestamps (via to_pydatetime) transparently.

    Args:
        dt: A datetime object (naive or timezone-aware).

    Returns:
        A timezone-aware datetime in UTC.
    """
    # Pandas Timestamp compatibility
    if hasattr(dt, "to_pydatetime"):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_iso_datetime(value: str | None) -> Optional[datetime]:
    """Parse an ISO 8601 datetime string, handling 'Z' suffix.

    Args:
        value: ISO datetime string (e.g., "2024-01-15T10:30:00Z")

    Returns:
        Parsed datetime or None if parsing fails or value is empty.
    """
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        logging.getLogger(__name__).debug(
            "Failed to parse ISO datetime '%s': %s", value, exc
        )
        return None


def _normalise_value(value: Any) -> Any:
    """Convert objects to JSON-friendly representations."""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, set):
        return sorted(_normalise_value(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_normalise_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalise_value(val) for key, val in value.items()}
    return value


def json_dumps_sorted(value: Any) -> str:
    """Return canonical JSON for hashing / comparisons."""

    normalised = _normalise_value(value)
    return json.dumps(normalised, sort_keys=True, separators=(",", ":"))


def coerce_int(value: Any) -> int | None:
    """Attempt to coerce a value to int, returning None on failure.

    Args:
        value: Any value to convert.

    Returns:
        The integer value, or None if conversion fails.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_float(value: Any) -> float | None:
    """Attempt to coerce a value to float, returning None on failure.

    Args:
        value: Any value to convert.

    Returns:
        The float value, or None if conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
