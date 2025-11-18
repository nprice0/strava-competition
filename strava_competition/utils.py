"""General utility helpers shared across modules."""

from __future__ import annotations

import json
from datetime import datetime, date
from decimal import Decimal
from typing import Any


def format_time(seconds: int) -> str:
    """Format seconds into a ``Xm Ys`` string."""

    mins, sec = divmod(seconds, 60)
    return f"{mins}m {sec}s"


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
