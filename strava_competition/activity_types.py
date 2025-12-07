"""Utilities for classifying Strava activity types."""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["normalize_activity_type", "activity_type_matches"]


def normalize_activity_type(value: Any) -> str | None:
    """Return a lowercase activity type string or ``None`` when missing.

    The Strava API can return either ``type`` or ``sport_type`` values, often
    with inconsistent casing. Normalising once keeps downstream comparisons
    cheap and deterministic.
    """

    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def activity_type_matches(activity: Mapping[str, Any], allowed: set[str]) -> bool:
    """Return ``True`` when ``activity`` is one of the ``allowed`` types.

    Args:
        activity: Activity payload with potential ``type``/``sport_type`` keys.
        allowed: Normalised set of permitted lower-case type names.

    Returns:
        ``True`` if either Strava-provided field matches one of the allowed
        values, otherwise ``False``. An empty ``allowed`` set implies no
        filtering should occur.
    """

    if not allowed:
        return True
    for key in ("sport_type", "type"):
        normalized = normalize_activity_type(activity.get(key))
        if normalized and normalized in allowed:
            return True
    return False
