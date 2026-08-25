"""Shared GPX text helpers for the GPX-producing CLI tools.

De-duplicates XML escaping and coordinate formatting between
``fetch_activity_gps`` and ``fetch_segment_gpx``, and validates that
coordinates are numeric before they are interpolated into GPX attributes.
"""

from __future__ import annotations

from typing import Any


def escape_xml(text: str) -> str:
    """Escape special XML characters for use in element text."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def coerce_coordinates(lat: Any, lon: Any) -> tuple[float, float]:
    """Cast a latitude/longitude pair to floats.

    Args:
        lat: Raw latitude value from a Strava payload.
        lon: Raw longitude value from a Strava payload.

    Returns:
        The (lat, lon) pair as floats.

    Raises:
        ValueError: If either value is not numeric.
    """
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid GPS coordinate pair: lat={lat!r} lon={lon!r}"
        ) from exc


def format_point_attrs(lat: Any, lon: Any) -> str:
    """Format a track/route point's lat/lon XML attributes.

    Coordinates are validated via :func:`coerce_coordinates` so malformed
    stream data fails loudly instead of producing broken GPX.

    Raises:
        ValueError: If either coordinate is not numeric.
    """
    lat_f, lon_f = coerce_coordinates(lat, lon)
    return f'lat="{lat_f}" lon="{lon_f}"'


__all__ = ["escape_xml", "coerce_coordinates", "format_point_attrs"]
