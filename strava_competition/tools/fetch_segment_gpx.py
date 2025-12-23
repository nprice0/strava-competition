#!/usr/bin/env python3
"""Fetch a Strava segment and export as GPX for route sharing.

This tool retrieves segment geometry from the Strava API and converts
the encoded polyline to GPX format, which can be imported into GPS
devices, mapping apps, or uploaded to other platforms.

Environment requirements:
- ``STRAVA_CLIENT_ID`` and ``STRAVA_CLIENT_SECRET`` must be set (or stored in
  ``.env``) so the refresh flow can succeed.
- ``requests`` and ``polyline`` are declared in ``requirements.txt``.

Usage examples:

    # Export segment as GPX to stdout
    python -m strava_competition.tools.fetch_segment_gpx \
        --refresh-token YOUR_REFRESH_TOKEN \
        --segment-id 12345678

    # Save to file
    python -m strava_competition.tools.fetch_segment_gpx \
        --refresh-token YOUR_REFRESH_TOKEN \
        --segment-id 12345678 \
        --output-file segment.gpx


"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from polyline import decode as polyline_decode

from strava_competition.auth import get_access_token
from strava_competition.config import REQUEST_TIMEOUT, STRAVA_BASE_URL

# Default output directory for GPX files
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "gpx_output"
)

LOGGER = logging.getLogger("fetch_segment_gpx")


def _http_get(
    url: str,
    token: str,
    *,
    params: Dict[str, Any] | None = None,
) -> Any:
    """Wrapper for authenticated GET requests with basic logging."""
    headers = {"Authorization": f"Bearer {token}"}
    LOGGER.debug("GET %s params=%s", url, params)
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def fetch_segment(token: str, segment_id: int) -> Dict[str, Any]:
    """Fetch segment details including polyline geometry.

    Args:
        token: Strava access token.
        segment_id: Strava segment ID.

    Returns:
        Dict with segment metadata and decoded lat/lng coordinates.
    """
    url = f"{STRAVA_BASE_URL}/segments/{segment_id}"
    data = _http_get(url, token)

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response type: {type(data)}")

    map_info = data.get("map") or {}
    encoded_polyline = map_info.get("polyline") or map_info.get("summary_polyline")

    if not encoded_polyline:
        raise RuntimeError(f"Segment {segment_id} has no polyline data")

    # Decode polyline to lat/lng points
    try:
        points = polyline_decode(encoded_polyline)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Failed to decode polyline: {exc}") from exc

    return {
        "segment_id": data.get("id", segment_id),
        "name": data.get("name", f"Segment {segment_id}"),
        "distance": data.get("distance"),
        "elevation_high": data.get("elevation_high"),
        "elevation_low": data.get("elevation_low"),
        "average_grade": data.get("average_grade"),
        "start_latlng": data.get("start_latlng"),
        "end_latlng": data.get("end_latlng"),
        "points": [(float(lat), float(lon)) for lat, lon in points],
        "polyline": encoded_polyline,
    }


def segment_to_gpx(segment: Dict[str, Any]) -> str:
    """Convert segment data to GPX route format.

    Args:
        segment: Segment dict with ``name`` and ``points`` keys.

    Returns:
        GPX XML string.
    """
    name = segment.get("name", "Strava Segment")
    points: List[Tuple[float, float]] = segment.get("points", [])
    distance = segment.get("distance")
    elevation_high = segment.get("elevation_high")
    elevation_low = segment.get("elevation_low")
    average_grade = segment.get("average_grade")

    # Build description with segment metadata
    desc_parts = []
    if distance is not None:
        desc_parts.append(f"Distance: {distance:.0f}m")
    if elevation_high is not None and elevation_low is not None:
        elevation_gain = elevation_high - elevation_low
        desc_parts.append(f"Elevation gain: {elevation_gain:.0f}m")
    if average_grade is not None:
        desc_parts.append(f"Average grade: {average_grade:.1f}%")
    description = " | ".join(desc_parts) if desc_parts else ""

    gpx_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="strava_competition"',
        '     xmlns="http://www.topografix.com/GPX/1/1"',
        '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '     xsi:schemaLocation="http://www.topografix.com/GPX/1/1 '
        'http://www.topografix.com/GPX/1/1/gpx.xsd">',
        "  <metadata>",
        f"    <name>{_escape_xml(name)}</name>",
    ]

    if description:
        gpx_lines.append(f"    <desc>{_escape_xml(description)}</desc>")

    gpx_lines.extend(
        [
            "  </metadata>",
            "  <rte>",
            f"    <name>{_escape_xml(name)}</name>",
        ]
    )

    if description:
        gpx_lines.append(f"    <desc>{_escape_xml(description)}</desc>")

    # Add route points
    for lat, lon in points:
        gpx_lines.append(f'    <rtept lat="{lat}" lon="{lon}"/>')

    gpx_lines.extend(
        [
            "  </rte>",
            "</gpx>",
        ]
    )

    return "\n".join(gpx_lines)


def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch a Strava segment and export as GPX for route sharing"
    )
    parser.add_argument(
        "--segment-id",
        type=int,
        required=True,
        help="Strava segment ID to fetch",
    )
    parser.add_argument(
        "--refresh-token",
        required=True,
        help="Refresh token for authentication (required)",
    )
    parser.add_argument(
        "--runner-name",
        default="Unknown Runner",
        help="Label used only for logging",
    )
    parser.add_argument(
        "--output-file",
        help="Output file path (default: gpx_output/segment_<id>.gpx)",
    )
    parser.add_argument(
        "--no-file",
        action="store_true",
        help="Print to stdout instead of writing to file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level",
    )
    return parser


def main() -> None:
    """Entry point for the fetch_segment_gpx tool."""
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    LOGGER.info("Fetching segment %s", args.segment_id)

    # Get access token
    access_token, maybe_new_refresh = get_access_token(
        args.refresh_token,
        runner_name=args.runner_name,
    )
    if maybe_new_refresh and maybe_new_refresh != args.refresh_token:
        LOGGER.warning(
            "Strava rotated the refresh token. Save this new value: %s",
            maybe_new_refresh,
        )

    # Fetch segment
    segment = fetch_segment(access_token, args.segment_id)
    points = segment.get("points", [])
    LOGGER.info(
        "Retrieved segment '%s' with %d points",
        segment.get("name"),
        len(points),
    )

    if not points:
        LOGGER.warning("Segment has no coordinate data")
        return

    # Format output as GPX
    output = segment_to_gpx(segment)

    # Determine output path
    if args.no_file:
        print(output)
    else:
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            # Default to gpx_output/segment_<id>.gpx
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = DEFAULT_OUTPUT_DIR / f"segment_{args.segment_id}.gpx"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        LOGGER.info("Output written to %s", output_path)


if __name__ == "__main__":
    main()
