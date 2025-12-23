#!/usr/bin/env python3
"""Fetch GPS coordinates for a specific Strava activity.

This tool uses the Strava Streams API to retrieve GPS points (latitude/longitude)
for a given activity ID. It can output as JSON or GPX format.

Environment requirements:
- ``STRAVA_CLIENT_ID`` and ``STRAVA_CLIENT_SECRET`` must be set (or stored in
  ``.env``) so the refresh flow can succeed.
- ``requests`` is already declared in ``requirements.txt`` for the project.

Usage examples:

    # Basic usage - outputs JSON with lat/lng points
    python -m strava_competition.tools.fetch_activity_gps \
        --refresh-token YOUR_REFRESH_TOKEN \
        --activity-id 12345678

    # Output as GPX file
    python -m strava_competition.tools.fetch_activity_gps \
        --refresh-token YOUR_REFRESH_TOKEN \
        --activity-id 12345678 \
        --output-format gpx \
        --output-file activity.gpx

    # Include additional stream data (altitude, time, distance)
    python -m strava_competition.tools.fetch_activity_gps \
        --refresh-token YOUR_REFRESH_TOKEN \
        --activity-id 12345678 \
        --include-altitude \
        --include-time
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from strava_competition.auth import get_access_token
from strava_competition.config import REQUEST_TIMEOUT, STRAVA_BASE_URL

# Default output directory for GPX files
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "gpx_output"
)

LOGGER = logging.getLogger("fetch_activity_gps")


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


def fetch_activity_streams(
    token: str,
    activity_id: int,
    *,
    include_altitude: bool = False,
    include_time: bool = False,
    include_distance: bool = False,
) -> Dict[str, List[Any]]:
    """Fetch GPS stream data for an activity.

    Args:
        token: Strava access token.
        activity_id: Strava activity ID.
        include_altitude: Include elevation data.
        include_time: Include time offsets from activity start.
        include_distance: Include cumulative distance.

    Returns:
        Dict mapping stream keys to data arrays. Always includes ``latlng``.
    """
    keys = ["latlng"]
    if include_altitude:
        keys.append("altitude")
    if include_time:
        keys.append("time")
    if include_distance:
        keys.append("distance")

    url = f"{STRAVA_BASE_URL}/activities/{activity_id}/streams"
    params = {
        "keys": ",".join(keys),
        "key_by_type": "true",
    }

    payload = _http_get(url, token, params=params)

    # Strava keys responses by stream type when key_by_type is set
    result: Dict[str, List[Any]] = {}
    if isinstance(payload, dict):
        for key in keys:
            if key in payload and "data" in payload[key]:
                result[key] = payload[key]["data"]
    else:
        LOGGER.warning("Unexpected streams response format: %s", type(payload))

    return result


def fetch_activity_metadata(token: str, activity_id: int) -> Dict[str, Any]:
    """Fetch basic activity metadata for GPX header info."""
    url = f"{STRAVA_BASE_URL}/activities/{activity_id}"
    data = _http_get(url, token)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response type: {type(data)}")
    return data


def streams_to_gpx(
    streams: Dict[str, List[Any]],
    metadata: Dict[str, Any],
) -> str:
    """Convert stream data to GPX format.

    Args:
        streams: Dict containing at minimum ``latlng`` stream.
        metadata: Activity metadata for name and timestamps.

    Returns:
        GPX XML string.
    """
    latlng = streams.get("latlng", [])
    altitude = streams.get("altitude", [])
    time_offsets = streams.get("time", [])

    activity_name = metadata.get("name", "Strava Activity")
    start_date_str = metadata.get("start_date")

    # Parse start time to compute absolute timestamps for each point
    start_time: Optional[datetime] = None
    if start_date_str:
        if start_date_str.endswith("Z"):
            start_date_str = start_date_str[:-1] + "+00:00"
        try:
            start_time = datetime.fromisoformat(start_date_str)
        except ValueError:
            LOGGER.warning("Could not parse start_date: %s", start_date_str)

    # Build GPX XML
    gpx_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="strava_competition"',
        '     xmlns="http://www.topografix.com/GPX/1/1"',
        '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '     xsi:schemaLocation="http://www.topografix.com/GPX/1/1 '
        'http://www.topografix.com/GPX/1/1/gpx.xsd">',
        "  <metadata>",
        f"    <name>{_escape_xml(activity_name)}</name>",
    ]

    if start_time:
        gpx_lines.append(f"    <time>{start_time.isoformat()}</time>")

    gpx_lines.extend(
        [
            "  </metadata>",
            "  <trk>",
            f"    <name>{_escape_xml(activity_name)}</name>",
            "    <trkseg>",
        ]
    )

    for i, point in enumerate(latlng):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue

        lat, lng = point[0], point[1]
        ele = altitude[i] if i < len(altitude) else None
        time_offset = time_offsets[i] if i < len(time_offsets) else None

        trkpt_attrs = f'lat="{lat}" lon="{lng}"'
        gpx_lines.append(f"      <trkpt {trkpt_attrs}>")

        if ele is not None:
            gpx_lines.append(f"        <ele>{ele}</ele>")

        if start_time and time_offset is not None:
            point_time = start_time.timestamp() + time_offset
            point_dt = datetime.fromtimestamp(point_time, tz=timezone.utc)
            gpx_lines.append(f"        <time>{point_dt.isoformat()}</time>")

        gpx_lines.append("      </trkpt>")

    gpx_lines.extend(
        [
            "    </trkseg>",
            "  </trk>",
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
        description="Fetch GPS coordinates for a Strava activity using the Streams API"
    )
    parser.add_argument(
        "--activity-id",
        type=int,
        required=True,
        help="Strava activity ID to fetch GPS data for",
    )
    parser.add_argument(
        "--refresh-token",
        required=True,
        help="Refresh token for the activity owner (required)",
    )
    parser.add_argument(
        "--runner-name",
        default="Unknown Runner",
        help="Label used only for logging",
    )
    parser.add_argument(
        "--no-altitude",
        action="store_true",
        help="Exclude altitude/elevation data from output",
    )
    parser.add_argument(
        "--no-time",
        action="store_true",
        help="Exclude time offsets from output",
    )
    parser.add_argument(
        "--no-distance",
        action="store_true",
        help="Exclude cumulative distance from output",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "gpx"],
        default="gpx",
        help="Output format (default: gpx)",
    )
    parser.add_argument(
        "--output-file",
        help="Output file path (default: gpx_output/activity_<id>.gpx)",
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
    """Entry point for the fetch_activity_gps tool."""
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    LOGGER.info("Fetching GPS data for activity %s", args.activity_id)

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

    # Fetch streams (all included by default, use --no-* flags to exclude)
    streams = fetch_activity_streams(
        access_token,
        args.activity_id,
        include_altitude=not args.no_altitude,
        include_time=not args.no_time,
        include_distance=not args.no_distance,
    )

    latlng = streams.get("latlng", [])
    LOGGER.info(
        "Retrieved %d GPS points for activity %s", len(latlng), args.activity_id
    )

    if not latlng:
        LOGGER.warning("No GPS data found for activity %s", args.activity_id)
        return

    # Format output
    if args.output_format == "gpx":
        metadata = fetch_activity_metadata(access_token, args.activity_id)
        output = streams_to_gpx(streams, metadata)
    else:
        # JSON output - include metadata summary
        output = json.dumps(
            {
                "activity_id": args.activity_id,
                "point_count": len(latlng),
                "streams": streams,
            },
            indent=2,
        )

    # Determine output path
    if args.no_file:
        print(output)
    else:
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            # Default to gpx_output/activity_<id>.gpx
            ext = "json" if args.output_format == "json" else "gpx"
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = DEFAULT_OUTPUT_DIR / f"activity_{args.activity_id}.{ext}"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        LOGGER.info("Output written to %s", output_path)


if __name__ == "__main__":
    main()
