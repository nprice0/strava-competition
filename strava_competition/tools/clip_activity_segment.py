#!/usr/bin/env python3
"""Clip a range of GPX track points into a standalone file.

This helper lets you reproduce a Strava segment effort locally by
extracting the exact track-point slice (either by index or by time) from
an activity GPX. It mirrors the ad-hoc script previously used to produce
`activity_16919797941_segment_40641291.gpx`, but packaged as a reusable
CLI tool.
"""

from __future__ import annotations

import argparse
import copy
import os
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from defusedxml import ElementTree as ET

from strava_competition.auth import get_access_token
from strava_competition.config import STRAVA_BASE_URL
from strava_competition.tools._http import http_get as _http_get

GPX_NS = {"g": "http://www.topografix.com/GPX/1/1"}
ET.register_namespace("", GPX_NS["g"])
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

DEFAULT_GPX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gpx_output"

LOGGER = logging.getLogger("clip_activity_segment")


@dataclass
class RemoteActivityFetcher:
    """Fetch activity metadata, streams, and derived indices from Strava."""

    activity_id: int
    refresh_token: str
    runner_name: str = "Unknown Runner"
    runner_id: int | None = None

    _access_token: str | None = None
    _activity_detail: dict[str, Any] | None = None

    def access_token(self) -> str:
        """Return a cached access token, refreshing when required."""
        if self._access_token is None:
            token, maybe_new = get_access_token(
                self.refresh_token,
                runner_name=self.runner_name,
            )
            if maybe_new and maybe_new != self.refresh_token:
                LOGGER.warning(
                    "Strava rotated the refresh token (ends …%s). "
                    "Update your credentials.",
                    maybe_new[-4:],
                )
            self._access_token = token
        return self._access_token

    def activity_detail(self) -> dict[str, Any]:
        """Load the activity detail payload with include_all_efforts=true."""
        if self._activity_detail is None:
            url = f"{STRAVA_BASE_URL}/activities/{self.activity_id}"
            params = {"include_all_efforts": "true"}
            self._activity_detail = _http_get(url, self.access_token(), params=params)
            if not isinstance(self._activity_detail, dict):
                raise SystemExit(
                    "Unexpected response for activity detail; expected object"
                )
        return self._activity_detail

    def download_gpx(
        self,
        *,
        cache_dir: Path = DEFAULT_GPX_DIR,
        force: bool = False,
    ) -> Path:
        """Download and cache the activity GPX file."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / f"activity_{self.activity_id}.gpx"
        if output_path.exists() and not force:
            LOGGER.debug(
                "Using cached GPX for activity %s at %s",
                self.activity_id,
                output_path,
            )
            return output_path

        LOGGER.info(
            "Fetching GPS streams for activity %s to generate GPX",
            self.activity_id,
        )
        url = f"{STRAVA_BASE_URL}/activities/{self.activity_id}/streams"
        params = {
            "keys": "latlng,time,altitude",
            "key_by_type": "true",
        }
        payload = _http_get(url, self.access_token(), params=params)
        if not isinstance(payload, dict):
            raise SystemExit("Streams response was not an object as expected")

        latlng_stream = payload.get("latlng", {})
        time_stream = payload.get("time", {})
        altitude_stream = payload.get("altitude", {})

        latlng_data = (
            latlng_stream.get("data") if isinstance(latlng_stream, dict) else None
        )
        time_data = time_stream.get("data") if isinstance(time_stream, dict) else None
        altitude_data = (
            altitude_stream.get("data") if isinstance(altitude_stream, dict) else None
        )

        if not latlng_data or not isinstance(latlng_data, list):
            raise SystemExit("latlng stream missing from Strava response")

        activity = self.activity_detail()
        activity_name = activity.get("name") or f"Activity {self.activity_id}"
        start_date_str = activity.get("start_date")
        start_dt = parse_iso8601(start_date_str) if start_date_str else None

        tree = build_gpx_tree(
            activity_name,
            start_dt,
            latlng_data,
            altitude_data if isinstance(altitude_data, list) else None,
            time_data if isinstance(time_data, list) else None,
        )
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        LOGGER.info("Saved activity GPX to %s", output_path)
        return output_path

    def indices_for_segment(self, segment_id: int) -> tuple[int, int]:
        """Return the start/end indices for the requested segment effort."""
        activity = self.activity_detail()
        efforts = activity.get("segment_efforts") or []
        for effort in efforts:
            segment = effort.get("segment") or {}
            if segment.get("id") != segment_id:
                continue
            try:
                start_idx = int(effort["start_index"])
                end_idx = int(effort["end_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit("Segment effort missing start/end index data") from exc
            return start_idx, end_idx
        raise SystemExit(
            "Segment effort not found in Strava activity detail for the provided segment ID"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip a GPX activity down to the points that form a segment effort."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Source GPX file containing the full activity. Optional when "
            "--activity-id and --refresh-token are supplied so the GPX can "
            "be downloaded automatically."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination GPX path. Defaults to <input>_segment.gpx inside "
            "the same folder."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        help="Zero-based index of the first track point to keep",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Zero-based index of the last track point to keep (inclusive)",
    )
    parser.add_argument(
        "--start-time",
        help="ISO timestamp (e.g. 2026-01-03T06:23:44+00:00) for the first "
        "point to keep",
    )
    parser.add_argument(
        "--end-time",
        help="ISO timestamp for the final point to keep",
    )
    parser.add_argument(
        "--elapsed",
        type=int,
        help="Window length in seconds when pairing with --start-time",
    )
    parser.add_argument(
        "--segment-efforts-json",
        type=Path,
        help=(
            "Path to a captured segment efforts JSON payload (as returned by "
            "Strava's /segment_efforts API). When provided together with "
            "--activity-id, the tool auto-detects start/end indices."
        ),
    )
    parser.add_argument(
        "--activity-id",
        type=int,
        help="Activity ID to match when deriving indices from --segment-efforts-json",
    )
    parser.add_argument(
        "--segment-id",
        type=int,
        help=(
            "Optional segment ID filter when using --segment-efforts-json; "
            "required for fully automatic mode without manual indices."
        ),
    )
    parser.add_argument(
        "--refresh-token",
        default=os.environ.get("STRAVA_REFRESH_TOKEN"),
        help=(
            "Runner refresh token used to fetch GPX/effort data when local "
            "files are not provided. "
            "Defaults to STRAVA_REFRESH_TOKEN env var."
        ),
    )
    parser.add_argument(
        "--runner-id",
        type=int,
        help="Optional Strava athlete ID for logging/capture metadata",
    )
    parser.add_argument(
        "--runner-name",
        default="Unknown Runner",
        help="Runner name used for logging (default: Unknown Runner)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_GPX_DIR,
        help=(
            "Directory for auto-downloaded GPX files when --input is omitted "
            "(default: data/gpx_output)."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the activity GPX even if it already exists locally",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def parse_iso8601(value: str) -> datetime:
    """Parse ISO timestamps and normalise trailing Z."""

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - user input validation
        raise SystemExit(f"Invalid ISO timestamp: {value}") from exc
    if dt.tzinfo is None:
        raise SystemExit("Timestamp must include a timezone offset")
    return dt


def load_trackpoints(path: Path) -> list[ET.Element]:
    tree = ET.parse(path)
    root = tree.getroot()
    trkseg = root.find(".//g:trkseg", GPX_NS)
    if trkseg is None:
        raise SystemExit("GPX file is missing a <trkseg> block")
    points = cast(list[ET.Element], trkseg.findall("g:trkpt", GPX_NS))
    if not points:
        raise SystemExit("GPX file contains no track points")
    return points


def resolve_indices(
    points: list[ET.Element],
    *,
    start_index: int | None,
    end_index: int | None,
    start_time: datetime | None,
    end_time: datetime | None,
    elapsed: int | None,
    efforts_path: Path | None,
    activity_id: int | None,
    segment_id: int | None,
) -> tuple[int, int]:
    """Convert CLI parameters into concrete slice indices."""

    uses_index = start_index is not None or end_index is not None
    uses_time = start_time is not None or end_time is not None or elapsed
    uses_json = efforts_path is not None

    modes_selected = sum(bool(x) for x in (uses_index, uses_time, uses_json))
    if modes_selected != 1:
        raise SystemExit(
            "Specify exactly one of: (start/end indices), (start/end times), "
            "or --segment-efforts-json"
        )

    if uses_json:
        if activity_id is None:
            raise SystemExit(
                "--activity-id is required when using --segment-efforts-json"
            )
        if efforts_path is None:  # mypy: guarded by uses_json
            raise SystemExit("--segment-efforts-json must be provided")
        start_index, end_index = _indices_from_segment_efforts(
            efforts_path, activity_id=activity_id, segment_id=segment_id
        )
        return start_index, end_index

    if uses_index:
        if start_index is None or end_index is None:
            raise SystemExit("Both --start-index and --end-index are required")
        if start_index < 0 or end_index < 0:
            raise SystemExit("Indices must be non-negative")
        if start_index > end_index:
            raise SystemExit("start-index must be <= end-index")
        if end_index >= len(points):
            raise SystemExit("end-index exceeds number of track points")
        return start_index, end_index

    if not uses_time:
        raise SystemExit(
            "Supply --start/--end times or --elapsed when not using indices"
            " or --segment-efforts-json"
        )
    if start_time is None:
        raise SystemExit("--start-time is required when slicing by time")
    if end_time is None:
        if elapsed is None:
            raise SystemExit("Specify --end-time or --elapsed (seconds)")
        end_time = start_time + timedelta(seconds=elapsed)

    point_times: list[datetime] = []
    for idx, pt in enumerate(points):
        time_el = pt.find("g:time", GPX_NS)
        if time_el is None or time_el.text is None:
            raise SystemExit(
                f"Track point {idx} is missing a timestamp; cannot slice by time"
            )
        point_times.append(parse_iso8601(time_el.text))
    try:
        start_idx = next(idx for idx, ts in enumerate(point_times) if ts >= start_time)
    except StopIteration as exc:
        raise SystemExit("Start time is after the final GPX point") from exc

    try:
        end_idx = max(idx for idx, ts in enumerate(point_times) if ts <= end_time)
    except ValueError as exc:  # pragma: no cover - indicates no points before end
        raise SystemExit("End time precedes the first GPX point") from exc

    if start_idx > end_idx:
        raise SystemExit("Time window does not contain any track points")
    return start_idx, end_idx


def build_output_tree(
    source: Path,
    points: list[ET.Element],
    start_idx: int,
    end_idx: int,
) -> ET.ElementTree[ET.Element]:
    tree: ET.ElementTree[ET.Element] = ET.parse(source)
    root = tree.getroot()
    trkseg = root.find(".//g:trkseg", GPX_NS)
    if trkseg is None:
        raise SystemExit(
            "GPX source is missing a <trkseg> block even though track points"
            " were parsed"
        )
    # Remove existing points and insert the slice while keeping metadata intact.
    for child in list(trkseg):
        trkseg.remove(child)
    for pt in points[start_idx : end_idx + 1]:
        trkseg.append(copy.deepcopy(pt))

    # Update metadata time so exported GPX starts at the first kept sample.
    first_time = trkseg.find("g:trkpt/g:time", GPX_NS)
    metadata_time = root.find("g:metadata/g:time", GPX_NS)
    if metadata_time is not None and first_time is not None:
        metadata_time.text = first_time.text
    return tree


def build_gpx_tree(
    activity_name: str,
    start_dt: datetime | None,
    latlng_data: list[Any],
    altitude_data: list[Any] | None,
    time_offsets: list[Any] | None,
) -> ET.ElementTree:
    attrs = {
        "version": "1.1",
        "creator": "strava_competition",
        "xmlns": GPX_NS["g"],
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd",
    }
    gpx = ET.Element("gpx", attrs)
    metadata = ET.SubElement(gpx, "metadata")
    name_el = ET.SubElement(metadata, "name")
    name_el.text = activity_name
    if start_dt is not None:
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        meta_time = ET.SubElement(metadata, "time")
        meta_time.text = start_dt.isoformat()

    trk = ET.SubElement(gpx, "trk")
    trk_name = ET.SubElement(trk, "name")
    trk_name.text = activity_name
    trkseg = ET.SubElement(trk, "trkseg")

    for idx, coords in enumerate(latlng_data):
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        lat, lon = coords[0], coords[1]
        trkpt = ET.SubElement(
            trkseg,
            "trkpt",
            {"lat": str(lat), "lon": str(lon)},
        )
        if (
            altitude_data
            and idx < len(altitude_data)
            and altitude_data[idx] is not None
        ):
            ele = ET.SubElement(trkpt, "ele")
            ele.text = str(altitude_data[idx])
        if (
            start_dt is not None
            and time_offsets
            and idx < len(time_offsets)
            and time_offsets[idx] is not None
        ):
            point_time = start_dt + timedelta(seconds=float(time_offsets[idx]))
            time_el = ET.SubElement(trkpt, "time")
            time_el.text = point_time.isoformat()

    return ET.ElementTree(gpx)


def _indices_from_segment_efforts(
    json_path: Path,
    *,
    activity_id: int,
    segment_id: int | None = None,
) -> tuple[int, int]:
    if not json_path.exists():  # pragma: no cover - user input validation
        raise SystemExit(f"Segment efforts file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "response" in payload:
        efforts = payload["response"]
    else:
        efforts = payload

    if not isinstance(efforts, list):  # pragma: no cover
        raise SystemExit("Segment efforts JSON must contain a list or 'response' list")

    for effort in efforts:
        activity = effort.get("activity", {}) if isinstance(effort, dict) else {}
        segment = effort.get("segment", {}) if isinstance(effort, dict) else {}
        if activity.get("id") != activity_id:
            continue
        if segment_id is not None and segment.get("id") != segment_id:
            continue
        try:
            start_idx = int(effort["start_index"])
            end_idx = int(effort["end_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit("Segment effort entry lacks start/end indices") from exc
        return start_idx, end_idx

    raise SystemExit(
        "No segment effort found in JSON for the provided activity/segment IDs"
    )


def main() -> None:  # pragma: no cover - CLI glue
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    remote_fetcher: RemoteActivityFetcher | None = None
    if args.refresh_token and args.activity_id:
        remote_fetcher = RemoteActivityFetcher(
            activity_id=args.activity_id,
            refresh_token=args.refresh_token,
            runner_name=args.runner_name,
            runner_id=args.runner_id,
        )

    source_path = args.input
    if source_path is None:
        if remote_fetcher is None:
            raise SystemExit(
                "Provide --input or supply --activity-id + --refresh-token "
                "so the GPX can be downloaded automatically."
            )
        source_path = remote_fetcher.download_gpx(
            cache_dir=args.download_dir,
            force=args.force_download,
        )

    points = load_trackpoints(source_path)

    auto_segment_mode = (
        args.segment_efforts_json is None
        and args.start_index is None
        and args.end_index is None
        and args.start_time is None
        and args.end_time is None
        and args.elapsed is None
        and remote_fetcher is not None
        and args.segment_id is not None
    )

    if auto_segment_mode:
        if remote_fetcher is None:
            raise SystemExit("Automatic segment mode requires a remote fetcher")
        if args.segment_id is None:
            raise SystemExit("Automatic segment mode requires --segment-id")
        auto_start, auto_end = remote_fetcher.indices_for_segment(args.segment_id)
        start_idx, end_idx = resolve_indices(
            points,
            start_index=auto_start,
            end_index=auto_end,
            start_time=None,
            end_time=None,
            elapsed=None,
            efforts_path=None,
            activity_id=None,
            segment_id=None,
        )
    else:
        start_time = parse_iso8601(args.start_time) if args.start_time else None
        end_time = parse_iso8601(args.end_time) if args.end_time else None
        start_idx, end_idx = resolve_indices(
            points,
            start_index=args.start_index,
            end_index=args.end_index,
            start_time=start_time,
            end_time=end_time,
            elapsed=args.elapsed,
            efforts_path=args.segment_efforts_json,
            activity_id=args.activity_id,
            segment_id=args.segment_id,
        )

    tree = build_output_tree(source_path, points, start_idx, end_idx)
    output_path = (
        args.output
        if args.output is not None
        else source_path.with_name(f"{source_path.stem}_segment.gpx")
    )
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    LOGGER.info(
        "Wrote %d points to %s (indices %d–%d).",
        end_idx - start_idx + 1,
        output_path,
        start_idx,
        end_idx,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
