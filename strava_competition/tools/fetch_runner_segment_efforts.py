#!/usr/bin/env python3
"""Quick Strava probe for a runner's segment efforts on a specific day.

The script mirrors the curl/HTTPie flow you described:

1. Exchange the provided refresh token for a short-lived access token.
2. Call ``GET /athlete/activities`` with ``before``/``after`` filters for the
   requested day.
3. For each returned activity, call ``GET /activities/{id}?include_all_efforts=true``
   and print every segment effort so you can inspect raw Strava output.

Environment requirements:
- ``STRAVA_CLIENT_ID`` and ``STRAVA_CLIENT_SECRET`` must be set (or stored in
  ``.env``) so the refresh flow can succeed.
- ``requests`` is already declared in ``requirements.txt`` for the project.

Usage example (defaults baked in for runner 829675 and 2005-11-16):

    python -m strava_competition.tools.fetch_runner_segment_efforts \
        --refresh-token 331ef40838b2e09a39ffd884585ad5ebe3700107

You can override the runner, date, paging window, and verbosity via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Tuple

import requests

from strava_competition.auth import get_access_token
from strava_competition.config import REQUEST_TIMEOUT, STRAVA_BASE_URL

LOGGER = logging.getLogger("fetch_runner_segment_efforts")


def _parse_day(day_str: str) -> date:
    """Return a ``date`` from ``YYYY-MM-DD`` (fallback to ``YYYY-DD-MM``).

    Strava uses epoch seconds for the ``before``/``after`` filters, so we only
    need the day boundary. The extra parser guards against accidental
    day-month swaps (e.g., ``2005-16-11``).
    """

    try:
        return datetime.strptime(day_str, "%Y-%m-%d").date()
    except ValueError:
        try:
            return datetime.strptime(day_str, "%Y-%d-%m").date()
        except ValueError as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(
                "Invalid date format. Expected YYYY-MM-DD"
            ) from exc


def _day_window(day_value: date) -> Tuple[int, int]:
    """Return inclusive ``after`` and exclusive ``before`` epoch seconds."""

    start_dt = datetime.combine(day_value, time.min, tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


def _iso8601(value: str) -> datetime:
    """Parse ISO8601 strings (accepting trailing ``Z``) into UTC datetimes."""

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _window_from_args(args: argparse.Namespace) -> Tuple[int, int]:
    if args.start or args.end:
        if not args.start or not args.end:
            raise SystemExit("Both --start and --end must be supplied together")
        start_dt = _iso8601(args.start)
        end_dt = _iso8601(args.end)
        if end_dt <= start_dt:
            raise SystemExit("--end must be later than --start")
        return int(start_dt.timestamp()), int(end_dt.timestamp())
    return _day_window(args.day)


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


def _fetch_activities(
    token: str,
    after_ts: int,
    before_ts: int,
    *,
    per_page: int,
) -> List[Dict[str, Any]]:
    """Collect all activities in the requested window."""

    activities: List[Dict[str, Any]] = []
    page = 1
    while True:
        params = {
            "after": after_ts,
            "before": before_ts,
            "per_page": per_page,
            "page": page,
        }
        url = f"{STRAVA_BASE_URL}/athlete/activities"
        payload = _http_get(url, token, params=params)
        if not isinstance(payload, list):
            raise RuntimeError(
                f"Unexpected payload type for /athlete/activities: {type(payload)}"
            )
        LOGGER.info(
            "Fetched %s activities (page %s)",
            len(payload),
            page,
        )
        activities.extend(payload)
        if len(payload) < per_page:
            break
        page += 1
    return activities


def _fetch_activity_detail(token: str, activity_id: int) -> Dict[str, Any]:
    """Fetch ``/activities/{id}`` with ``include_all_efforts=true``."""

    url = f"{STRAVA_BASE_URL}/activities/{activity_id}"
    params = {"include_all_efforts": "true"}
    payload = _http_get(url, token, params=params)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Unexpected payload type for /activities/{activity_id}: {type(payload)}"
        )
    return payload


def _print_efforts(activity: Dict[str, Any], *, show_raw: bool) -> None:
    """Pretty-print the activity metadata plus each segment effort.

    Args:
        activity: JSON payload returned by ``include_all_efforts``.
        show_raw: When True, emit the full JSON payload for inspection.
    """

    activity_id = activity.get("id")
    name = activity.get("name", "<unnamed>")
    start = activity.get("start_date_local")
    effort_list = activity.get("segment_efforts") or []
    LOGGER.info(
        "Activity %s | %s | start=%s | segment_efforts=%s",
        activity_id,
        name,
        start,
        len(effort_list),
    )
    if show_raw:
        print(json.dumps(activity, ensure_ascii=False))
    for effort in effort_list:
        segment = effort.get("segment", {})
        summary = {
            "segment_id": segment.get("id"),
            "segment_name": segment.get("name"),
            "elapsed_time": effort.get("elapsed_time"),
            "moving_time": effort.get("moving_time"),
            "start_date_local": effort.get("start_date_local"),
            "is_kom": effort.get("is_kom"),
        }
        print(json.dumps(summary, ensure_ascii=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download Strava activities for a runner on a specific day and "
            "print every segment effort from include_all_efforts"
        )
    )
    parser.add_argument(
        "--runner-id",
        default="829675",
        help="Runner Strava athlete ID (default: 829675)",
    )
    parser.add_argument(
        "--runner-name",
        default="Runner 829675",
        help="Label used only for logging",
    )
    parser.add_argument(
        "--refresh-token",
        required=True,
        help="Refresh token for the runner (required)",
    )
    parser.add_argument(
        "--day",
        default="2005-11-16",
        type=_parse_day,
        help="Day to inspect (YYYY-MM-DD, default 2005-11-16)",
    )
    parser.add_argument(
        "--start",
        help="ISO8601 start datetime (overrides --day when paired with --end)",
    )
    parser.add_argument(
        "--end",
        help="ISO8601 end datetime (exclusive, required when --start is set)",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=50,
        help="Activity page size passed to /athlete/activities",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help=(
            "Emit the full include_all_efforts JSON payload for each activity "
            "(in addition to the summary rows)"
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    after_ts, before_ts = _window_from_args(args)
    if args.start:
        window_label = f"{args.start} to {args.end}"
    else:
        window_label = args.day.isoformat()
    LOGGER.info(
        "Collecting activities for runner %s (%s) (after=%s, before=%s)",
        args.runner_id,
        window_label,
        after_ts,
        before_ts,
    )

    access_token, maybe_new_refresh = get_access_token(
        args.refresh_token,
        runner_name=args.runner_name,
    )
    if maybe_new_refresh and maybe_new_refresh != args.refresh_token:
        LOGGER.warning(
            "Strava rotated the refresh token. Save this new value: %s",
            maybe_new_refresh,
        )

    activities = _fetch_activities(
        access_token,
        after_ts,
        before_ts,
        per_page=args.per_page,
    )
    if not activities:
        LOGGER.warning("No activities returned for runner %s", args.runner_id)
        return

    for act in activities:
        activity_id = act.get("id")
        if activity_id is None:
            continue
        detail = _fetch_activity_detail(access_token, int(activity_id))
        _print_efforts(detail, show_raw=args.print_json)


if __name__ == "__main__":
    main()
