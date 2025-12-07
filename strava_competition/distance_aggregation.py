"""Distance / elevation aggregation helpers.

Pure transformation: given runners, distance windows and an activity cache
it produces a list of (sheet_name, rows) including a final summary sheet.
Formerly ``distance_competition.py`` – renamed for clarity (separates
aggregation from orchestration in ``DistanceService``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

from .models import Runner
from .utils import parse_iso_datetime, to_utc_aware

Activity = (
    dict  # minimal structure expected: distance, total_elevation_gain, start_date_local
)


def _activity_start_utc(act: Activity) -> datetime | None:
    """Return cached UTC start datetime for an activity."""

    raw: str | None = None
    start_date = act.get("start_date")
    start_local = act.get("start_date_local")
    if isinstance(start_date, str) and start_date:
        raw = start_date
    elif isinstance(start_local, str) and start_local:
        raw = start_local
    if not raw:
        return None
    dt = parse_iso_datetime(raw)
    if dt is None:
        return None
    return to_utc_aware(dt)


def _activity_in_window(act: Activity, start_dt: datetime, end_dt: datetime) -> bool:
    start_utc = act.get("_start_utc")
    if start_utc is None:
        start_utc = _activity_start_utc(act)
        if start_utc is not None:
            act["_start_utc"] = start_utc
    if start_utc is None:
        return False
    return start_dt <= start_utc <= end_dt


def _normalize_windows(
    distance_windows: List[Tuple[datetime, datetime, float | None]],
) -> List[Tuple[datetime, datetime, float | None]]:
    """Normalize all window boundaries to UTC-aware once up front."""
    return [
        (to_utc_aware(s), to_utc_aware(e), threshold)
        for s, e, threshold in distance_windows
    ]


def _sheet_name_for_window(start_dt: datetime, end_dt: datetime) -> str:
    return f"Distance_{start_dt.date()}_{end_dt.date()}"


def _row_for_window(
    runner: Runner,
    acts: List[Activity],
    start_dt: datetime,
    end_dt: datetime,
    threshold: float | None,
) -> dict:
    total_distance = 0.0
    total_elev = 0.0
    run_count = 0
    thr_count = 0
    for act in acts:
        if not _activity_in_window(act, start_dt, end_dt):
            continue
        run_count += 1
        dist = act.get("distance")
        elev = act.get("total_elevation_gain")
        if isinstance(dist, (int, float)):
            dist_val = float(dist)
            total_distance += dist_val
            if threshold is not None and dist_val / 1000.0 >= threshold:
                thr_count += 1
        if isinstance(elev, (int, float)):
            total_elev += float(elev)
    km = total_distance / 1000.0
    row: dict = {
        "Runner": runner.name,
        "Team": runner.distance_team,
        "Runs": run_count,
        "Total Distance (km)": round(km, 2),
        "Total Elev Gain (m)": round(total_elev, 1),
    }
    if threshold is not None:
        row[f"Runs >= {threshold} km"] = thr_count
    return row


def _summary_row(runner: Runner, acts: List[Activity]) -> dict:
    total_distance = 0.0
    total_elev = 0.0
    run_count = 0
    for act in acts:
        dist = act.get("distance")
        elev = act.get("total_elevation_gain")
        if isinstance(dist, (int, float)):
            total_distance += float(dist)
        if isinstance(elev, (int, float)):
            total_elev += float(elev)
        run_count += 1
    km_total = total_distance / 1000.0
    return {
        "Runner": runner.name,
        "Team": runner.distance_team,
        "Total Runs": run_count,
        "Total Distance (km)": round(km_total, 2),
        "Total Elev Gain (m)": round(total_elev, 1),
        "Avg Distance per Run (km)": round(km_total / run_count, 2)
        if run_count
        else 0.0,
    }


def build_distance_outputs(
    runners: List[Runner],
    distance_windows: List[Tuple[datetime, datetime, float | None]],
    runner_activity_cache: Dict[int | str, List[Activity]],
) -> List[Tuple[str, List[dict]]]:
    """Return list of (sheet_name, rows) including Distance_Summary last.

    Summary is computed from the full activity cache (unique activities per
    runner) and is NOT a sum of the per-window sheets (avoids double counting
    when windows overlap).
    """
    outputs: List[Tuple[str, List[dict]]] = []

    # Per-window sheets
    normalised_windows = _normalize_windows(distance_windows)
    for start_dt, end_dt, threshold in normalised_windows:
        rows: List[dict] = []
        for runner in runners:
            if not runner.distance_team:
                continue
            acts = runner_activity_cache.get(runner.strava_id, [])
            rows.append(_row_for_window(runner, acts, start_dt, end_dt, threshold))
        rows.sort(
            key=lambda r: (
                -r["Total Distance (km)"],
                -r["Total Elev Gain (m)"],
                r["Runner"],
            )
        )
        outputs.append((_sheet_name_for_window(start_dt, end_dt), rows))

    # Summary from full activity cache (unique activities only once)
    summary_rows: List[dict] = []
    for runner in runners:
        if not runner.distance_team:
            continue
        acts = runner_activity_cache.get(runner.strava_id, [])
        summary_rows.append(_summary_row(runner, acts))
    summary_rows.sort(
        key=lambda r: (
            -r["Total Distance (km)"],
            -r["Total Elev Gain (m)"],
            r["Runner"],
        )
    )
    outputs.append(("Distance_Summary", summary_rows))
    return outputs


__all__ = ["build_distance_outputs"]
