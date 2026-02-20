"""Distance / elevation aggregation helpers.

Pure transformation: given runners, distance windows and an activity cache
it produces a list of (sheet_name, rows) including a final summary sheet.
Formerly ``distance_competition.py`` – renamed for clarity (separates
aggregation from orchestration in ``DistanceService``).
"""

from __future__ import annotations

from datetime import datetime

from .models import Runner
from .utils import parse_iso_datetime

Activity = (
    dict  # minimal structure expected: distance, total_elevation_gain, start_date_local
)

# Lookup cache keyed by id(act) to avoid mutating shared activity dicts.
_start_local_cache: dict[int, datetime] = {}


def _to_naive(dt: datetime) -> datetime:
    """Strip timezone info so both sides of a comparison are naive."""
    return dt.replace(tzinfo=None)


def _activity_start_local(act: Activity) -> datetime | None:
    """Return the activity's *local* start datetime as a naive value.

    Uses ``start_date_local`` by preference because the distance windows
    originate from an Excel file in the athlete's local timezone.  This
    mirrors the segment-scanner's ``_parse_effort_date`` which also uses
    local-to-local comparison.

    Falls back to ``start_date`` (true UTC) only when
    ``start_date_local`` is absent, and strips the timezone so that the
    comparison remains naive-to-naive (the offset error is bounded by the
    athlete's UTC offset — a few hours at window boundaries).
    """
    raw = act.get("start_date_local") or act.get("start_date")
    if not isinstance(raw, str) or not raw:
        return None
    dt = parse_iso_datetime(raw)
    if dt is None:
        return None
    return _to_naive(dt)


def _activity_in_window(act: Activity, start_dt: datetime, end_dt: datetime) -> bool:
    """Check whether *act* falls within the (inclusive) window.

    Both *start_dt* / *end_dt* and the activity timestamp are naive
    datetimes in the athlete's local timezone so the comparison is
    local-to-local (consistent with the segment scanner).
    """
    start_local = _start_local_cache.get(id(act))
    if start_local is None:
        start_local = _activity_start_local(act)
        if start_local is not None:
            _start_local_cache[id(act)] = start_local
    if start_local is None:
        return False
    return start_dt <= start_local <= end_dt


def _normalize_windows(
    distance_windows: list[tuple[datetime, datetime, float | None]],
) -> list[tuple[datetime, datetime, float | None]]:
    """Strip timezone info from window boundaries so comparisons are naive.

    Excel-sourced dates are typically already naive (local time).  If any
    have tzinfo attached we strip it to keep everything in the same naive
    local domain.
    """
    return [
        (_to_naive(s), _to_naive(e), threshold) for s, e, threshold in distance_windows
    ]


def _sheet_name_for_window(start_dt: datetime, end_dt: datetime) -> str:
    return f"Distance_{start_dt.date()}_{end_dt.date()}"


def _row_for_window(
    runner: Runner,
    acts: list[Activity],
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


def _summary_row(runner: Runner, acts: list[Activity]) -> dict:
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
    runners: list[Runner],
    distance_windows: list[tuple[datetime, datetime, float | None]],
    runner_activity_cache: dict[int | str, list[Activity]],
) -> list[tuple[str, list[dict]]]:
    """Return list of (sheet_name, rows) including Distance_Summary last.

    Summary is computed from the full activity cache (unique activities per
    runner) and is NOT a sum of the per-window sheets (avoids double counting
    when windows overlap).
    """
    # Clear the id()-keyed lookup cache between invocations so that stale
    # object-identity keys from a previous call cannot return wrong results.
    _start_local_cache.clear()

    outputs: list[tuple[str, list[dict]]] = []

    # Per-window sheets
    normalised_windows = _normalize_windows(distance_windows)
    for start_dt, end_dt, threshold in normalised_windows:
        rows: list[dict] = []
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
    summary_rows: list[dict] = []
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
