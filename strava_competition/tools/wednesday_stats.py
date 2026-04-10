"""Club run statistics for a group of runners.

Reads runners from the standard competition Excel workbook, fetches their
activities over a configurable date range, filters by day-of-week and
optional start-time window, and prints per-runner and group statistics.
Optionally writes results to an Excel file.

Usage:
    # Wednesday club runs starting between 19:25 and 19:40:
    python -m strava_competition.tools.wednesday_stats \
        --input competition_input.xlsx --start 2022-01-01 --end 2026-04-10 \
        --day wednesday --time-from 19:25 --time-to 19:40

    # Any day, no time filter, with Excel output:
    python -m strava_competition.tools.wednesday_stats \
        --input competition_input.xlsx --start 2022-01-01 --end 2026-04-10 \
        --day thursday --output results.xlsx

Environment variables required:
    STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET  (for token refresh)
    STRAVA_API_CACHE_MODE  (optional – defaults to 'cache')
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, time as dt_time, timedelta
from itertools import combinations
from typing import Any, TypedDict

import pandas as pd

from ..excel_reader import read_runners, workbook_context
from ..models import Runner
from ..strava_api import StravaClient, get_activities, get_default_client
from ..utils import parse_iso_datetime

LOGGER = logging.getLogger(__name__)


class _RunnerStatsRequired(TypedDict):
    """Required fields for per-runner aggregate statistics."""

    runner: str
    team: str
    total_runs: int
    total_km: float
    total_elevation_m: float
    avg_km: float
    avg_elevation_m: float
    longest_run_km: float
    total_moving_time: str
    avg_pace_per_km: str
    fastest_pace_per_km: str
    _avg_pace_seconds: float | None
    longest_streak_weeks: int
    attendance_pct: float
    total_target_days: int
    first_run: str
    last_run: str


class RunnerStats(_RunnerStatsRequired, total=False):
    """Per-runner statistics with optional ranking fields.

    Ranking fields are added by ``add_rankings`` after initial computation.
    """

    rank_distance: int
    rank_attendance: int
    rank_pace: int
    rank_streak: int


_DAY_NAMES: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _parse_date(value: str) -> datetime:
    """Parse a YYYY-MM-DD date string into a naive datetime."""
    return datetime.strptime(value, "%Y-%m-%d")


def _parse_time(value: str) -> dt_time:
    """Parse an HH:MM time string into a ``datetime.time``."""
    return datetime.strptime(value, "%H:%M").time()


def _parse_day(value: str) -> int:
    """Parse a day name to a weekday int (Monday=0).

    Raises:
        ValueError: If the day name is not recognised.
    """
    key = value.strip().lower()
    if key not in _DAY_NAMES:
        raise ValueError(
            f"Unknown day '{value}'. Expected one of: {', '.join(_DAY_NAMES)}"
        )
    return _DAY_NAMES[key]


def _day_label(weekday: int) -> str:
    """Return the capitalised English name for a weekday int."""
    for name, num in _DAY_NAMES.items():
        if num == weekday:
            return name.capitalize()
    return str(weekday)


def _activity_start_local(act: dict[str, Any]) -> datetime | None:
    """Return the activity's local start as a naive datetime."""
    raw = act.get("start_date_local") or act.get("start_date")
    if not isinstance(raw, str) or not raw:
        return None
    dt = parse_iso_datetime(raw)
    if dt is None:
        return None
    return dt.replace(tzinfo=None)


def _matches_day_and_time(
    act: dict[str, Any],
    weekday: int,
    time_from: dt_time | None,
    time_to: dt_time | None,
) -> bool:
    """Return True if the activity matches the day and optional time window.

    Args:
        act: Raw activity dict from Strava.
        weekday: Target weekday (Monday=0 … Sunday=6).
        time_from: Earliest start time (inclusive), or None to skip.
        time_to: Latest start time (inclusive), or None to skip.
    """
    dt = _activity_start_local(act)
    if dt is None or dt.weekday() != weekday:
        return False
    if time_from is not None and dt.time() < time_from:
        return False
    if time_to is not None and dt.time() > time_to:
        return False
    return True


def _elapsed_time_display(seconds: float) -> str:
    """Format seconds into H:MM:SS."""
    h, remainder = divmod(int(seconds), 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _pace_per_km(distance_m: float, elapsed_s: float) -> str:
    """Return pace as M:SS /km."""
    if distance_m <= 0 or elapsed_s <= 0:
        return "-"
    pace_s = elapsed_s / (distance_m / 1000.0)
    m, s = divmod(int(pace_s), 60)
    return f"{m}:{s:02d}"


def fetch_matching_activities(
    runner: Runner,
    start_date: datetime,
    end_date: datetime,
    weekday: int,
    time_from: dt_time | None = None,
    time_to: dt_time | None = None,
    *,
    client: StravaClient | None = None,
) -> list[dict[str, Any]]:
    """Fetch runs matching the day-of-week and optional start-time window.

    Args:
        runner: Runner to fetch activities for.
        start_date: Window start (inclusive).
        end_date: Window end (inclusive).
        weekday: Target weekday (Monday=0 … Sunday=6).
        time_from: Earliest start time (inclusive), or None.
        time_to: Latest start time (inclusive), or None.
        client: Optional pre-initialised Strava client.
    """
    if client is None:
        client = get_default_client()
    client.ensure_runner_token(runner)

    activities = get_activities(runner, start_date, end_date, activity_types=("Run",))
    if activities is None:
        LOGGER.warning("Failed to fetch activities for %s", runner.name)
        return []

    return [
        a for a in activities if _matches_day_and_time(a, weekday, time_from, time_to)
    ]


def _count_target_days(start: datetime, end: datetime, weekday: int) -> int:
    """Count occurrences of *weekday* between *start* and *end* inclusive."""
    days_until = (weekday - start.weekday()) % 7
    first = start + timedelta(days=days_until)
    if first > end:
        return 0
    return (end - first).days // 7 + 1


def _accumulate_activity_metrics(
    matched_acts: list[dict[str, Any]],
) -> tuple[float, float, float, float, float | None, list[datetime]]:
    """Sum distance, elevation, time and track fastest pace across activities.

    Args:
        matched_acts: Activity dicts matching the day/time filter.

    Returns:
        Tuple of (total_distance_m, total_elevation_m, total_moving_s,
        longest_distance_m, fastest_pace_s_per_km, dates).
    """
    total_distance_m = 0.0
    total_elevation_m = 0.0
    total_moving_s = 0.0
    longest_distance_m = 0.0
    fastest_pace_s_per_km: float | None = None
    dates: list[datetime] = []

    for act in matched_acts:
        dist = float(act.get("distance", 0) or 0)
        elev = float(act.get("total_elevation_gain", 0) or 0)
        moving = float(act.get("moving_time", 0) or 0)

        total_distance_m += dist
        total_elevation_m += elev
        total_moving_s += moving

        if dist > longest_distance_m:
            longest_distance_m = dist

        if dist > 0 and moving > 0:
            pace = moving / (dist / 1000.0)
            if fastest_pace_s_per_km is None or pace < fastest_pace_s_per_km:
                fastest_pace_s_per_km = pace

        dt = _activity_start_local(act)
        if dt:
            dates.append(dt)

    return (
        total_distance_m,
        total_elevation_m,
        total_moving_s,
        longest_distance_m,
        fastest_pace_s_per_km,
        dates,
    )


def _longest_weekly_streak(sorted_dates: list[datetime]) -> int:
    """Return the longest run of consecutive weekly target days (7-day gaps)."""
    max_streak = 0
    current_streak = 0
    for i, d in enumerate(sorted_dates):
        if i == 0:
            current_streak = 1
        elif (d.date() - sorted_dates[i - 1].date()).days == 7:
            current_streak += 1
        else:
            current_streak = 1
        max_streak = max(max_streak, current_streak)
    return max_streak


def compute_runner_stats(
    runner: Runner,
    matched_acts: list[dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
    weekday: int,
) -> RunnerStats:
    """Compute aggregate stats for one runner's matched runs."""
    (
        total_distance_m,
        total_elevation_m,
        total_moving_s,
        longest_distance_m,
        fastest_pace_s_per_km,
        dates,
    ) = _accumulate_activity_metrics(matched_acts)

    run_count = len(matched_acts)
    total_km = total_distance_m / 1000.0
    avg_km = total_km / run_count if run_count else 0.0
    avg_elev = total_elevation_m / run_count if run_count else 0.0

    sorted_dates = sorted(dates)
    max_streak = _longest_weekly_streak(sorted_dates)

    total_target_days = _count_target_days(start_date, end_date, weekday)
    attendance_pct = (
        round(run_count / total_target_days * 100, 1) if total_target_days > 0 else 0.0
    )

    return {
        "runner": runner.name,
        "team": runner.distance_team or runner.segment_team or "-",
        "total_runs": run_count,
        "total_km": round(total_km, 2),
        "total_elevation_m": round(total_elevation_m, 1),
        "avg_km": round(avg_km, 2),
        "avg_elevation_m": round(avg_elev, 1),
        "longest_run_km": round(longest_distance_m / 1000.0, 2),
        "total_moving_time": _elapsed_time_display(total_moving_s),
        "avg_pace_per_km": _pace_per_km(total_distance_m, total_moving_s),
        "fastest_pace_per_km": (
            _pace_per_km(1000.0, fastest_pace_s_per_km)
            if fastest_pace_s_per_km
            else "-"
        ),
        "_avg_pace_seconds": (
            total_moving_s / (total_distance_m / 1000.0)
            if total_distance_m > 0 and total_moving_s > 0
            else None
        ),
        "longest_streak_weeks": max_streak,
        "attendance_pct": attendance_pct,
        "total_target_days": total_target_days,
        "first_run": sorted_dates[0].strftime("%Y-%m-%d") if sorted_dates else "-",
        "last_run": sorted_dates[-1].strftime("%Y-%m-%d") if sorted_dates else "-",
    }


def _assign_rank(
    values: list[float],
    descending: bool = True,
) -> list[int]:
    """Return 1-based ordinal ranks for *values*.

    Args:
        values: Numeric values to rank.
        descending: If True, highest value gets rank 1.
    """
    indexed = sorted(
        enumerate(values),
        key=lambda iv: iv[1],
        reverse=descending,
    )
    ranks = [0] * len(values)
    for rank, (orig_idx, _) in enumerate(indexed, start=1):
        ranks[orig_idx] = rank
    return ranks


def add_rankings(all_stats: list[RunnerStats]) -> None:
    """Mutate *all_stats* in place to add ranking columns.

    Ranks by: total distance, attendance %, avg pace, best streak.
    """
    if not all_stats:
        return
    km_ranks = _assign_rank([s["total_km"] for s in all_stats], descending=True)
    att_ranks = _assign_rank(
        [s["attendance_pct"] for s in all_stats],
        descending=True,
    )
    streak_ranks = _assign_rank(
        [float(s["longest_streak_weeks"]) for s in all_stats],
        descending=True,
    )
    # Pace: lower is better → ascending rank on raw seconds.
    pace_seconds: list[float] = [
        s["_avg_pace_seconds"] if s["_avg_pace_seconds"] is not None else 99999.0
        for s in all_stats
    ]
    pace_ranks = _assign_rank(pace_seconds, descending=False)

    for i, s in enumerate(all_stats):
        s["rank_distance"] = km_ranks[i]
        s["rank_attendance"] = att_ranks[i]
        s["rank_pace"] = pace_ranks[i]
        s["rank_streak"] = streak_ranks[i]


def _group_activities_by_month(
    acts: list[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Group activities by (year, month) using local start time."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for act in acts:
        dt = _activity_start_local(act)
        if dt:
            grouped[(dt.year, dt.month)].append(act)
    return grouped


def _group_activities_by_year(
    acts: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Group activities by year using local start time."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for act in acts:
        dt = _activity_start_local(act)
        if dt:
            grouped[dt.year].append(act)
    return grouped


def _period_row(
    runner_name: str,
    period_label: str,
    acts: list[dict[str, Any]],
    total_target_days: int,
) -> dict[str, Any]:
    """Build a single summary row for a time period (month or year).

    Re-accumulates metrics from *acts* for this period.  This trades a
    small amount of redundant iteration for keeping the period-row logic
    self-contained and independent of the caller's aggregation.

    Args:
        runner_name: Display name.
        period_label: e.g. "2024-03" or "2024".
        acts: Matched activities in this period.
        total_target_days: Number of target days in this period.
    """
    (
        total_dist_m,
        total_elev_m,
        total_moving_s,
        longest_m,
        _fastest_pace,
        _dates,
    ) = _accumulate_activity_metrics(acts)

    run_count = len(acts)
    total_km = total_dist_m / 1000.0
    attendance = (
        round(run_count / total_target_days * 100, 1) if total_target_days > 0 else 0.0
    )
    return {
        "Runner": runner_name,
        "Period": period_label,
        "Runs": run_count,
        "Target Days": total_target_days,
        "Attendance %": attendance,
        "Total Distance (km)": round(total_km, 2),
        "Total Elevation (m)": round(total_elev_m, 1),
        "Avg Distance (km)": round(total_km / run_count, 2) if run_count else 0.0,
        "Avg Pace (/km)": _pace_per_km(total_dist_m, total_moving_s),
        "Longest Run (km)": round(longest_m / 1000.0, 2),
    }


def compute_monthly_breakdown(
    runner: Runner,
    matched_acts: list[dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
    weekday: int,
) -> list[dict[str, Any]]:
    """Return one row per month the runner was active (or had target days).

    Args:
        runner: Runner instance.
        matched_acts: All matched runs for the runner.
        start_date: Overall start of the query window.
        end_date: Overall end of the query window.
        weekday: Target weekday (Monday=0 … Sunday=6).
    """
    by_month = _group_activities_by_month(matched_acts)

    # Walk every month in range so we show 0-run months too.
    rows: list[dict[str, Any]] = []
    cursor = datetime(start_date.year, start_date.month, 1)
    while cursor <= end_date:
        year, month = cursor.year, cursor.month
        # Month boundaries for target-day count
        month_start = max(cursor, start_date)
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        month_end = min(next_month - timedelta(days=1), end_date)
        days_in_month = _count_target_days(month_start, month_end, weekday)

        acts = by_month.get((year, month), [])
        label = f"{year}-{month:02d}"
        rows.append(_period_row(runner.name, label, acts, days_in_month))
        cursor = next_month
    return rows


def compute_yearly_breakdown(
    runner: Runner,
    matched_acts: list[dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
    weekday: int,
) -> list[dict[str, Any]]:
    """Return one row per year the runner was active.

    Args:
        runner: Runner instance.
        matched_acts: All matched runs for the runner.
        start_date: Overall start of the query window.
        end_date: Overall end of the query window.
        weekday: Target weekday (Monday=0 … Sunday=6).
    """
    by_year = _group_activities_by_year(matched_acts)

    rows: list[dict[str, Any]] = []
    for year in range(start_date.year, end_date.year + 1):
        year_start = max(datetime(year, 1, 1), start_date)
        year_end = min(datetime(year, 12, 31), end_date)
        days_in_year = _count_target_days(year_start, year_end, weekday)
        acts = by_year.get(year, [])
        rows.append(_period_row(runner.name, str(year), acts, days_in_year))
    return rows


def _activity_avg_speed_kmh(act: dict[str, Any]) -> float | None:
    """Return average speed in km/h from activity, or None if unavailable."""
    speed = act.get("average_speed")
    if isinstance(speed, (int, float)) and speed > 0:
        return float(speed) * 3.6  # m/s → km/h
    dist = float(act.get("distance", 0) or 0)
    moving = float(act.get("moving_time", 0) or 0)
    if dist > 0 and moving > 0:
        return (dist / 1000.0) / (moving / 3600.0)
    return None


def _tagged_activity(
    runner_name: str,
    act: dict[str, Any],
) -> tuple[str, str, str]:
    """Return (runner_name, date_str, activity_name) for a given activity."""
    dt = _activity_start_local(act)
    date_str = dt.strftime("%Y-%m-%d") if dt else "?"
    return runner_name, date_str, act.get("name", "")


def _find_max_distance(
    runner_activities: dict[str, list[dict[str, Any]]],
) -> dict[str, str] | None:
    """Find the single run with the greatest distance."""
    best: tuple[float, str, str, str] | None = None
    for name, acts in runner_activities.items():
        for act in acts:
            dist_km = float(act.get("distance", 0) or 0) / 1000.0
            if dist_km > 0 and (best is None or dist_km > best[0]):
                runner, date, act_name = _tagged_activity(name, act)
                best = (dist_km, runner, date, act_name)
    if best is None:
        return None
    return {
        "Record": "Most Distance (single run)",
        "Value": f"{best[0]:.2f} km",
        "Runner": best[1],
        "Date": best[2],
        "Activity": best[3],
    }


def _find_longest_time(
    runner_activities: dict[str, list[dict[str, Any]]],
) -> dict[str, str] | None:
    """Find the single run with the longest moving time."""
    best: tuple[float, str, str, str] | None = None
    for name, acts in runner_activities.items():
        for act in acts:
            moving_s = float(act.get("moving_time", 0) or 0)
            if moving_s > 0 and (best is None or moving_s > best[0]):
                runner, date, act_name = _tagged_activity(name, act)
                best = (moving_s, runner, date, act_name)
    if best is None:
        return None
    return {
        "Record": "Longest Run (by time)",
        "Value": _elapsed_time_display(best[0]),
        "Runner": best[1],
        "Date": best[2],
        "Activity": best[3],
    }


def _iter_speeds(
    runner_activities: dict[str, list[dict[str, Any]]],
) -> list[tuple[float, str, str, str]]:
    """Return (speed_kmh, runner, date, activity_name) for valid activities."""
    results: list[tuple[float, str, str, str]] = []
    for name, acts in runner_activities.items():
        for act in acts:
            speed = _activity_avg_speed_kmh(act)
            if speed is not None and speed > 0:
                runner, date, act_name = _tagged_activity(name, act)
                results.append((speed, runner, date, act_name))
    return results


def _speed_record_row(
    label: str,
    entry: tuple[float, str, str, str],
) -> dict[str, str]:
    """Format a speed tuple into a record row dict."""
    return {
        "Record": label,
        "Value": f"{entry[0]:.2f} km/h",
        "Runner": entry[1],
        "Date": entry[2],
        "Activity": entry[3],
    }


def _find_speed_extremes(
    runner_activities: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    """Find the fastest and slowest single-run average speeds."""
    speeds = _iter_speeds(runner_activities)
    if not speeds:
        return None, None

    fastest = max(speeds, key=lambda s: s[0])
    slowest = min(speeds, key=lambda s: s[0])
    return (
        _speed_record_row("Fastest Avg Speed", fastest),
        _speed_record_row("Slowest Avg Speed", slowest),
    )


def compute_records(
    runner_activities: dict[str, list[dict[str, Any]]],
) -> list[dict[str, str]]:
    """Compute group-wide records across all runners' matched activities.

    Returns a list of dicts with keys: Record, Value, Runner, Date, Activity.

    Records computed:
        - Most distance in a single run
        - Longest run by time
        - Fastest average speed
        - Slowest average speed
    """
    rows: list[dict[str, str]] = []
    dist_record = _find_max_distance(runner_activities)
    if dist_record:
        rows.append(dist_record)
    time_record = _find_longest_time(runner_activities)
    if time_record:
        rows.append(time_record)
    fast_record, slow_record = _find_speed_extremes(runner_activities)
    if fast_record:
        rows.append(fast_record)
    if slow_record:
        rows.append(slow_record)
    return rows


def compute_running_pairs(
    runner_activities: dict[str, list[dict[str, Any]]],
    top_n: int = 15,
) -> list[dict[str, Any]]:
    """Find the most common pairs of runners who ran on the same day.

    Two runners are counted as a "pair" if they both have a matched run
    on the same date.

    Args:
        runner_activities: Mapping of runner name → matched activities.
        top_n: How many pairs to return.

    Returns:
        List of dicts with keys: Runner A, Runner B, Shared Runs.
    """
    # Build a set of matching dates per runner.
    runner_dates: dict[str, set[str]] = {}
    for name, acts in runner_activities.items():
        dates: set[str] = set()
        for act in acts:
            dt = _activity_start_local(act)
            if dt:
                dates.add(dt.strftime("%Y-%m-%d"))
        runner_dates[name] = dates

    pair_counts: Counter[tuple[str, str]] = Counter()
    for (a, dates_a), (b, dates_b) in combinations(runner_dates.items(), 2):
        shared = len(dates_a & dates_b)
        if shared > 0:
            pair = (a, b) if a < b else (b, a)
            pair_counts[pair] = shared

    rows: list[dict[str, Any]] = []
    for (runner_a, runner_b), count in pair_counts.most_common(top_n):
        rows.append(
            {
                "Runner A": runner_a,
                "Runner B": runner_b,
                "Shared Runs": count,
            }
        )
    return rows


def print_stats(
    all_stats: list[RunnerStats],
    day_label: str,
    records: list[dict[str, str]] | None = None,
    pairs: list[dict[str, Any]] | None = None,
) -> None:
    """Print a formatted table of stats to stdout."""
    if not all_stats:
        print(f"No {day_label} runs found.")
        return

    sorted_stats = sorted(all_stats, key=lambda s: -s["total_km"])

    # Header
    print(f"\n{'=' * 100}")
    print(f"{day_label.upper()} RUN STATISTICS")
    print(f"{'=' * 100}\n")

    for s in sorted_stats:
        print(f"  {s['runner']} ({s['team']})")
        print(
            f"    Runs:            {s['total_runs']} / {s['total_target_days']} {day_label}s ({s['attendance_pct']}%)"
        )
        print(f"    Total Distance:  {s['total_km']} km")
        print(f"    Total Elevation: {s['total_elevation_m']} m")
        print(f"    Avg Distance:    {s['avg_km']} km")
        print(f"    Avg Elevation:   {s['avg_elevation_m']} m")
        print(f"    Longest Run:     {s['longest_run_km']} km")
        print(f"    Total Moving:    {s['total_moving_time']}")
        print(f"    Avg Pace:        {s['avg_pace_per_km']} /km")
        print(f"    Fastest Pace:    {s['fastest_pace_per_km']} /km")
        print(f"    Best Streak:     {s['longest_streak_weeks']} consecutive weeks")
        print(f"    Active Period:   {s['first_run']} → {s['last_run']}")
        if "rank_distance" in s:
            n = len(all_stats)
            print(
                f"    Rankings:        distance #{s['rank_distance']}/{n}  "
                f"attendance #{s['rank_attendance']}/{n}  "
                f"pace #{s['rank_pace']}/{n}  "
                f"streak #{s['rank_streak']}/{n}"
            )
        print()

    # Group totals
    total_runs = sum(s["total_runs"] for s in all_stats)
    total_km = sum(s["total_km"] for s in all_stats)
    total_elev = sum(s["total_elevation_m"] for s in all_stats)

    print(f"  {'─' * 50}")
    print(f"  GROUP TOTALS ({len(all_stats)} runners)")
    print(f"    Total Runs:      {total_runs}")
    print(f"    Total Distance:  {round(total_km, 2)} km")
    print(f"    Total Elevation: {round(total_elev, 1)} m")
    print()

    if records:
        print(f"  {'─' * 50}")
        print("  RECORDS")
        for r in records:
            print(f"    {r['Record']}: {r['Value']} — {r['Runner']} ({r['Date']})")
        print()

    if pairs:
        print(f"  {'─' * 50}")
        print("  MOST COMMON RUNNING PAIRS")
        for p in pairs:
            print(
                f"    {p['Runner A']} & {p['Runner B']}: "
                f"{p['Shared Runs']} shared {day_label}s"
            )
        print()


_EXCEL_COLUMNS: dict[str, str] = {
    "runner": "Runner",
    "team": "Team",
    "total_runs": "Total Runs",
    "total_target_days": "Total Target Days",
    "attendance_pct": "Attendance %",
    "total_km": "Total Distance (km)",
    "total_elevation_m": "Total Elevation (m)",
    "avg_km": "Avg Distance (km)",
    "avg_elevation_m": "Avg Elevation (m)",
    "longest_run_km": "Longest Run (km)",
    "total_moving_time": "Total Moving Time",
    "avg_pace_per_km": "Avg Pace (/km)",
    "fastest_pace_per_km": "Fastest Pace (/km)",
    "longest_streak_weeks": "Best Streak (weeks)",
    "first_run": "First Run",
    "last_run": "Last Run",
    "rank_distance": "Rank: Distance",
    "rank_attendance": "Rank: Attendance",
    "rank_pace": "Rank: Pace",
    "rank_streak": "Rank: Streak",
}


def write_excel(
    all_stats: list[RunnerStats],
    monthly_rows: list[dict[str, Any]],
    yearly_rows: list[dict[str, Any]],
    records: list[dict[str, str]],
    pairs: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Write stats to an Excel file with multiple sheets.

    Args:
        all_stats: Per-runner stat dicts from ``compute_runner_stats``.
        monthly_rows: All monthly breakdown rows across runners.
        yearly_rows: All yearly breakdown rows across runners.
        records: Group-wide records (fastest, longest, etc.).
        pairs: Most common running pairs.
        output_path: Destination ``.xlsx`` path.
    """
    sorted_stats = sorted(all_stats, key=lambda s: -s["total_km"])
    df_summary = pd.DataFrame(sorted_stats)
    # Drop internal columns not meant for Excel output.
    df_summary = df_summary.drop(
        columns=[c for c in df_summary.columns if c.startswith("_")],
    )
    df_summary = df_summary.rename(columns=_EXCEL_COLUMNS)
    df_monthly = pd.DataFrame(monthly_rows)
    df_yearly = pd.DataFrame(yearly_rows)
    df_records = pd.DataFrame(records)
    df_pairs = pd.DataFrame(pairs)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="Summary")
        df_monthly.to_excel(writer, index=False, sheet_name="Monthly Breakdown")
        df_yearly.to_excel(writer, index=False, sheet_name="Year over Year")
        df_records.to_excel(writer, index=False, sheet_name="Records")
        df_pairs.to_excel(writer, index=False, sheet_name="Running Pairs")

    LOGGER.info(
        "Wrote %d runners, %d monthly rows, %d yearly rows, %d records, %d pairs to %s",
        len(df_summary),
        len(df_monthly),
        len(df_yearly),
        len(df_records),
        len(df_pairs),
        output_path,
    )


def _fetch_all_runner_stats(
    runners: list[Runner],
    start_date: datetime,
    end_date: datetime,
    weekday: int,
    day_name: str,
    time_from: dt_time | None,
    time_to: dt_time | None,
) -> tuple[list[RunnerStats], dict[str, list[dict[str, Any]]]]:
    """Fetch activities for all runners and compute per-runner stats.

    Runners whose fetch fails are logged and skipped.

    Args:
        runners: Runners to process.
        start_date: Window start (inclusive).
        end_date: Window end (inclusive).
        weekday: Target weekday (Monday=0 … Sunday=6).
        day_name: Capitalised day name for logging.
        time_from: Earliest start time (inclusive), or None.
        time_to: Latest start time (inclusive), or None.

    Returns:
        Tuple of (all_stats, runner_activities) where *all_stats* is a list
        of per-runner stat dicts and *runner_activities* maps runner names
        to their matched activity lists.
    """
    all_stats: list[RunnerStats] = []
    runner_activities: dict[str, list[dict[str, Any]]] = {}
    client = get_default_client()
    for runner in runners:
        LOGGER.info("Fetching activities for %s …", runner.name)
        try:
            matched = fetch_matching_activities(
                runner,
                start_date,
                end_date,
                weekday,
                time_from,
                time_to,
                client=client,
            )
        except Exception:
            LOGGER.exception(
                "Failed to fetch activities for %s, skipping",
                runner.name,
            )
            continue
        LOGGER.info("  → %d %s runs", len(matched), day_name)
        stats = compute_runner_stats(
            runner,
            matched,
            start_date,
            end_date,
            weekday,
        )
        all_stats.append(stats)
        runner_activities[runner.name] = matched
    return all_stats, runner_activities


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Club run statistics for a group of runners.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the competition Excel workbook.",
    )
    parser.add_argument(
        "--start",
        "-s",
        required=True,
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        "-e",
        required=True,
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--day",
        "-d",
        default="wednesday",
        help="Day of week to filter on (default: wednesday).",
    )
    parser.add_argument(
        "--time-from",
        default=None,
        help="Earliest start time HH:MM, inclusive (e.g. 19:25).",
    )
    parser.add_argument(
        "--time-to",
        default=None,
        help="Latest start time HH:MM, inclusive (e.g. 19:40).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output Excel file path (e.g. wednesday_results.xlsx).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level (default: INFO).",
    )
    return parser


def main() -> None:
    """Entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_path = args.input
    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    if start_date >= end_date:
        parser.error("--start must be before --end")
    weekday = _parse_day(args.day)
    day_name = _day_label(weekday)
    time_from = _parse_time(args.time_from) if args.time_from else None
    time_to = _parse_time(args.time_to) if args.time_to else None

    time_desc = ""
    if time_from or time_to:
        f_str = time_from.strftime("%H:%M") if time_from else "*"
        t_str = time_to.strftime("%H:%M") if time_to else "*"
        time_desc = f" starting {f_str}–{t_str}"

    LOGGER.info("Reading runners from %s", input_path)
    with workbook_context(input_path) as wb:
        runners = read_runners(input_path, wb)

    LOGGER.info(
        "Loaded %d runners, fetching %s runs%s %s → %s",
        len(runners),
        day_name,
        time_desc,
        start_date.date(),
        end_date.date(),
    )

    all_stats, runner_activities = _fetch_all_runner_stats(
        runners,
        start_date,
        end_date,
        weekday,
        day_name,
        time_from,
        time_to,
    )
    add_rankings(all_stats)

    records = compute_records(runner_activities)
    pairs = compute_running_pairs(runner_activities)

    print_stats(all_stats, day_label=day_name, records=records, pairs=pairs)

    if args.output:
        monthly_rows: list[dict[str, Any]] = []
        yearly_rows: list[dict[str, Any]] = []
        for runner in runners:
            acts = runner_activities.get(runner.name, [])
            monthly_rows.extend(
                compute_monthly_breakdown(
                    runner,
                    acts,
                    start_date,
                    end_date,
                    weekday,
                ),
            )
            yearly_rows.extend(
                compute_yearly_breakdown(
                    runner,
                    acts,
                    start_date,
                    end_date,
                    weekday,
                ),
            )
        try:
            write_excel(
                all_stats,
                monthly_rows,
                yearly_rows,
                records,
                pairs,
                args.output,
            )
        except OSError:
            LOGGER.exception("Failed to write Excel file to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
