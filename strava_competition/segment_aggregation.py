"""Segment aggregation helpers.

Pure functions that transform in-memory segment competition results into
DataFrames ready to be written to Excel. Separated from the writer to mirror
the distance aggregation module and keep transformation logic testable.
"""
from __future__ import annotations

from typing import List, Tuple, Dict
import pandas as pd

from .models import SegmentResult

ResultsMapping = dict[str, dict[str, List[SegmentResult]]]

__all__ = [
    "build_segment_outputs",
]


def _rank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "Fastest Date" in df.columns:
        dt = pd.to_datetime(df["Fastest Date"], utc=True, errors="coerce")
        df["Fastest Date"] = dt.dt.tz_localize(None)
    if "Fastest Time (sec)" in df.columns:
        df["Fastest Time (sec)"] = pd.to_numeric(df["Fastest Time (sec)"], errors="coerce")
        df = df.dropna(subset=["Fastest Time (sec)"])
        if not df.empty:
            df["Rank"] = (
                df["Fastest Time (sec)"].rank(method="min", ascending=True).astype(int)
            )
            if "Team" in df.columns:
                df["Team Rank"] = (
                    df.groupby("Team")["Fastest Time (sec)"]
                    .rank(method="min", ascending=True)
                    .astype(int)
                )
    sort_cols = [c for c in ["Team", "Fastest Time (sec)"] if c in df.columns]
    if sort_cols:
        df.sort_values(by=sort_cols, inplace=True)
    from .config import SEGMENT_ENFORCE_COLUMN_ORDER, SEGMENT_COLUMN_ORDER
    preferred_source = SEGMENT_COLUMN_ORDER if SEGMENT_ENFORCE_COLUMN_ORDER else [
        "Team","Runner","Rank","Team Rank","Attempts","Fastest Time (sec)","Fastest Date"
    ]
    preferred_cols = [c for c in preferred_source if c in df.columns]
    remaining = [c for c in df.columns if c not in preferred_cols]
    return df[preferred_cols + remaining]


def _rows_for_segment(segment_team_mapping: dict[str, List[SegmentResult]]) -> List[dict]:
    rows: List[dict] = []
    for _team, seg_results in segment_team_mapping.items():
        for r in seg_results:
            rows.append({
                "Team": r.team,
                "Runner": r.runner,
                "Attempts": r.attempts,
                "Fastest Time (sec)": r.fastest_time,
                "Fastest Date": r.fastest_date,
            })
    return rows


def _build_summary_df(results: ResultsMapping) -> pd.DataFrame:
    from collections import defaultdict
    team_runners: dict[str, set[str]] = defaultdict(set)
    team_segments: dict[str, set[str]] = defaultdict(set)
    team_attempts: dict[str, int] = defaultdict(int)
    team_fastest_times: dict[str, list[float]] = defaultdict(list)
    for segment_name, team_data in results.items():
        for team, seg_results in team_data.items():
            if not seg_results:
                continue
            team_segments[team].add(segment_name)
            for r in seg_results:
                team_runners[team].add(r.runner)
                team_attempts[team] += r.attempts
                if r.fastest_time is not None:
                    team_fastest_times[team].append(r.fastest_time)
    summary_rows: list[dict] = []
    for team, times in team_fastest_times.items():
        if not times:
            continue
        total_time = sum(times)
        avg_time = total_time / len(times)
        summary_rows.append({
            "Team": team,
            "Runners Participating": len(team_runners[team]),
            "Segments With Participation": len(team_segments[team]),
            "Total Attempts": team_attempts[team],
            "Average Fastest Time (sec)": round(avg_time, 2),
            "Total Fastest Times (sec)": round(total_time, 2),
        })
    if not summary_rows:
        return pd.DataFrame()
    df = pd.DataFrame(summary_rows)
    # Prefer sorting by total if available, else by average
    if "Total Fastest Times (sec)" in df.columns:
        df.sort_values(by=["Total Fastest Times (sec)", "Team"], inplace=True)
    else:
        df.sort_values(by=["Average Fastest Time (sec)", "Team"], inplace=True)
    return df


def build_segment_outputs(
    results: ResultsMapping, include_summary: bool = True
) -> List[Tuple[str, pd.DataFrame]]:
    """Return list of (sheet_base_name, dataframe) for each segment + optional summary.

    Sheet base names are the raw segment names and "Summary" (if included).
    The caller (writer) is responsible for resolving Excel sheet name conflicts.
    """
    outputs: List[Tuple[str, pd.DataFrame]] = []
    for segment_name, team_data in results.items():
        rows = _rows_for_segment(team_data)
        if not rows:
            # empty segment -> message sheet handled by writer (kept consistent with previous behaviour)
            df = pd.DataFrame({"Message": ["No results for this segment."]})
            outputs.append((segment_name + "_msg", df))
            continue
        df = _rank_dataframe(pd.DataFrame(rows))
        outputs.append((segment_name, df))
    if include_summary and results:
        summary_df = _build_summary_df(results)
        if not summary_df.empty:
            outputs.append(("Summary", summary_df))
    return outputs
