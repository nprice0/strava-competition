"""Segment aggregation helpers.

Pure functions that transform in-memory segment competition results into
DataFrames ready to be written to Excel. Separated from the writer to mirror
the distance aggregation module and keep transformation logic testable.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import math
import statistics
import pandas as pd

from .models import SegmentResult

ResultsMapping = dict[str, dict[str, List[SegmentResult]]]

TEAM_COL = "Team"
RUNNER_COL = "Runner"
ATTEMPTS_COL = "Attempts"
FASTEST_SEC_COL = "Fastest Time (sec)"
FASTEST_FMT_COL = "Fastest Time (h:mm:ss)"
FASTEST_DATE_COL = "Fastest Date"
RANK_COL = "Rank"
TEAM_RANK_COL = "Team Rank"

SUMMARY_TEAM_COL = "Team"
SUMMARY_TOTAL_FASTEST_COL = "Total Fastest time"
SUMMARY_GAP_COL = "Gap to Leader"
SUMMARY_RANK_COL = "Total Rank"
SUMMARY_FASTEST_RUNNER_COL = "Fastest (Runner)"
SUMMARY_AVG_FASTEST_COL = "Avg. Fastest Time"
SUMMARY_MEDIAN_FASTEST_COL = "Median Fastest Time"
SUMMARY_TOTAL_ATTEMPTS_COL = "Total Attempts"
SUMMARY_AVG_ATTEMPTS_COL = "Avg. Attempts"

SUMMARY_COLUMNS = [
    SUMMARY_TEAM_COL,
    SUMMARY_TOTAL_FASTEST_COL,
    SUMMARY_GAP_COL,
    SUMMARY_RANK_COL,
    SUMMARY_FASTEST_RUNNER_COL,
    SUMMARY_AVG_FASTEST_COL,
    SUMMARY_MEDIAN_FASTEST_COL,
    SUMMARY_TOTAL_ATTEMPTS_COL,
    SUMMARY_AVG_ATTEMPTS_COL,
]

SUMMARY_RACER_COUNT_COL = "Runners Participating"
SUMMARY_SEGMENT_COUNT_COL = "Segments With Participation"
SUMMARY_TOTAL_FASTEST_SEC_COL = "Total Fastest Times (sec)"
SUMMARY_AVG_FASTEST_SEC_COL = "Average Fastest Time (sec)"

__all__ = [
    "build_segment_outputs",
]


def _rank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if FASTEST_DATE_COL in df.columns:
        dt = pd.to_datetime(df[FASTEST_DATE_COL], utc=True, errors="coerce")
        df[FASTEST_DATE_COL] = dt.dt.tz_localize(None)
    if FASTEST_SEC_COL in df.columns:
        df[FASTEST_SEC_COL] = pd.to_numeric(df[FASTEST_SEC_COL], errors="coerce")
        df = df.dropna(subset=[FASTEST_SEC_COL])
        if not df.empty:
            df[RANK_COL] = (
                df[FASTEST_SEC_COL].rank(method="min", ascending=True).astype(int)
            )
            if TEAM_COL in df.columns:
                df[TEAM_RANK_COL] = (
                    df.groupby(TEAM_COL)[FASTEST_SEC_COL]
                    .rank(method="min", ascending=True)
                    .astype(int)
                )
    sort_cols = [c for c in [TEAM_COL, FASTEST_SEC_COL] if c in df.columns]
    if sort_cols:
        df.sort_values(by=sort_cols, inplace=True)
    from .config import SEGMENT_ENFORCE_COLUMN_ORDER, SEGMENT_COLUMN_ORDER

    preferred_source = (
        SEGMENT_COLUMN_ORDER
        if SEGMENT_ENFORCE_COLUMN_ORDER
        else [
            TEAM_COL,
            RUNNER_COL,
            RANK_COL,
            TEAM_RANK_COL,
            ATTEMPTS_COL,
            FASTEST_SEC_COL,
            FASTEST_FMT_COL,
            FASTEST_DATE_COL,
        ]
    )
    preferred_cols = [c for c in preferred_source if c in df.columns]
    remaining = [c for c in df.columns if c not in preferred_cols]
    ordered = df[preferred_cols + remaining]
    return _insert_team_spacers(ordered)


def _rows_for_segment(
    segment_team_mapping: dict[str, List[SegmentResult]],
) -> List[dict]:
    rows: List[dict] = []
    for _team, seg_results in segment_team_mapping.items():
        for r in seg_results:
            rows.append(
                {
                    TEAM_COL: r.team,
                    RUNNER_COL: r.runner,
                    ATTEMPTS_COL: r.attempts,
                    FASTEST_SEC_COL: r.fastest_time,
                    FASTEST_FMT_COL: _format_fastest_time(r.fastest_time),
                    FASTEST_DATE_COL: r.fastest_date,
                }
            )
    return rows


def _format_fastest_time(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    sign = "-" if value < 0 else ""
    total_seconds = int(round(abs(value)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{sign}{hours}:{minutes:02d}:{secs:02d}"


def _insert_team_spacers(df: pd.DataFrame) -> pd.DataFrame:
    if TEAM_COL not in df.columns or df.empty:
        return df
    buffer: list[dict] = []
    df_reset = df.reset_index(drop=True)
    columns = list(df_reset.columns)

    def _blank_row() -> dict:
        return dict.fromkeys(columns)

    for idx in range(len(df_reset)):
        row = df_reset.iloc[idx].to_dict()
        buffer.append(row)
        next_is_new_team = True
        if idx < len(df_reset) - 1:
            next_is_new_team = (
                df_reset.at[idx, TEAM_COL] != df_reset.at[idx + 1, TEAM_COL]
            )
        if next_is_new_team and idx != len(df_reset) - 1:
            buffer.append(_blank_row())
    return pd.DataFrame(buffer, columns=columns)


def _compute_team_rank_totals(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or RANK_COL not in df.columns or TEAM_COL not in df.columns:
        return {}
    filtered = df[df[RUNNER_COL].notna() & df[RANK_COL].notna()]
    if filtered.empty:
        return {}
    grouped = filtered.groupby(TEAM_COL)[RANK_COL].sum()
    return {team: int(total) for team, total in grouped.items()}


def _build_segment_team_summary(
    segment_team_mapping: dict[str, List[SegmentResult]],
    team_rank_totals: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict] = []
    for team, seg_results in segment_team_mapping.items():
        summary_row = _summarise_team_segment(team, seg_results, team_rank_totals)
        if summary_row is not None:
            rows.append(summary_row)

    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows.sort(key=lambda item: (item["_rank_sum"], item["_total_seconds"]))
    leader_total = min(row["_total_seconds"] for row in rows)
    for rank, row in enumerate(rows, start=1):
        row[SUMMARY_RANK_COL] = (
            None if row["_rank_sum"] in (None, float("inf")) else int(row["_rank_sum"])
        )
        row[SUMMARY_GAP_COL] = _format_fastest_time(
            row["_total_seconds"] - leader_total
        )
        row[SUMMARY_TOTAL_FASTEST_COL] = _format_fastest_time(row["_total_seconds"])
        row[SUMMARY_AVG_FASTEST_COL] = _format_fastest_time(row["_avg_seconds"])
        row[SUMMARY_MEDIAN_FASTEST_COL] = _format_fastest_time(row["_median_seconds"])
        row[SUMMARY_AVG_ATTEMPTS_COL] = round(row[SUMMARY_AVG_ATTEMPTS_COL], 2)

    display_rows = [
        {
            SUMMARY_TEAM_COL: row[SUMMARY_TEAM_COL],
            SUMMARY_TOTAL_FASTEST_COL: row[SUMMARY_TOTAL_FASTEST_COL],
            SUMMARY_GAP_COL: row[SUMMARY_GAP_COL],
            SUMMARY_RANK_COL: row[SUMMARY_RANK_COL],
            SUMMARY_FASTEST_RUNNER_COL: row[SUMMARY_FASTEST_RUNNER_COL],
            SUMMARY_AVG_FASTEST_COL: row[SUMMARY_AVG_FASTEST_COL],
            SUMMARY_MEDIAN_FASTEST_COL: row[SUMMARY_MEDIAN_FASTEST_COL],
            SUMMARY_TOTAL_ATTEMPTS_COL: row[SUMMARY_TOTAL_ATTEMPTS_COL],
            SUMMARY_AVG_ATTEMPTS_COL: row[SUMMARY_AVG_ATTEMPTS_COL],
        }
        for row in rows
    ]
    return pd.DataFrame(display_rows, columns=SUMMARY_COLUMNS)


def _summarise_team_segment(
    team: str, seg_results: List[SegmentResult], team_rank_totals: dict[str, int]
) -> dict | None:
    if not seg_results:
        return None
    valid_results = [r for r in seg_results if r.fastest_time is not None]
    if not valid_results:
        return None
    times = [float(r.fastest_time) for r in valid_results]
    total_time = sum(times)
    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    total_attempts = sum((r.attempts or 0) for r in seg_results)
    avg_attempts = total_attempts / len(seg_results) if seg_results else 0.0
    fastest_runner = min(valid_results, key=lambda r: r.fastest_time or float("inf"))
    total_rank = team_rank_totals.get(team)
    rank_sum = total_rank if total_rank is not None else float("inf")
    return {
        SUMMARY_TEAM_COL: team,
        "_total_seconds": total_time,
        "_avg_seconds": avg_time,
        "_median_seconds": median_time,
        "_rank_sum": rank_sum,
        SUMMARY_FASTEST_RUNNER_COL: (
            f"{fastest_runner.runner} ({_format_fastest_time(fastest_runner.fastest_time)})"
            if fastest_runner.fastest_time is not None
            else fastest_runner.runner
        ),
        SUMMARY_TOTAL_ATTEMPTS_COL: total_attempts,
        SUMMARY_AVG_ATTEMPTS_COL: avg_attempts,
    }


def _build_summary_df(results: ResultsMapping) -> pd.DataFrame:
    summary_rows = _calculate_summary_rows(results)
    if not summary_rows:
        return pd.DataFrame()
    df = pd.DataFrame(summary_rows)
    return _sort_summary_dataframe(df)


def _calculate_summary_rows(results: ResultsMapping) -> list[dict]:
    team_stats = _aggregate_team_stats(results)
    summary_rows: list[dict] = []
    for team, stats in team_stats.items():
        if not stats["times"]:
            continue
        total_time = sum(stats["times"])
        avg_time = total_time / len(stats["times"])
        summary_rows.append(
            {
                TEAM_COL: team,
                SUMMARY_RACER_COUNT_COL: len(stats["runners"]),
                SUMMARY_SEGMENT_COUNT_COL: len(stats["segments"]),
                SUMMARY_TOTAL_ATTEMPTS_COL: stats["attempts"],
                SUMMARY_AVG_FASTEST_SEC_COL: round(avg_time, 2),
                SUMMARY_TOTAL_FASTEST_SEC_COL: round(total_time, 2),
            }
        )
    return summary_rows


def _aggregate_team_stats(results: ResultsMapping) -> dict[str, dict]:
    from collections import defaultdict

    def _default_stats() -> dict:
        return {
            "runners": set(),
            "segments": set(),
            "attempts": 0,
            "times": [],
        }

    stats: dict[str, dict] = defaultdict(_default_stats)
    for segment_name, team_data in results.items():
        for team, seg_results in team_data.items():
            if not seg_results:
                continue
            team_entry = stats[team]
            team_entry["segments"].add(segment_name)
            _populate_team_entry(team_entry, seg_results)
    return stats


def _populate_team_entry(entry: dict, seg_results: List[SegmentResult]) -> None:
    for result in seg_results:
        entry["runners"].add(result.runner)
        entry["attempts"] += result.attempts
        if result.fastest_time is not None:
            entry["times"].append(result.fastest_time)


def _sort_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer sorting by total if available, else by average
    if SUMMARY_TOTAL_FASTEST_SEC_COL in df.columns:
        df.sort_values(by=[SUMMARY_TOTAL_FASTEST_SEC_COL, TEAM_COL], inplace=True)
    elif SUMMARY_AVG_FASTEST_SEC_COL in df.columns:
        df.sort_values(by=[SUMMARY_AVG_FASTEST_SEC_COL, TEAM_COL], inplace=True)
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
        team_rank_totals = _compute_team_rank_totals(df)
        summary_df = _build_segment_team_summary(team_data, team_rank_totals)
        if not summary_df.empty:
            df.attrs["segment_summary"] = summary_df
        outputs.append((segment_name, df))
    if include_summary and results:
        summary_df = _build_summary_df(results)
        if not summary_df.empty:
            outputs.append(("Summary", summary_df))
    return outputs
