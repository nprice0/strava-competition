"""Excel I/O utilities.

All functions are thin wrappers around pandas with validation, type hints and
defensive logic so failures are clearer in production.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pandas as pd

from .models import Runner, Segment, SegmentResult


SEGMENTS_SHEET = "Segments"
RUNNERS_SHEET = "Runners"
MAX_SHEET_NAME_LEN = 31
EXCEL_DATETIME_FORMAT = "yyyy-mm-dd hh:mm:ss"  # Excel-compatible

# Public exports
__all__ = [
    "ExcelFormatError",
    "read_segments",
    "read_runners",
    "write_results",
    "update_runner_refresh_tokens",
]
ResultsMapping = dict[str, dict[str, List[SegmentResult]]]

def _assert_file_exists(path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")


_REQUIRED_SEGMENT_COLS = {"Segment ID", "Segment Name", "Start Date", "End Date"}
_REQUIRED_RUNNER_COLS = {"Name", "Strava ID", "Refresh Token", "Team"}


class ExcelFormatError(RuntimeError):
    """Raised when an expected sheet or required columns are missing."""


def _coerce_path(pathlike) -> str:
    return str(Path(pathlike))


def _validate_columns(df: pd.DataFrame, required: set[str], sheet: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ExcelFormatError(
            f"Missing columns in '{sheet}' sheet: {', '.join(sorted(missing))}. Present: {list(df.columns)}"
        )


def read_segments(filepath) -> List[Segment]:
    """Read segments from the workbook.

    Returns list of Segment objects. Raises ExcelFormatError on schema issues.
    """
    filepath = _coerce_path(filepath)
    _assert_file_exists(filepath)
    try:  # sheet read
        df = pd.read_excel(filepath, sheet_name=SEGMENTS_SHEET)
    except ValueError as e:  # Sheet missing
        raise ExcelFormatError(f"Sheet '{SEGMENTS_SHEET}' not found: {e}") from e
    _validate_columns(df, _REQUIRED_SEGMENT_COLS, SEGMENTS_SHEET)
    # Vectorized datetime conversion then tuple iteration for speed
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df["End Date"] = pd.to_datetime(df["End Date"], errors="coerce")
    segs: List[Segment] = []
    for seg_id, seg_name, start_dt, end_dt in df[
        ["Segment ID", "Segment Name", "Start Date", "End Date"]
    ].itertuples(index=False, name=None):
        segs.append(
            Segment(
                id=int(seg_id),
                name=str(seg_name),
                start_date=start_dt,
                end_date=end_dt,
            )
        )
    return segs


def read_runners(filepath) -> List[Runner]:
    """Read runners (and their refresh tokens) from workbook."""
    filepath = _coerce_path(filepath)
    _assert_file_exists(filepath)
    try:  # sheet read
        df = pd.read_excel(filepath, sheet_name=RUNNERS_SHEET)
    except ValueError as e:
        raise ExcelFormatError(f"Sheet '{RUNNERS_SHEET}' not found: {e}") from e
    _validate_columns(df, _REQUIRED_RUNNER_COLS, RUNNERS_SHEET)
    runners: List[Runner] = []
    for name, strava_id, refresh_token, team in df[
        ["Name", "Strava ID", "Refresh Token", "Team"]
    ].itertuples(index=False, name=None):
        runners.append(
            Runner(
                name=str(name),
                strava_id=int(strava_id),
                refresh_token=str(refresh_token),
                team=str(team),
            )
        )
    return runners


def _rank_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "Fastest Date" in df.columns:
        dt = pd.to_datetime(df["Fastest Date"], utc=True, errors="coerce")
        df["Fastest Date"] = dt.dt.tz_localize(None)
    if "Fastest Time (sec)" in df.columns:
        df["Fastest Time (sec)"] = pd.to_numeric(df["Fastest Time (sec)"], errors="coerce")
        # Drop rows with missing performance metric (cannot rank)
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
    # Sort: team then time for predictable display
    sort_cols = [c for c in ["Team", "Fastest Time (sec)"] if c in df.columns]
    if sort_cols:
        df.sort_values(by=sort_cols, inplace=True)
    preferred_cols = [
        c
        for c in [
            "Team",
            "Runner",
            "Rank",
            "Team Rank",
            "Attempts",
            "Fastest Time (sec)",
            "Fastest Date",
        ]
        if c in df.columns
    ]
    remaining = [c for c in df.columns if c not in preferred_cols]
    return df[preferred_cols + remaining]


def _unique_sheet_name(base: str, used: set[str]) -> str:
    base = base[:MAX_SHEET_NAME_LEN]
    name = base
    i = 1
    while name in used:
        suffix = f"_{i}"
        name = base[: MAX_SHEET_NAME_LEN - len(suffix)] + suffix
        i += 1
    used.add(name)
    return name


def write_results(filepath, results: ResultsMapping) -> None:
    """Write per-segment results to an Excel workbook.

    results structure:
      { segment_name: { team_name: [SegmentResult, ...] } }
    """
    filepath = _coerce_path(filepath)
    with pd.ExcelWriter(
        filepath, engine="openpyxl", datetime_format=EXCEL_DATETIME_FORMAT
    ) as writer:
        if not results:
            pd.DataFrame({"Message": ["No results to display."]}).to_excel(
                writer, sheet_name="Summary", index=False
            )
            return
        used_sheet_names: set[str] = set()
        for segment_name, team_data in results.items():
            rows: list[dict] = []
            for team, seg_results in team_data.items():
                for r in seg_results:
                    rows.append(
                        {
                            "Team": r.team,
                            "Runner": r.runner,
                            "Attempts": r.attempts,
                            "Fastest Time (sec)": r.fastest_time,
                            "Fastest Date": r.fastest_date,
                        }
                    )
            if rows:
                df = _rank_dataframe(pd.DataFrame(rows))
                sheet_name = _unique_sheet_name(segment_name, used_sheet_names)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                msg_df = pd.DataFrame({"Message": ["No results for this segment."]})
                sheet_name = _unique_sheet_name(segment_name + "_msg", used_sheet_names)
                msg_df.to_excel(writer, sheet_name=sheet_name[:MAX_SHEET_NAME_LEN], index=False)
        # Safety: ensure at least one sheet exists
        if not used_sheet_names:
            pd.DataFrame({"Message": ["No data generated."]}).to_excel(
                writer, sheet_name="Summary", index=False
            )


def update_runner_refresh_tokens(filepath, runners: Sequence[Runner]) -> None:
    """Persist updated runner refresh tokens back to the runners sheet.

    Only the refresh token column is modified; order and other columns preserved.
    """
    filepath = _coerce_path(filepath)
    _assert_file_exists(filepath)
    df = pd.read_excel(filepath, sheet_name=RUNNERS_SHEET)
    _validate_columns(df, _REQUIRED_RUNNER_COLS, RUNNERS_SHEET)
    for runner in runners:
        df.loc[df["Strava ID"] == runner.strava_id, "Refresh Token"] = runner.refresh_token
    with pd.ExcelWriter(
        filepath, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        df.to_excel(writer, sheet_name=RUNNERS_SHEET, index=False)
