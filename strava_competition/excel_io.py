"""Excel I/O utilities.

All functions are thin wrappers around pandas with validation, type hints and
defensive logic so failures are clearer in production.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Iterable, Tuple

import pandas as pd

from .models import Runner, SegmentResult
from .excel_reader import (
    ExcelFormatError,
    read_segments as _read_segments,
    read_runners as _read_runners,
    read_distance_windows as _read_distance_windows,
)


SEGMENTS_SHEET = "Segment Series"  # re-export for compatibility
RUNNERS_SHEET = "Runners"
DISTANCE_SHEET = "Distance Series"
MAX_SHEET_NAME_LEN = 31
EXCEL_DATETIME_FORMAT = "yyyy-mm-dd hh:mm:ss"  # Excel-compatible

# Public exports
__all__ = [
    "ExcelFormatError",
    "read_segments",
    "read_runners",
    "read_distance_windows",
    "write_results",
    "update_runner_refresh_tokens",
]
ResultsMapping = dict[str, dict[str, List[SegmentResult]]]

def _assert_file_exists(path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")


_REQUIRED_RUNNER_COLS = {"Name", "Strava ID", "Refresh Token", "Segment Series Team", "Distance Series Team"}


def _coerce_path(pathlike) -> str:  # retained for writer path normalization
    return str(Path(pathlike))


def _validate_columns(df: pd.DataFrame, required: set[str], sheet: str) -> None:  # pragma: no cover (legacy path)
    missing = required - set(df.columns)
    if missing:
        raise ExcelFormatError(
            f"Missing columns in '{sheet}' sheet: {', '.join(sorted(missing))}. Present: {list(df.columns)}"
        )


def read_segments(path):  # shim
    return _read_segments(path)

def read_runners(path):  # shim
    return _read_runners(path)

def read_distance_windows(path):  # shim
    return _read_distance_windows(path)


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
    # Apply configurable segment column ordering if enabled
    from .config import SEGMENT_ENFORCE_COLUMN_ORDER, SEGMENT_COLUMN_ORDER
    if SEGMENT_ENFORCE_COLUMN_ORDER:
        preferred_source = SEGMENT_COLUMN_ORDER
    else:
        preferred_source = [
            "Team",
            "Runner",
            "Rank",
            "Team Rank",
            "Attempts",
            "Fastest Time (sec)",
            "Fastest Date",
        ]
    preferred_cols = [c for c in preferred_source if c in df.columns]
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


def _autosize(ws) -> None:
    """Auto-size worksheet columns with guards for performance and width bounds."""
    from .config import (
        EXCEL_AUTOSIZE_COLUMNS,
        EXCEL_AUTOSIZE_MAX_WIDTH,
        EXCEL_AUTOSIZE_MIN_WIDTH,
        EXCEL_AUTOSIZE_PADDING,
        EXCEL_AUTOSIZE_MAX_ROWS,
    )
    if not EXCEL_AUTOSIZE_COLUMNS:
        return
    try:
        dim = ws.dimensions  # force openpyxl to know bounds
    except Exception:
        dim = None
    try:
        if ws.max_row > EXCEL_AUTOSIZE_MAX_ROWS:
            return  # skip large sheets to avoid O(n^2) walk
        for col_cells in ws.columns:
            # Collect string lengths of values (header + data)
            max_len = 0
            col_letter = getattr(col_cells[0], "column_letter", None)
            for cell in col_cells:
                val = cell.value
                if val is None:
                    continue
                l = len(str(val))
                if l > max_len:
                    max_len = l
            width = min(EXCEL_AUTOSIZE_MAX_WIDTH, max(EXCEL_AUTOSIZE_MIN_WIDTH, max_len + EXCEL_AUTOSIZE_PADDING))
            if col_letter:
                ws.column_dimensions[col_letter].width = width
    except Exception:
        # Autosize is best-effort; never raise
        pass


def _rows_for_segment(segment_team_mapping: dict[str, List[SegmentResult]]) -> List[dict]:
    rows: List[dict] = []
    for team, seg_results in segment_team_mapping.items():
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
    return rows


def _write_segment_sheets(writer, results: ResultsMapping, used_sheet_names: set[str]) -> None:
    for segment_name, team_data in results.items():
        rows = _rows_for_segment(team_data)
        if rows:
            df = _rank_dataframe(pd.DataFrame(rows))
            sheet_name = _unique_sheet_name(segment_name, used_sheet_names)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            _autosize(writer.sheets[sheet_name])
        else:
            msg_df = pd.DataFrame({"Message": ["No results for this segment."]})
            sheet_name = _unique_sheet_name(segment_name + "_msg", used_sheet_names)
            msg_df.to_excel(writer, sheet_name=sheet_name[:MAX_SHEET_NAME_LEN], index=False)
            _autosize(writer.sheets[sheet_name[:MAX_SHEET_NAME_LEN]])


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
        summary_rows.append(
            {
                "Team": team,
                "Runners Participating": len(team_runners[team]),
                "Segments With Participation": len(team_segments[team]),
                "Total Attempts": team_attempts[team],
                "Average Fastest Time (sec)": round(avg_time, 2),
                "Total Fastest Times (sec)": round(total_time, 2),
            }
        )
    if not summary_rows:
        return pd.DataFrame()
    df = pd.DataFrame(summary_rows)
    if "Sum Fastest Times (sec)" in df.columns:
        df.sort_values(by=["Sum Fastest Times (sec)", "Team"], inplace=True)
    else:
        df.sort_values(by=["Average Fastest Time (sec)", "Team"], inplace=True)
    return df


def _write_summary_sheet(writer, summary_df: pd.DataFrame, used_sheet_names: set[str]) -> None:
    if summary_df.empty:
        return
    sheet_name = _unique_sheet_name("Summary", used_sheet_names)
    summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
    _autosize(writer.sheets[sheet_name])


def _write_distance_sheets(
    writer,
    distance_windows_results: Iterable[Tuple[str, List[dict]]],
    used_sheet_names: set[str],
    log_prefix: str = "Wrote distance window sheet",
) -> None:
    import logging
    from .config import (
        DISTANCE_CREATE_EMPTY_WINDOW_SHEETS,
        DISTANCE_ENFORCE_COLUMN_ORDER,
        DISTANCE_COLUMN_ORDER,
    )

    for sheet_base, rows in distance_windows_results:
        if not rows and not DISTANCE_CREATE_EMPTY_WINDOW_SHEETS:
            continue
        dfw = pd.DataFrame(rows)
        if DISTANCE_ENFORCE_COLUMN_ORDER and not dfw.empty:
            ordered = [c for c in DISTANCE_COLUMN_ORDER if c in dfw.columns]
            remaining = [c for c in dfw.columns if c not in ordered]
            if ordered:
                dfw = dfw[ordered + remaining]
        sheet_name = _unique_sheet_name(sheet_base, used_sheet_names)
        dfw.to_excel(writer, sheet_name=sheet_name, index=False)
        logging.info(
            "%s: %s rows=%s (empty=%s)",
            log_prefix,
            sheet_name,
            len(rows),
            not bool(rows),
        )
        # Use workbook lookup to avoid any timing issues with pandas' internal sheet map
        try:
            ws = writer.book[sheet_name]
        except Exception:
            ws = writer.sheets.get(sheet_name)
        if ws is not None:
            _autosize(ws)


def write_results(
    filepath,
    results: ResultsMapping,
    include_summary: bool = True,
    distance_windows_results: list[tuple[str, list[dict]]] | None = None,
) -> None:
    """Write competition results (segment + optional distance windows) to Excel.

    results structure: { segment_name: { team_name: [SegmentResult, ...] } }
    """
    filepath = _coerce_path(filepath)
    import logging

    with pd.ExcelWriter(filepath, engine="openpyxl", datetime_format=EXCEL_DATETIME_FORMAT) as writer:
        used_sheet_names: set[str] = set()
        if results:
            _write_segment_sheets(writer, results, used_sheet_names)
            if include_summary:
                summary_df = _build_summary_df(results)
                _write_summary_sheet(writer, summary_df, used_sheet_names)
        else:
            # No segment data – write a placeholder summary sheet
            pd.DataFrame({"Message": ["No results to display."]}).to_excel(
                writer, sheet_name="Summary", index=False
            )
            _autosize(writer.sheets["Summary"])  # best-effort
        # Ensure at least one sheet exists (defensive – though guaranteed above)
        if not used_sheet_names and "Summary" not in used_sheet_names:
            used_sheet_names.add("Summary")

        # Distance sheets appended last
        if distance_windows_results:
            _write_distance_sheets(
                writer, distance_windows_results, used_sheet_names
            )
            # Final defensive autosize pass (in case any sheet missed earlier)
            for sn in used_sheet_names:
                try:
                    ws = writer.book[sn]
                except Exception:
                    ws = writer.sheets.get(sn)
                if ws is not None:
                    _autosize(ws)


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
