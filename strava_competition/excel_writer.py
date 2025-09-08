"""Excel writer for segment and distance outputs (reads live in excel_reader)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Iterable, Tuple
import pandas as pd

from .models import Runner, SegmentResult
from .errors import ExcelFormatError
from .segment_aggregation import build_segment_outputs, ResultsMapping as SegResultsMapping

SEGMENTS_SHEET = "Segment Series"
RUNNERS_SHEET = "Runners"
DISTANCE_SHEET = "Distance Series"
MAX_SHEET_NAME_LEN = 31
EXCEL_DATETIME_FORMAT = "yyyy-mm-dd hh:mm:ss"

__all__ = [
    "ExcelFormatError",
    "write_results",
    "update_runner_refresh_tokens",
    "update_single_runner_refresh_token",
]

ResultsMapping = SegResultsMapping

def _assert_file_exists(path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")

_REQUIRED_RUNNER_COLS = {"Name", "Strava ID", "Refresh Token", "Segment Series Team", "Distance Series Team"}

def _coerce_path(pathlike) -> str:
    return str(Path(pathlike))

## Segment aggregation lives in segment_aggregation.build_segment_outputs

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
        if ws.max_row > EXCEL_AUTOSIZE_MAX_ROWS:
            return
        for col_cells in ws.columns:
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
        pass

## Row construction handled in segment_aggregation

def _write_segment_sheets(writer, results: ResultsMapping, used_sheet_names: set[str], include_summary: bool) -> None:
    outputs = build_segment_outputs(results, include_summary=include_summary)
    for base_name, df in outputs:
        sheet_name = _unique_sheet_name(base_name, used_sheet_names)
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        _autosize(writer.sheets[sheet_name])

## Summary building handled via segment_aggregation

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
        logging.info("%s: %s rows=%s (empty=%s)", log_prefix, sheet_name, len(rows), not bool(rows))
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
    filepath = _coerce_path(filepath)
    with pd.ExcelWriter(filepath, engine="openpyxl", datetime_format=EXCEL_DATETIME_FORMAT) as writer:
        used_sheet_names: set[str] = set()
        if results:
            _write_segment_sheets(writer, results, used_sheet_names, include_summary)
        else:
            pd.DataFrame({"Message": ["No results to display."]}).to_excel(writer, sheet_name="Summary", index=False)
            _autosize(writer.sheets["Summary"])
        if distance_windows_results:
            _write_distance_sheets(writer, distance_windows_results, used_sheet_names)
            for sn in used_sheet_names:
                try:
                    ws = writer.book[sn]
                except Exception:
                    ws = writer.sheets.get(sn)
                if ws is not None:
                    _autosize(ws)

def update_runner_refresh_tokens(filepath, runners: Sequence[Runner]) -> None:
    filepath = _coerce_path(filepath)
    _assert_file_exists(filepath)
    df = pd.read_excel(filepath, sheet_name=RUNNERS_SHEET)
    missing = _REQUIRED_RUNNER_COLS - set(df.columns)
    if missing:
        raise ExcelFormatError(f"Missing columns in '{RUNNERS_SHEET}' sheet: {', '.join(sorted(missing))}")
    for runner in runners:
        df.loc[df["Strava ID"] == runner.strava_id, "Refresh Token"] = runner.refresh_token
    with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=RUNNERS_SHEET, index=False)


def update_single_runner_refresh_token(filepath, runner: Runner) -> None:
    """Persist refresh token for a single runner (crash-safe incremental update).

    Reads only the Runners sheet, updates the row for the given runner, rewrites
    that sheet. Lightweight enough for occasional rotations.
    """
    filepath = _coerce_path(filepath)
    _assert_file_exists(filepath)
    try:
        df = pd.read_excel(filepath, sheet_name=RUNNERS_SHEET)
    except Exception:
        return
    if "Strava ID" not in df.columns or "Refresh Token" not in df.columns:
        return
    df.loc[df["Strava ID"] == runner.strava_id, "Refresh Token"] = runner.refresh_token
    try:
        with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=RUNNERS_SHEET, index=False)
    except Exception:
        # Silent failure acceptable; final write at shutdown still attempts full persistence.
        return
