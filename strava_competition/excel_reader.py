"""Workbook reading layer (pure reads + validation).

Separated from writing logic for clearer responsibilities and easier testing.
"""
from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd

from .models import Segment, Runner

SEGMENTS_SHEET = "Segment Series"
RUNNERS_SHEET = "Runners"
DISTANCE_SHEET = "Distance Series"

_REQUIRED_SEGMENT_COLS = {"Segment ID", "Segment Name", "Start Date", "End Date"}
_REQUIRED_RUNNER_COLS = {"Name", "Strava ID", "Refresh Token", "Segment Series Team", "Distance Series Team"}
_REQUIRED_DISTANCE_COLS = {"Start Date", "End Date", "Distance Threshold (km)"}

class ExcelFormatError(RuntimeError):
    pass

def _assert_file_exists(path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")

def _coerce_path(pathlike) -> str:
    return str(Path(pathlike))

def _validate_columns(df: pd.DataFrame, required: set[str], sheet: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ExcelFormatError(
            f"Missing columns in '{sheet}' sheet: {', '.join(sorted(missing))}. Present: {list(df.columns)}"
        )

def read_segments(filepath) -> List[Segment]:
    filepath = _coerce_path(filepath)
    _assert_file_exists(filepath)
    try:
        df = pd.read_excel(filepath, sheet_name=SEGMENTS_SHEET)
    except ValueError as e:
        raise ExcelFormatError(f"Sheet '{SEGMENTS_SHEET}' not found: {e}") from e
    _validate_columns(df, _REQUIRED_SEGMENT_COLS, SEGMENTS_SHEET)
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df["End Date"] = pd.to_datetime(df["End Date"], errors="coerce")
    segs: List[Segment] = []
    for seg_id, seg_name, start_dt, end_dt in df[["Segment ID", "Segment Name", "Start Date", "End Date"]].itertuples(index=False, name=None):
        segs.append(Segment(id=int(seg_id), name=str(seg_name), start_date=start_dt, end_date=end_dt))
    return segs

def read_runners(filepath) -> List[Runner]:
    filepath = _coerce_path(filepath)
    _assert_file_exists(filepath)
    try:
        df = pd.read_excel(filepath, sheet_name=RUNNERS_SHEET)
    except ValueError as e:
        raise ExcelFormatError(f"Sheet '{RUNNERS_SHEET}' not found: {e}") from e
    _validate_columns(df, _REQUIRED_RUNNER_COLS, RUNNERS_SHEET)
    runners: List[Runner] = []
    for name, strava_id, refresh_token, seg_team, dist_team in df[["Name", "Strava ID", "Refresh Token", "Segment Series Team", "Distance Series Team"]].itertuples(index=False, name=None):
        runners.append(
            Runner(
                name=str(name),
                strava_id=int(strava_id),
                refresh_token=str(refresh_token),
                segment_team=(str(seg_team).strip() or None) if pd.notna(seg_team) else None,
                distance_team=(str(dist_team).strip() or None) if pd.notna(dist_team) else None,
            )
        )
    return runners

def read_distance_windows(filepath) -> List[tuple[pd.Timestamp, pd.Timestamp, float | None]]:
    filepath = _coerce_path(filepath)
    _assert_file_exists(filepath)
    try:
        df = pd.read_excel(filepath, sheet_name=DISTANCE_SHEET)
    except ValueError:
        return []
    _validate_columns(df, _REQUIRED_DISTANCE_COLS, DISTANCE_SHEET)
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df["End Date"] = pd.to_datetime(df["End Date"], errors="coerce")
    windows: List[tuple[pd.Timestamp, pd.Timestamp, float | None]] = []
    for start_dt, end_dt, threshold in df[["Start Date", "End Date", "Distance Threshold (km)"]].itertuples(index=False, name=None):
        if pd.isna(start_dt) or pd.isna(end_dt) or start_dt > end_dt:
            continue
        thr_val: float | None = None
        if not pd.isna(threshold):
            try:
                thr_val = float(threshold)
            except Exception:
                thr_val = None
        windows.append((start_dt, end_dt, thr_val))
    return windows

__all__ = [
    "ExcelFormatError",
    "read_segments",
    "read_runners",
    "read_distance_windows",
]