"""Workbook reading layer (pure reads + validation).

Separated from writing logic for clearer responsibilities and easier testing.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

from .models import Segment, Runner

SEGMENTS_SHEET = "Segment Series"
RUNNERS_SHEET = "Runners"
DISTANCE_SHEET = "Distance Series"

_SEGMENT_ID_COL = "Segment ID"
_SEGMENT_NAME_COL = "Segment Name"
_SEGMENT_START_COL = "Start Date"
_SEGMENT_END_COL = "End Date"
_SEGMENT_DEFAULT_TIME_COL = "Default Time"
_REQUIRED_SEGMENT_COLS = {
    _SEGMENT_ID_COL,
    _SEGMENT_NAME_COL,
    _SEGMENT_START_COL,
    _SEGMENT_END_COL,
    _SEGMENT_DEFAULT_TIME_COL,
}
_REQUIRED_RUNNER_COLS = {
    "Name",
    "Strava ID",
    "Refresh Token",
    "Segment Series Team",
    "Distance Series Team",
}
_REQUIRED_DISTANCE_COLS = {
    _SEGMENT_START_COL,
    _SEGMENT_END_COL,
    "Distance Threshold (km)",
}


class ExcelFormatError(RuntimeError):
    pass


def _is_blank(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _clean_required_name(value: object, row_label: str) -> str:
    if _is_blank(value):
        raise ExcelFormatError(
            f"Runner name missing in {row_label} of '{RUNNERS_SHEET}' sheet"
        )
    return str(value).strip()


def _normalise_strava_id(raw: str) -> str:
    candidate = raw.strip()
    if candidate.endswith(".0"):
        candidate = candidate[:-2]
    return candidate


def _parse_strava_id(value: object, runner_name: str, row_label: str) -> str:
    if _is_blank(value):
        raise ExcelFormatError(
            f"Runner '{runner_name}' in {row_label} missing Strava ID in '{RUNNERS_SHEET}' sheet"
        )
    candidate = _normalise_strava_id(str(value))
    if not candidate.isdigit():
        raise ExcelFormatError(
            f"Runner '{runner_name}' in {row_label} has invalid Strava ID '{value}' (expected digits only)"
        )
    return candidate


def _clean_refresh_token(value: object, runner_name: str, row_label: str) -> str:
    if _is_blank(value):
        raise ExcelFormatError(
            f"Runner '{runner_name}' in {row_label} missing refresh token in '{RUNNERS_SHEET}' sheet"
        )
    return str(value).strip()


def _clean_team(value: object) -> Optional[str]:
    if _is_blank(value):
        return None
    return str(value).strip()


def _seconds_from_timedelta(value: object) -> float | None:
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    if isinstance(value, timedelta):
        return float(value.total_seconds())
    return None


def _seconds_from_datetime(value: object) -> float | None:
    if isinstance(value, (pd.Timestamp, datetime)):
        return float(value.hour * 3600 + value.minute * 60 + value.second)
    return None


def _seconds_from_time(value: object) -> float | None:
    if isinstance(value, time):
        return float(value.hour * 3600 + value.minute * 60 + value.second)
    return None


def _seconds_from_string(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    parsed = pd.to_timedelta(candidate, errors="coerce")
    if not pd.isna(parsed):
        return float(parsed.total_seconds())
    try:
        return float(candidate)
    except ValueError:
        return None


def _seconds_from_number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or pd.isna(value):
        return None
    numeric = float(value)
    if 0 < numeric < 1:
        return numeric * 24 * 3600
    return numeric


def _coerce_duration_seconds(value: object) -> float | None:
    if value is None:
        return None
    for converter in (
        _seconds_from_timedelta,
        _seconds_from_datetime,
        _seconds_from_time,
        _seconds_from_string,
        _seconds_from_number,
    ):
        seconds = converter(value)
        if seconds is not None:
            return seconds
    return None


def _parse_segment_default_time(
    value: object, seg_name: str, row_label: str
) -> float | None:
    seconds = _coerce_duration_seconds(value)
    if seconds is None:
        return None
    if seconds <= 0:
        raise ExcelFormatError(
            f"Segment '{seg_name}' in {row_label} has non-positive default time '{value}'"
        )
    return float(seconds)


def _assert_file_exists(path: str | Path) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Workbook not found: {path}")


def _coerce_path(pathlike: str | Path) -> str:
    return str(Path(pathlike))


class _WorkbookReader:
    """Caches parsed sheets so each Excel workbook is read once."""

    def __init__(self, filepath: str) -> None:
        self._excel = pd.ExcelFile(filepath)
        self._cache: dict[str, pd.DataFrame] = {}

    def parse(self, sheet_name: str) -> pd.DataFrame:
        if sheet_name not in self._cache:
            self._cache[sheet_name] = self._excel.parse(sheet_name=sheet_name)
        return self._cache[sheet_name]

    def close(self) -> None:
        self._excel.close()


def _validate_columns(df: pd.DataFrame, required: set[str], sheet: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ExcelFormatError(
            f"Missing columns in '{sheet}' sheet: {', '.join(sorted(missing))}. Present: {list(df.columns)}"
        )


def _resolve_sheet(
    filepath: str,
    workbook: Optional[object],
    sheet_name: str,
    *,
    optional: bool = False,
) -> Optional[pd.DataFrame]:
    try:
        if isinstance(workbook, _WorkbookReader):
            return workbook.parse(sheet_name)
        if isinstance(workbook, pd.ExcelFile):  # pragma: no cover - legacy path
            return workbook.parse(sheet_name=sheet_name)
        return pd.read_excel(filepath, sheet_name=sheet_name)
    except ValueError as exc:
        if optional:
            return None
        raise ExcelFormatError(f"Sheet '{sheet_name}' not found: {exc}") from exc


def read_segments(
    filepath: str | Path, workbook: Optional[object] = None
) -> List[Segment]:
    filepath = _coerce_path(filepath)
    if workbook is None:
        _assert_file_exists(filepath)
    df = _resolve_sheet(filepath, workbook, SEGMENTS_SHEET)
    if df is None:
        raise ExcelFormatError(
            f"Sheet '{SEGMENTS_SHEET}' is required but was missing or empty"
        )
    _validate_columns(df, _REQUIRED_SEGMENT_COLS, SEGMENTS_SHEET)
    df[_SEGMENT_START_COL] = pd.to_datetime(df[_SEGMENT_START_COL], errors="coerce")
    df[_SEGMENT_END_COL] = pd.to_datetime(df[_SEGMENT_END_COL], errors="coerce")
    segs: List[Segment] = []
    columns = [
        _SEGMENT_ID_COL,
        _SEGMENT_NAME_COL,
        _SEGMENT_START_COL,
        _SEGMENT_END_COL,
        _SEGMENT_DEFAULT_TIME_COL,
    ]
    for row_offset, (
        seg_id,
        seg_name,
        start_dt,
        end_dt,
        default_time_raw,
    ) in enumerate(df[columns].itertuples(index=False, name=None), start=2):
        row_label = f"row {row_offset}"
        segs.append(
            Segment(
                id=int(seg_id),
                name=str(seg_name),
                start_date=start_dt,
                end_date=end_dt,
                default_time_seconds=_parse_segment_default_time(
                    default_time_raw, str(seg_name), row_label
                ),
            )
        )
    return segs


def read_runners(
    filepath: str | Path, workbook: Optional[object] = None
) -> List[Runner]:
    filepath = _coerce_path(filepath)
    if workbook is None:
        _assert_file_exists(filepath)
    df = _resolve_sheet(filepath, workbook, RUNNERS_SHEET)
    if df is None:
        raise ExcelFormatError(
            f"Sheet '{RUNNERS_SHEET}' is required but was missing or empty"
        )
    _validate_columns(df, _REQUIRED_RUNNER_COLS, RUNNERS_SHEET)
    runners: List[Runner] = []
    columns = [
        "Name",
        "Strava ID",
        "Refresh Token",
        "Segment Series Team",
        "Distance Series Team",
    ]
    for row_offset, (name, strava_id, refresh_token, seg_team, dist_team) in enumerate(
        df[columns].itertuples(index=False, name=None), start=2
    ):
        if all(
            _is_blank(value)
            for value in (name, strava_id, refresh_token, seg_team, dist_team)
        ):
            continue

        row_label = f"row {row_offset}"
        clean_name = _clean_required_name(name, row_label)
        strava_id_val = _parse_strava_id(strava_id, clean_name, row_label)
        refresh_val = _clean_refresh_token(refresh_token, clean_name, row_label)

        runners.append(
            Runner(
                name=clean_name,
                strava_id=strava_id_val,
                refresh_token=refresh_val,
                segment_team=_clean_team(seg_team),
                distance_team=_clean_team(dist_team),
            )
        )
    return runners


def read_distance_windows(
    filepath: str | Path,
    workbook: Optional[object] = None,
) -> List[tuple[pd.Timestamp, pd.Timestamp, float | None]]:
    filepath = _coerce_path(filepath)
    if workbook is None:
        _assert_file_exists(filepath)
    df = _resolve_sheet(filepath, workbook, DISTANCE_SHEET, optional=True)
    if df is None:
        return []
    _validate_columns(df, _REQUIRED_DISTANCE_COLS, DISTANCE_SHEET)
    df[_SEGMENT_START_COL] = pd.to_datetime(df[_SEGMENT_START_COL], errors="coerce")
    df[_SEGMENT_END_COL] = pd.to_datetime(df[_SEGMENT_END_COL], errors="coerce")
    windows: List[tuple[pd.Timestamp, pd.Timestamp, float | None]] = []
    for start_dt, end_dt, threshold in df[
        [_SEGMENT_START_COL, _SEGMENT_END_COL, "Distance Threshold (km)"]
    ].itertuples(index=False, name=None):
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
    "workbook_context",
]


@contextmanager
def workbook_context(filepath: str | Path) -> Iterator[_WorkbookReader]:
    """Yield a caching workbook reader so callers reuse a single file handle."""

    file_path = _coerce_path(filepath)
    _assert_file_exists(file_path)
    reader = _WorkbookReader(file_path)
    try:
        yield reader
    finally:
        reader.close()
