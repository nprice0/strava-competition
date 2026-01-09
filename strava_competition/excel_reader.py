"""Workbook reading layer (pure reads + validation).

Separated from writing logic for clearer responsibilities and easier testing.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, time, timedelta
import logging
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

from .errors import ExcelFormatError
from .models import Segment, Runner, SegmentWindow, SegmentGroup

LOGGER = logging.getLogger(__name__)

SEGMENTS_SHEET = "Segment Series"
RUNNERS_SHEET = "Runners"
DISTANCE_SHEET = "Distance Series"

_SEGMENT_ID_COL = "Segment ID"
_SEGMENT_NAME_COL = "Segment Name"
_SEGMENT_START_COL = "Start Date"
_SEGMENT_END_COL = "End Date"
_SEGMENT_WINDOW_LABEL_COL = "Window Label"
_SEGMENT_DEFAULT_TIME_COL = "Default Time"
_SEGMENT_MIN_DISTANCE_COL = "Minimum Distance (m)"
_SEGMENT_BIRTHDAY_BONUS_COL = "Birthday Bonus (secs)"
_SEGMENT_TIME_BONUS_COL = "Time Bonus (secs)"
_REQUIRED_SEGMENT_COLS = {
    _SEGMENT_ID_COL,
    _SEGMENT_NAME_COL,
    _SEGMENT_START_COL,
    _SEGMENT_END_COL,
    _SEGMENT_DEFAULT_TIME_COL,
    _SEGMENT_MIN_DISTANCE_COL,
    _SEGMENT_BIRTHDAY_BONUS_COL,
}
# Window Label and Time Bonus are optional
_OPTIONAL_SEGMENT_COLS = {_SEGMENT_WINDOW_LABEL_COL, _SEGMENT_TIME_BONUS_COL}
_REQUIRED_RUNNER_COLS = {
    "Name",
    "Strava ID",
    "Refresh Token",
    "Segment Series Team",
    "Distance Series Team",
    "Birthday (dd-mmm)",
}
_REQUIRED_DISTANCE_COLS = {
    _SEGMENT_START_COL,
    _SEGMENT_END_COL,
    "Distance Threshold (km)",
}


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


def _parse_runner_birthday(
    value: object, runner_name: str, row_label: str
) -> tuple[int, int] | None:
    if _is_blank(value):
        return None
    parsed: pd.Timestamp | None = None
    if isinstance(value, (pd.Timestamp, datetime)):
        parsed = pd.Timestamp(value)
    else:
        string_value = str(value).strip()
        if not string_value:
            return None
        parsed = None
        if re.fullmatch(r"\d{1,2}-[A-Za-z]{3}", string_value):
            try:
                parsed_dt = datetime.strptime(f"{string_value}-2000", "%d-%b-%Y")
                parsed = pd.Timestamp(parsed_dt)
            except ValueError:
                parsed = None
        if parsed is None:
            parsed = pd.to_datetime(string_value, errors="coerce")
    if parsed is None or pd.isna(parsed):
        raise ExcelFormatError(
            f"Runner '{runner_name}' in {row_label} has invalid birthday value '{value}'"
        )
    return (int(parsed.month), int(parsed.day))


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


def _parse_segment_birthday_bonus(
    value: object, seg_name: str, row_label: str
) -> float | None:
    if _is_blank(value):
        return None
    seconds = _coerce_duration_seconds(value)
    if seconds is None:
        raise ExcelFormatError(
            f"Segment '{seg_name}' in {row_label} has invalid Birthday Bonus (secs) value '{value}'"
        )
    if seconds < 0:
        raise ExcelFormatError(
            f"Segment '{seg_name}' in {row_label} has negative Birthday Bonus (secs) value '{value}'"
        )
    return float(seconds)


def _parse_time_bonus_seconds(value: object, row_label: str) -> float:
    """Parse time bonus seconds from cell value, defaulting to 0.0.

    Logs a warning for invalid non-blank values. Unlike birthday bonus,
    negative values are allowed (they add time as a penalty).
    """
    if _is_blank(value):
        return 0.0
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    LOGGER.warning(
        "Invalid time bonus value '%s' in %s; defaulting to 0.0",
        value,
        row_label,
    )
    return 0.0


def _parse_segment_min_distance(value: object, seg_name: str, row_label: str) -> float:
    if _is_blank(value):
        return 0.0
    candidate: float | None = None
    if isinstance(value, (int, float)) and not pd.isna(value):
        candidate = float(value)
    else:
        string_value = str(value).strip()
        if string_value:
            try:
                candidate = float(string_value)
            except ValueError:
                candidate = None
    if candidate is None:
        LOGGER.warning(
            "Segment '%s' in %s has invalid Minimum Distance (m) '%s'; defaulting to 0.",
            seg_name,
            row_label,
            value,
        )
        return 0.0
    if candidate <= 0:
        return 0.0
    return candidate


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
        _SEGMENT_MIN_DISTANCE_COL,
        _SEGMENT_BIRTHDAY_BONUS_COL,
    ]
    for row_offset, (
        seg_id,
        seg_name,
        start_dt,
        end_dt,
        default_time_raw,
        min_distance_raw,
        birthday_bonus_raw,
    ) in enumerate(df[columns].itertuples(index=False, name=None), start=2):
        row_label = f"row {row_offset}"
        # Validate date range early
        if pd.isna(start_dt) or pd.isna(end_dt):
            raise ExcelFormatError(
                f"Segment '{seg_name}' in {row_label} has invalid date(s) in "
                f"'{SEGMENTS_SHEET}' sheet"
            )
        if start_dt > end_dt:
            raise ExcelFormatError(
                f"Segment '{seg_name}' in {row_label} has inverted date range "
                f"(start={start_dt} > end={end_dt}) in '{SEGMENTS_SHEET}' sheet"
            )
        segs.append(
            Segment(
                id=int(seg_id),
                name=str(seg_name),
                start_date=start_dt,
                end_date=end_dt,
                default_time_seconds=_parse_segment_default_time(
                    default_time_raw, str(seg_name), row_label
                ),
                min_distance_meters=_parse_segment_min_distance(
                    min_distance_raw, str(seg_name), row_label
                ),
                birthday_bonus_seconds=_parse_segment_birthday_bonus(
                    birthday_bonus_raw, str(seg_name), row_label
                ),
            )
        )
    return segs


def _windows_fully_overlap(w1: SegmentWindow, w2: SegmentWindow) -> bool:
    """Check if two windows have identical date ranges."""
    return w1.start_date == w2.start_date and w1.end_date == w2.end_date


def read_segment_groups(
    filepath: str | Path, workbook: Optional[object] = None
) -> List[SegmentGroup]:
    """Parse segment rows into SegmentGroup objects, grouping by Segment ID.

    Validates:
    - All rows in a group must have the same Segment Name (error if mismatch).
    - If Default Time appears on multiple rows, values must match.
    - If Min Distance (m) appears on multiple rows, values must match.
    - Warns (doesn't error) if windows fully overlap.
    """
    import warnings

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

    # Check for optional Window Label column
    has_window_label = _SEGMENT_WINDOW_LABEL_COL in df.columns
    # Check for optional Time Bonus column
    has_time_bonus = _SEGMENT_TIME_BONUS_COL in df.columns

    # Collect rows by segment ID
    RawRow = Tuple[
        int,  # segment_id
        str,  # segment_name
        pd.Timestamp,  # start_date
        pd.Timestamp,  # end_date
        str | None,  # window_label
        float | None,  # default_time_seconds
        float,  # min_distance_meters
        float,  # birthday_bonus_seconds
        float,  # time_bonus_seconds
        str,  # row_label
    ]
    rows_by_id: Dict[int, List[RawRow]] = {}

    columns = [
        _SEGMENT_ID_COL,
        _SEGMENT_NAME_COL,
        _SEGMENT_START_COL,
        _SEGMENT_END_COL,
        _SEGMENT_DEFAULT_TIME_COL,
        _SEGMENT_MIN_DISTANCE_COL,
        _SEGMENT_BIRTHDAY_BONUS_COL,
    ]
    if has_window_label:
        columns.insert(4, _SEGMENT_WINDOW_LABEL_COL)
    if has_time_bonus:
        columns.append(_SEGMENT_TIME_BONUS_COL)

    for row_offset, row_values in enumerate(
        df[columns].itertuples(index=False, name=None), start=2
    ):
        row_label = f"row {row_offset}"
        if has_window_label:
            if has_time_bonus:
                (
                    seg_id,
                    seg_name,
                    start_dt,
                    end_dt,
                    window_label_raw,
                    default_time_raw,
                    min_distance_raw,
                    birthday_bonus_raw,
                    time_bonus_raw,
                ) = row_values
            else:
                (
                    seg_id,
                    seg_name,
                    start_dt,
                    end_dt,
                    window_label_raw,
                    default_time_raw,
                    min_distance_raw,
                    birthday_bonus_raw,
                ) = row_values
                time_bonus_raw = None
        else:
            if has_time_bonus:
                (
                    seg_id,
                    seg_name,
                    start_dt,
                    end_dt,
                    default_time_raw,
                    min_distance_raw,
                    birthday_bonus_raw,
                    time_bonus_raw,
                ) = row_values
            else:
                (
                    seg_id,
                    seg_name,
                    start_dt,
                    end_dt,
                    default_time_raw,
                    min_distance_raw,
                    birthday_bonus_raw,
                ) = row_values
                time_bonus_raw = None
            window_label_raw = None

        # Validate date range
        if pd.isna(start_dt) or pd.isna(end_dt):
            raise ExcelFormatError(
                f"Segment '{seg_name}' in {row_label} has invalid date(s) in "
                f"'{SEGMENTS_SHEET}' sheet"
            )
        if start_dt > end_dt:
            raise ExcelFormatError(
                f"Segment '{seg_name}' in {row_label} has inverted date range "
                f"(start={start_dt} > end={end_dt}) in '{SEGMENTS_SHEET}' sheet"
            )

        seg_id_int = int(seg_id)
        window_label = (
            str(window_label_raw).strip()
            if window_label_raw is not None and not _is_blank(window_label_raw)
            else None
        )
        default_time = _parse_segment_default_time(
            default_time_raw, str(seg_name), row_label
        )
        min_distance = _parse_segment_min_distance(
            min_distance_raw, str(seg_name), row_label
        )
        birthday_bonus_parsed = _parse_segment_birthday_bonus(
            birthday_bonus_raw, str(seg_name), row_label
        )
        birthday_bonus = birthday_bonus_parsed if birthday_bonus_parsed else 0.0
        time_bonus = _parse_time_bonus_seconds(time_bonus_raw, row_label)

        row_data: RawRow = (
            seg_id_int,
            str(seg_name).strip(),
            start_dt,
            end_dt,
            window_label,
            default_time,
            min_distance,
            birthday_bonus,
            time_bonus,
            row_label,
        )
        rows_by_id.setdefault(seg_id_int, []).append(row_data)

    # Build SegmentGroup objects with validation
    groups: List[SegmentGroup] = []
    for seg_id, rows in rows_by_id.items():
        # Validate all rows have same segment name
        names = {r[1] for r in rows}
        if len(names) > 1:
            raise ExcelFormatError(
                f"Segment ID {seg_id} has conflicting names: {sorted(names)}. "
                f"All rows with the same Segment ID must have the same Segment Name."
            )
        seg_name = next(iter(names))

        # Validate default_time matches if specified on multiple rows
        default_times = [r[5] for r in rows if r[5] is not None]
        if default_times and len(set(default_times)) > 1:
            raise ExcelFormatError(
                f"Segment '{seg_name}' has conflicting Default Time values: "
                f"{sorted(set(default_times))}. Values must match across rows."
            )
        default_time_seconds = default_times[0] if default_times else None

        # Validate min_distance matches if specified (non-zero) on multiple rows
        min_distances = [r[6] for r in rows if r[6] > 0]
        if min_distances and len(set(min_distances)) > 1:
            raise ExcelFormatError(
                f"Segment '{seg_name}' has conflicting Minimum Distance (m) values: "
                f"{sorted(set(min_distances))}. Values must match across rows."
            )
        min_distance_meters = min_distances[0] if min_distances else None

        # Build windows
        windows: List[SegmentWindow] = []
        for row in rows:
            window = SegmentWindow(
                start_date=row[2],
                end_date=row[3],
                label=row[4],
                birthday_bonus_seconds=row[7],
                time_bonus_seconds=row[8],
            )
            windows.append(window)

        # Warn if any windows fully overlap
        for i, w1 in enumerate(windows):
            for w2 in windows[i + 1 :]:
                if _windows_fully_overlap(w1, w2):
                    warnings.warn(
                        f"Segment '{seg_name}' has fully overlapping windows: "
                        f"{w1.start_date} to {w1.end_date}. This is likely a mistake.",
                        UserWarning,
                        stacklevel=2,
                    )

        groups.append(
            SegmentGroup(
                id=seg_id,
                name=seg_name,
                windows=windows,
                default_time_seconds=default_time_seconds,
                min_distance_meters=min_distance_meters,
            )
        )

    return groups


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
        "Birthday (dd-mmm)",
    ]
    for row_offset, (
        name,
        strava_id,
        refresh_token,
        seg_team,
        dist_team,
        birthday_raw,
    ) in enumerate(df[columns].itertuples(index=False, name=None), start=2):
        if all(
            _is_blank(value)
            for value in (name, strava_id, refresh_token, seg_team, dist_team)
        ):
            continue

        row_label = f"row {row_offset}"
        clean_name = _clean_required_name(name, row_label)
        strava_id_val = _parse_strava_id(strava_id, clean_name, row_label)
        refresh_val = _clean_refresh_token(refresh_token, clean_name, row_label)
        birthday_value = _parse_runner_birthday(birthday_raw, clean_name, row_label)

        runners.append(
            Runner(
                name=clean_name,
                strava_id=strava_id_val,
                refresh_token=refresh_val,
                segment_team=_clean_team(seg_team),
                distance_team=_clean_team(dist_team),
                birthday=birthday_value,
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
    "read_segment_groups",
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
