"""Regression tests for mixed timezone-aware/naive date columns.

Reproduces the bug where a Distance/Segment sheet containing a mix of
timezone-aware date cells (ISO-8601 strings ending in ``Z``) and naive date
cells caused pandas to coerce the naive rows to ``NaT`` (or raise
``ValueError: Mixed timezones``), silently dropping otherwise-valid rows.
See :func:`strava_competition.excel_reader._parse_date_column`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from strava_competition.excel_reader import (
    read_distance_windows,
    read_segment_groups,
)

_RUNNERS_COLUMNS = [
    "Name",
    "Strava ID",
    "Refresh Token",
    "Segment Series Team",
    "Distance Series Team",
]
_SEGMENT_COLUMNS = [
    "Segment ID",
    "Segment Name",
    "Start Date",
    "End Date",
    "Default Time",
    "Minimum Distance (m)",
    "Birthday Bonus (secs)",
]


def _write_workbook(
    path: Path,
    *,
    segments: pd.DataFrame | None = None,
    distance: pd.DataFrame | None = None,
) -> None:
    """Write a minimal workbook with the required sheets."""
    seg_df = (
        segments if segments is not None else pd.DataFrame(columns=_SEGMENT_COLUMNS)
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        seg_df.to_excel(writer, sheet_name="Segment Series", index=False)
        pd.DataFrame(columns=_RUNNERS_COLUMNS).to_excel(
            writer, sheet_name="Runners", index=False
        )
        if distance is not None:
            distance.to_excel(writer, sheet_name="Distance Series", index=False)


def test_read_distance_windows_handles_mixed_timezones(tmp_path: Path) -> None:
    """Both a tz-aware row and a naive row must survive parsing."""
    path = tmp_path / "input.xlsx"
    distance = pd.DataFrame(
        [
            {
                # timezone-aware ISO string
                "Start Date": "2025-01-01T00:00:00Z",
                "End Date": "2025-01-31T00:00:00Z",
                "Distance Threshold (km)": 5,
            },
            {
                # timezone-naive plain date string
                "Start Date": "2025-08-01",
                "End Date": "2025-08-31",
                "Distance Threshold (km)": 10,
            },
        ]
    )
    _write_workbook(path, distance=distance)

    windows = read_distance_windows(path)

    assert len(windows) == 2, "Naive-date row was dropped (mixed-timezone regression)"
    assert windows[0][0] == pd.Timestamp("2025-01-01")
    assert windows[1][0] == pd.Timestamp("2025-08-01")
    # Boundaries must be timezone-naive so downstream comparisons work.
    assert windows[0][0].tzinfo is None
    assert windows[1][1].tzinfo is None


def test_read_segment_groups_handles_mixed_timezones(tmp_path: Path) -> None:
    """A segment with mixed tz-aware and naive windows keeps every window."""
    path = tmp_path / "input.xlsx"
    segments = pd.DataFrame(
        [
            {
                "Segment ID": 123,
                "Segment Name": "Test Segment",
                "Start Date": "2025-01-01T00:00:00Z",  # tz-aware
                "End Date": "2025-01-31T00:00:00Z",
                "Default Time": 100,
                "Minimum Distance (m)": 0,
                "Birthday Bonus (secs)": 0,
            },
            {
                "Segment ID": 123,
                "Segment Name": "Test Segment",
                "Start Date": "2025-08-01",  # tz-naive
                "End Date": "2025-08-31",
                "Default Time": 100,
                "Minimum Distance (m)": 0,
                "Birthday Bonus (secs)": 0,
            },
        ]
    )
    _write_workbook(path, segments=segments)

    groups = read_segment_groups(path)

    assert len(groups) == 1
    windows = groups[0].windows
    assert len(windows) == 2, (
        "Naive-date window was dropped (mixed-timezone regression)"
    )
    starts = sorted(w.start_date for w in windows)
    assert starts[0] == pd.Timestamp("2025-01-01")
    assert starts[1] == pd.Timestamp("2025-08-01")
    assert all(w.start_date.tzinfo is None for w in windows)
