"""End Date inclusivity tests.

Date-only End Date cells parse to midnight; the reader promotes them to
23:59:59.999999 so efforts/activities anywhere on the final day are included
(see ``excel_reader._promote_date_only_end``). These tests pin the boundary
behaviour for both segment windows and distance windows.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from strava_competition import excel_reader
from strava_competition.distance_aggregation import build_distance_outputs
from strava_competition.models import Runner
from strava_competition.services.segment_service import SegmentService

_END_OF_DAY = datetime(2024, 1, 31, 23, 59, 59, 999999)


def _write_workbook(
    path: Path,
    *,
    segment_rows: list[dict] | None = None,
    distance_rows: list[dict] | None = None,
    runner_rows: list[dict] | None = None,
) -> None:
    if runner_rows is None:
        runner_rows = [
            {
                "Name": "Alice",
                "Strava ID": 1,
                "Refresh Token": "rt1",
                "Segment Series Team": "Red",
                "Distance Series Team": "Red",
                "Birthday (dd-mmm)": None,
            },
        ]
    if segment_rows is None:
        segment_rows = []
    seg_columns = [
        "Segment ID",
        "Segment Name",
        "Start Date",
        "End Date",
        "Default Time",
        "Minimum Distance (m)",
        "Birthday Bonus (secs)",
    ]
    seg_df = (
        pd.DataFrame(segment_rows)
        if segment_rows
        else pd.DataFrame(columns=seg_columns)
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        seg_df.to_excel(writer, sheet_name="Segment Series", index=False)
        pd.DataFrame(runner_rows).to_excel(writer, sheet_name="Runners", index=False)
        if distance_rows is not None:
            pd.DataFrame(distance_rows).to_excel(
                writer, sheet_name="Distance Series", index=False
            )


def test_segment_window_end_date_promoted_to_end_of_day(tmp_path: Path) -> None:
    path = tmp_path / "input.xlsx"
    _write_workbook(
        path,
        segment_rows=[
            {
                "Segment ID": 101,
                "Segment Name": "Hill Climb",
                "Start Date": datetime(2024, 1, 1),
                "End Date": datetime(2024, 1, 31),
                "Default Time": None,
                "Minimum Distance (m)": 0,
                "Birthday Bonus (secs)": 0,
            }
        ],
    )
    groups = excel_reader.read_segment_groups(path)
    window = groups[0].windows[0]
    assert window.start_date == pd.Timestamp("2024-01-01 00:00:00")
    assert window.end_date == pd.Timestamp(_END_OF_DAY)


def test_segment_end_date_with_explicit_time_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "input.xlsx"
    _write_workbook(
        path,
        segment_rows=[
            {
                "Segment ID": 101,
                "Segment Name": "Hill Climb",
                "Start Date": datetime(2024, 1, 1),
                "End Date": datetime(2024, 1, 31, 12, 30),
                "Default Time": None,
                "Minimum Distance (m)": 0,
                "Birthday Bonus (secs)": 0,
            }
        ],
    )
    groups = excel_reader.read_segment_groups(path)
    assert groups[0].windows[0].end_date == pd.Timestamp("2024-01-31 12:30:00")


def test_segment_efforts_on_end_date_included(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Efforts at 00:00, 09:00 and 23:59 on the End Date count; day after doesn't."""
    path = tmp_path / "input.xlsx"
    _write_workbook(
        path,
        segment_rows=[
            {
                "Segment ID": 101,
                "Segment Name": "Hill Climb",
                "Start Date": datetime(2024, 1, 1),
                "End Date": datetime(2024, 1, 31),
                "Default Time": None,
                "Minimum Distance (m)": 0,
                "Birthday Bonus (secs)": 0,
            }
        ],
    )
    groups = excel_reader.read_segment_groups(path)
    runners = excel_reader.read_runners(path)

    def fake_get_activities(
        runner: Any, start_date: Any, end_date: Any, **kwargs: Any
    ) -> Any:
        return [{"id": 9001}]

    efforts = [
        # Included: on the End Date
        {
            "segment": {"id": 101},
            "elapsed_time": 120,
            "start_date_local": "2024-01-31T00:00:00Z",
        },
        {
            "segment": {"id": 101},
            "elapsed_time": 110,
            "start_date_local": "2024-01-31T09:00:00Z",
        },
        {
            "segment": {"id": 101},
            "elapsed_time": 100,
            "start_date_local": "2024-01-31T23:59:00Z",
        },
        # Excluded: the day after
        {
            "segment": {"id": 101},
            "elapsed_time": 50,
            "start_date_local": "2024-02-01T00:30:00Z",
        },
    ]

    def fake_get_detail(runner: Any, activity_id: Any, **kwargs: Any) -> Any:
        return {"id": activity_id, "segment_efforts": efforts}

    import strava_competition.services.segment_service as mod

    monkeypatch.setattr(mod, "get_activities", fake_get_activities)
    monkeypatch.setattr(
        "strava_competition.activity_scan.scanner.get_activity_with_efforts",
        fake_get_detail,
    )

    service = SegmentService(max_workers=1)
    results = service.process_groups(groups, runners)

    alice = results["Hill Climb"]["Red"][0]
    assert alice.attempts == 3
    assert alice.fastest_time == 100.0


def _activity(iso_local: str) -> dict:
    return {
        "distance": 5000,
        "total_elevation_gain": 10,
        "start_date_local": iso_local,
    }


def test_distance_activities_on_end_date_included(tmp_path: Path) -> None:
    """Activities at 00:00, 09:00 and 23:59 on the End Date count; day after doesn't."""
    path = tmp_path / "input.xlsx"
    _write_workbook(
        path,
        distance_rows=[
            {
                "Start Date": datetime(2024, 1, 1),
                "End Date": datetime(2024, 1, 31),
                "Distance Threshold (km)": None,
            }
        ],
    )
    windows = excel_reader.read_distance_windows(path)
    assert windows[0][1] == pd.Timestamp(_END_OF_DAY)

    runner = Runner(
        name="Alice", strava_id="1", refresh_token="rt", distance_team="Red"
    )
    cache = {
        "1": [
            _activity("2024-01-31T00:00:00Z"),
            _activity("2024-01-31T09:00:00Z"),
            _activity("2024-01-31T23:59:00Z"),
            _activity("2024-02-01T00:30:00Z"),  # excluded
        ]
    }
    outputs = build_distance_outputs([runner], list(windows), cache)
    window_rows = outputs[0][1]
    assert window_rows[0]["Runs"] == 3
    assert window_rows[0]["Total Distance (km)"] == pytest.approx(15.0)
