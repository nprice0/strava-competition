"""Tests for segment split windows feature.

Covers:
- read_segment_groups() parsing and validation
- Multi-window processing and best-time selection
- Per-window birthday bonus application
- Sheet naming when SEGMENT_SPLIT_WINDOWS_ENABLED is disabled
"""

from __future__ import annotations

import os
import tempfile
import warnings
from datetime import datetime
from typing import List

import pandas as pd
import pytest

from strava_competition import excel_reader
from strava_competition.errors import ExcelFormatError
from strava_competition.models import SegmentGroup, SegmentWindow
from strava_competition.services.segment_service import SegmentService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_segment_workbook(
    path: str,
    segment_rows: List[dict],
    runner_rows: List[dict] | None = None,
) -> None:
    """Create a minimal workbook with Segment Series and Runners sheets."""
    segs_df = pd.DataFrame(segment_rows)
    if runner_rows is None:
        runner_rows = [
            {
                "Name": "Alice",
                "Strava ID": 1,
                "Refresh Token": "rt1",
                "Segment Series Team": "Red",
                "Distance Series Team": None,
                "Birthday (dd-mmm)": "15-Jan",
            },
        ]
    runners_df = pd.DataFrame(runner_rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        segs_df.to_excel(writer, sheet_name="Segment Series", index=False)
        runners_df.to_excel(writer, sheet_name="Runners", index=False)


# ---------------------------------------------------------------------------
# read_segment_groups() Tests
# ---------------------------------------------------------------------------


class TestReadSegmentGroups:
    """Tests for excel_reader.read_segment_groups()."""

    def test_single_window_produces_one_group(self):
        """A single row produces one SegmentGroup with one window."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 10,
                    }
                ],
            )
            groups = excel_reader.read_segment_groups(path)

            assert len(groups) == 1
            grp = groups[0]
            assert grp.id == 101
            assert grp.name == "Hill Climb"
            assert len(grp.windows) == 1
            assert grp.windows[0].birthday_bonus_seconds == 10.0

    def test_multiple_rows_same_id_grouped(self):
        """Rows with the same Segment ID are grouped into one SegmentGroup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 15),
                        "Default Time": "00:10:00",
                        "Minimum Distance (m)": 500,
                        "Birthday Bonus (secs)": 10,
                        "Window Label": "Week 1",
                    },
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 16),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 500,
                        "Birthday Bonus (secs)": 15,
                        "Window Label": "Week 2",
                    },
                ],
            )
            groups = excel_reader.read_segment_groups(path)

            assert len(groups) == 1
            grp = groups[0]
            assert grp.id == 101
            assert len(grp.windows) == 2
            assert grp.default_time_seconds == 600.0  # 10 minutes
            assert grp.min_distance_meters == 500.0

            # Windows have different birthday bonuses
            labels = {w.label for w in grp.windows}
            assert labels == {"Week 1", "Week 2"}
            bonuses = {w.birthday_bonus_seconds for w in grp.windows}
            assert bonuses == {10.0, 15.0}

    def test_different_segment_ids_separate_groups(self):
        """Rows with different Segment IDs produce separate groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                    },
                    {
                        "Segment ID": 202,
                        "Segment Name": "Sprint",
                        "Start Date": datetime(2024, 2, 1),
                        "End Date": datetime(2024, 2, 28),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                    },
                ],
            )
            groups = excel_reader.read_segment_groups(path)

            assert len(groups) == 2
            ids = {g.id for g in groups}
            assert ids == {101, 202}

    def test_conflicting_segment_names_raises_error(self):
        """Rows with same ID but different names raise ExcelFormatError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 15),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                    },
                    {
                        "Segment ID": 101,
                        "Segment Name": "Mountain Sprint",  # Different name!
                        "Start Date": datetime(2024, 1, 16),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                    },
                ],
            )
            with pytest.raises(ExcelFormatError, match="conflicting names"):
                excel_reader.read_segment_groups(path)

    def test_conflicting_default_time_raises_error(self):
        """Rows with same ID but different Default Time raise ExcelFormatError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 15),
                        "Default Time": "00:10:00",
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                    },
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 16),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": "00:15:00",  # Different default time!
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                    },
                ],
            )
            with pytest.raises(ExcelFormatError, match="conflicting Default Time"):
                excel_reader.read_segment_groups(path)

    def test_conflicting_min_distance_raises_error(self):
        """Rows with same ID but different min distance raise ExcelFormatError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 15),
                        "Default Time": None,
                        "Minimum Distance (m)": 500,
                        "Birthday Bonus (secs)": 0,
                    },
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 16),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 1000,  # Different!
                        "Birthday Bonus (secs)": 0,
                    },
                ],
            )
            with pytest.raises(ExcelFormatError, match="conflicting Minimum Distance"):
                excel_reader.read_segment_groups(path)

    def test_overlapping_windows_warns(self):
        """Fully overlapping windows emit a warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                    },
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),  # Same dates = overlap
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                    },
                ],
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                groups = excel_reader.read_segment_groups(path)
                assert len(groups) == 1
                assert len(groups[0].windows) == 2
                # Check warning was emitted
                overlap_warnings = [
                    x for x in w if "overlapping windows" in str(x.message)
                ]
                assert len(overlap_warnings) >= 1

    def test_birthday_bonus_defaults_to_zero(self):
        """Missing birthday bonus defaults to 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": None,  # Blank
                    }
                ],
            )
            groups = excel_reader.read_segment_groups(path)
            assert groups[0].windows[0].birthday_bonus_seconds == 0.0


# ---------------------------------------------------------------------------
# SegmentService.process_groups() Tests
# ---------------------------------------------------------------------------


class TestProcessGroups:
    """Tests for SegmentService.process_groups() multi-window logic."""

    def test_best_time_selected_across_windows(self, monkeypatch):
        """Runner's fastest time across all windows is selected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 15),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                        "Window Label": "Week 1",
                    },
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 16),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                        "Window Label": "Week 2",
                    },
                ],
            )
            groups = excel_reader.read_segment_groups(path)
            runners = excel_reader.read_runners(path)

            # Alice: slower in Week 1 (120s), faster in Week 2 (100s)
            def fake_get_efforts(runner, segment_id, start_date, end_date):
                if start_date.day <= 15:  # Week 1
                    return [
                        {
                            "elapsed_time": 120,
                            "start_date_local": "2024-01-10T09:00:00Z",
                        }
                    ]
                else:  # Week 2
                    return [
                        {
                            "elapsed_time": 100,
                            "start_date_local": "2024-01-20T09:00:00Z",
                        }
                    ]

            import strava_competition.services.segment_service as mod

            monkeypatch.setattr(mod, "get_segment_efforts", fake_get_efforts)
            monkeypatch.setattr(mod, "get_activities", lambda *a, **k: [])
            monkeypatch.setattr(mod, "FORCE_ACTIVITY_SCAN_FALLBACK", False)
            monkeypatch.setattr(mod, "SEGMENT_SPLIT_WINDOWS_ENABLED", True)

            service = SegmentService(max_workers=1)
            results = service.process_groups(groups, runners)

            assert "Hill Climb" in results
            team_results = results["Hill Climb"]
            assert "Red" in team_results
            alice_result = team_results["Red"][0]
            assert alice_result.runner == "Alice"
            # Best time should be 100s from Week 2
            assert alice_result.fastest_time == 100.0
            # Total attempts = 2 (one per window)
            assert alice_result.attempts == 2

    def test_per_window_birthday_bonus_applied(self, monkeypatch):
        """Birthday bonus from the specific window is applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 30,  # 30 second bonus
                        "Window Label": "January",
                    },
                ],
                runner_rows=[
                    {
                        "Name": "Alice",
                        "Strava ID": 1,
                        "Refresh Token": "rt1",
                        "Segment Series Team": "Red",
                        "Distance Series Team": None,
                        "Birthday (dd-mmm)": "15-Jan",  # Birthday on 15th
                    },
                ],
            )
            groups = excel_reader.read_segment_groups(path)
            runners = excel_reader.read_runners(path)

            # Alice runs on her birthday with 120s elapsed
            def fake_get_efforts(runner, segment_id, start_date, end_date):
                return [
                    {
                        "elapsed_time": 120,
                        "start_date_local": "2024-01-15T09:00:00Z",  # Birthday!
                    }
                ]

            import strava_competition.services.segment_service as mod

            monkeypatch.setattr(mod, "get_segment_efforts", fake_get_efforts)
            monkeypatch.setattr(mod, "get_activities", lambda *a, **k: [])
            monkeypatch.setattr(mod, "FORCE_ACTIVITY_SCAN_FALLBACK", False)
            monkeypatch.setattr(mod, "SEGMENT_SPLIT_WINDOWS_ENABLED", True)

            service = SegmentService(max_workers=1)
            results = service.process_groups(groups, runners)

            alice_result = results["Hill Climb"]["Red"][0]
            # 120s - 30s birthday bonus = 90s
            assert alice_result.fastest_time == 90.0
            assert alice_result.birthday_bonus_applied is True

    def test_different_bonus_per_window(self, monkeypatch):
        """Different windows can have different birthday bonuses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 15),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 10,  # Small bonus
                        "Window Label": "Week 1",
                    },
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 16),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 50,  # Big bonus
                        "Window Label": "Week 2",
                    },
                ],
                runner_rows=[
                    {
                        "Name": "Alice",
                        "Strava ID": 1,
                        "Refresh Token": "rt1",
                        "Segment Series Team": "Red",
                        "Distance Series Team": None,
                        "Birthday (dd-mmm)": "20-Jan",  # Birthday in Week 2
                    },
                ],
            )
            groups = excel_reader.read_segment_groups(path)
            runners = excel_reader.read_runners(path)

            def fake_get_efforts(runner, segment_id, start_date, end_date):
                if start_date.day <= 15:  # Week 1
                    return [
                        {
                            "elapsed_time": 100,
                            "start_date_local": "2024-01-10T09:00:00Z",
                        }
                    ]
                else:  # Week 2 - birthday run
                    return [
                        {
                            "elapsed_time": 120,
                            "start_date_local": "2024-01-20T09:00:00Z",  # Birthday
                        }
                    ]

            import strava_competition.services.segment_service as mod

            monkeypatch.setattr(mod, "get_segment_efforts", fake_get_efforts)
            monkeypatch.setattr(mod, "get_activities", lambda *a, **k: [])
            monkeypatch.setattr(mod, "FORCE_ACTIVITY_SCAN_FALLBACK", False)
            monkeypatch.setattr(mod, "SEGMENT_SPLIT_WINDOWS_ENABLED", True)

            service = SegmentService(max_workers=1)
            results = service.process_groups(groups, runners)

            alice_result = results["Hill Climb"]["Red"][0]
            # Week 1: 100s (no bonus)
            # Week 2: 120s - 50s = 70s (birthday bonus)
            # Best is 70s from Week 2
            assert alice_result.fastest_time == 70.0
            assert alice_result.birthday_bonus_applied is True


# ---------------------------------------------------------------------------
# Sheet Naming Tests (disabled mode)
# ---------------------------------------------------------------------------


class TestSheetNaming:
    """Tests for sheet naming when SEGMENT_SPLIT_WINDOWS_ENABLED=False."""

    def test_single_window_uses_segment_name(self, monkeypatch):
        """Single-window segment uses just the segment name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                    },
                ],
            )
            groups = excel_reader.read_segment_groups(path)
            runners = excel_reader.read_runners(path)

            import strava_competition.services.segment_service as mod

            monkeypatch.setattr(mod, "get_segment_efforts", lambda *a, **k: [])
            monkeypatch.setattr(mod, "get_activities", lambda *a, **k: [])
            monkeypatch.setattr(mod, "FORCE_ACTIVITY_SCAN_FALLBACK", False)
            monkeypatch.setattr(mod, "SEGMENT_SPLIT_WINDOWS_ENABLED", False)

            service = SegmentService(max_workers=1)
            results = service.process_groups(groups, runners)

            # Single window = just segment name
            assert "Hill Climb" in results

    def test_multi_window_uses_label(self, monkeypatch):
        """Multi-window segment uses '{name} - {label}' format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 15),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                        "Window Label": "Week 1",
                    },
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 16),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                        "Window Label": "Week 2",
                    },
                ],
            )
            groups = excel_reader.read_segment_groups(path)
            runners = excel_reader.read_runners(path)

            import strava_competition.services.segment_service as mod

            monkeypatch.setattr(mod, "get_segment_efforts", lambda *a, **k: [])
            monkeypatch.setattr(mod, "get_activities", lambda *a, **k: [])
            monkeypatch.setattr(mod, "FORCE_ACTIVITY_SCAN_FALLBACK", False)
            monkeypatch.setattr(mod, "SEGMENT_SPLIT_WINDOWS_ENABLED", False)

            service = SegmentService(max_workers=1)
            results = service.process_groups(groups, runners)

            assert "Hill Climb - Week 1" in results
            assert "Hill Climb - Week 2" in results

    def test_multi_window_no_label_uses_dates(self, monkeypatch):
        """Multi-window without labels uses date range in sheet name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "input.xlsx")
            _make_segment_workbook(
                path,
                [
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 1),
                        "End Date": datetime(2024, 1, 15),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                    },
                    {
                        "Segment ID": 101,
                        "Segment Name": "Hill Climb",
                        "Start Date": datetime(2024, 1, 16),
                        "End Date": datetime(2024, 1, 31),
                        "Default Time": None,
                        "Minimum Distance (m)": 0,
                        "Birthday Bonus (secs)": 0,
                    },
                ],
            )
            groups = excel_reader.read_segment_groups(path)
            runners = excel_reader.read_runners(path)

            import strava_competition.services.segment_service as mod

            monkeypatch.setattr(mod, "get_segment_efforts", lambda *a, **k: [])
            monkeypatch.setattr(mod, "get_activities", lambda *a, **k: [])
            monkeypatch.setattr(mod, "FORCE_ACTIVITY_SCAN_FALLBACK", False)
            monkeypatch.setattr(mod, "SEGMENT_SPLIT_WINDOWS_ENABLED", False)

            service = SegmentService(max_workers=1)
            results = service.process_groups(groups, runners)

            assert "Hill Climb - 2024-01-01 to 2024-01-15" in results
            assert "Hill Climb - 2024-01-16 to 2024-01-31" in results
