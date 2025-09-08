"""Global pytest fixtures & helpers.

Adds project root to path and provides reusable fixtures for segment and
distance aggregation tests to avoid duplication across files.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strava_competition.models import SegmentResult, Runner


# --- Factory helpers -------------------------------------------------
def make_segment_results():
    return {
        "Segment One": {
            "Team A": [
                SegmentResult(runner="Runner1", team="Team A", segment="Segment One", attempts=2, fastest_time=100, fastest_date="2025-01-01T10:00:00"),
                SegmentResult(runner="Runner2", team="Team A", segment="Segment One", attempts=1, fastest_time=120, fastest_date="2025-01-01T10:05:00"),
            ],
            "Team B": [
                SegmentResult(runner="Runner3", team="Team B", segment="Segment One", attempts=3, fastest_time=90, fastest_date="2025-01-01T09:50:00"),
            ],
        },
        "Segment Two": {
            "Team A": [
                SegmentResult(runner="Runner1", team="Team A", segment="Segment Two", attempts=1, fastest_time=110, fastest_date="2025-01-02T11:00:00"),
            ],
            "Team C": [
                SegmentResult(runner="Runner4", team="Team C", segment="Segment Two", attempts=2, fastest_time=80, fastest_date="2025-01-02T11:15:00"),
            ],
        },
    }


def make_activity(distance_m, elev_m, iso):
    return {"distance": distance_m, "total_elevation_gain": elev_m, "start_date_local": iso}


# --- Fixtures --------------------------------------------------------
@pytest.fixture
def segment_results():
    return make_segment_results()


@pytest.fixture
def distance_runners():
    r1 = Runner(name="Alice", strava_id=1, refresh_token="tok1", segment_team=None, distance_team="DTeam")
    r2 = Runner(name="Ben", strava_id=2, refresh_token="tok2", segment_team=None, distance_team="DTeam")
    return [r1, r2]


@pytest.fixture
def distance_windows():
    start = datetime(2025, 1, 1)
    end = datetime(2025, 1, 31)
    return [(start, end, 5.0)]


@pytest.fixture
def distance_activity_cache():
    return {
        1: [
            make_activity(6000.0, 50.0, "2025-01-05T10:00:00Z"),
            make_activity(4000.0, 30.0, "2025-01-10T10:00:00Z"),
        ],
        2: [
            make_activity(7000.0, 40.0, "2025-01-06T10:00:00Z"),
            make_activity(3000.0, 20.0, "2025-01-11T10:30:00Z"),
        ],
    }


@pytest.fixture
def assert_summary_columns():
    def _assert(df):
        expected_cols = {
            "Team",
            "Runners Participating",
            "Segments With Participation",
            "Total Attempts",
            "Average Fastest Time (sec)",
            "Total Fastest Times (sec)",
        }
        missing = expected_cols - set(df.columns)
        assert not missing, f"Missing columns in summary: {missing}"
    return _assert

