"""Global pytest fixtures & helpers.

Adds project root to path and provides reusable fixtures for segment and
distance aggregation tests to avoid duplication across files.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Ensure cache hashing has a deterministic salt during tests.
os.environ.setdefault("STRAVA_CACHE_ID_SALT", "pytest-salt")

from strava_competition.models import SegmentResult, Runner  # noqa: E402


# --- Shared test doubles for requests.Session mocking ----------------
class FakeResp:
    """Unified fake response for mocking requests.Session methods.

    Supports all common use cases across test files:
    - `data`: JSON body returned by .json()
    - `text`: Raw text body returned by .text property
    - `headers`: Response headers dict
    - `status_code`: HTTP status code

    If `data` is an Exception instance, .json() will raise it (useful for
    testing JSON parse error handling).
    """

    def __init__(
        self,
        status_code: int = 200,
        data: dict | list | Exception | None = None,
        *,
        text: str | None = None,
        headers: dict | None = None,
    ):
        self.status_code = status_code
        self._data = data
        self._text = text
        self.headers = headers or {}

    def json(self) -> Any:
        if isinstance(self._data, Exception):
            raise self._data
        return self._data

    def raise_for_status(self) -> None:
        if 400 <= self.status_code:
            import requests

            raise requests.exceptions.HTTPError(response=self)  # type: ignore[arg-type]

    @property
    def content(self) -> bytes:
        if self._text is not None:
            return self._text.encode()
        if self._data is not None:
            import json

            return json.dumps(self._data).encode()
        return b""

    @property
    def text(self) -> str:
        if self._text is not None:
            return self._text
        return self.content.decode()


def _patch_session(monkeypatch: pytest.MonkeyPatch, mock_session: Any) -> None:
    """Patch get_default_session in all modules that import it.

    Also resets the default client to force re-creation with the mocked session.

    Args:
        monkeypatch: pytest monkeypatch fixture
        mock_session: A mock session object with .get() and/or .post() methods
    """
    from strava_competition import strava_api
    from strava_competition.strava_client import (
        session as session_module,
    )

    monkeypatch.setattr(session_module, "get_default_session", lambda: mock_session)
    monkeypatch.setattr(strava_api, "get_default_session", lambda: mock_session)
    # Reset the module-level cached client so get_default_client creates a fresh one
    monkeypatch.setattr(strava_api, "_default_client", None)
    # Force recreation and update the module-level reference
    strava_api.get_default_client()


# --- Factory helpers -------------------------------------------------
def make_segment_results() -> dict[str, dict[str, list[SegmentResult]]]:
    return {
        "Segment One": {
            "Team A": [
                SegmentResult(
                    runner="Runner1",
                    team="Team A",
                    segment="Segment One",
                    attempts=2,
                    fastest_time=100,
                    fastest_date="2025-01-01T10:00:00",  # type: ignore[arg-type]
                    fastest_distance_m=3100.0,
                ),
                SegmentResult(
                    runner="Runner2",
                    team="Team A",
                    segment="Segment One",
                    attempts=1,
                    fastest_time=120,
                    fastest_date="2025-01-01T10:05:00",  # type: ignore[arg-type]
                    fastest_distance_m=3200.0,
                ),
            ],
            "Team B": [
                SegmentResult(
                    runner="Runner3",
                    team="Team B",
                    segment="Segment One",
                    attempts=3,
                    fastest_time=90,
                    fastest_date="2025-01-01T09:50:00",  # type: ignore[arg-type]
                    fastest_distance_m=3000.0,
                ),
            ],
        },
        "Segment Two": {
            "Team A": [
                SegmentResult(
                    runner="Runner1",
                    team="Team A",
                    segment="Segment Two",
                    attempts=1,
                    fastest_time=110,
                    fastest_date="2025-01-02T11:00:00",  # type: ignore[arg-type]
                    fastest_distance_m=4100.0,
                ),
            ],
            "Team C": [
                SegmentResult(
                    runner="Runner4",
                    team="Team C",
                    segment="Segment Two",
                    attempts=2,
                    fastest_time=80,
                    fastest_date="2025-01-02T11:15:00",  # type: ignore[arg-type]
                    fastest_distance_m=4200.0,
                ),
            ],
        },
    }


def make_activity(distance_m: Any, elev_m: Any, iso: Any) -> dict[str, Any]:
    return {
        "distance": distance_m,
        "total_elevation_gain": elev_m,
        "start_date_local": iso,
    }


# --- Fixtures --------------------------------------------------------
@pytest.fixture
def segment_results() -> dict[str, dict[str, list[SegmentResult]]]:
    return make_segment_results()


@pytest.fixture
def distance_runners() -> list[Runner]:
    r1 = Runner(
        name="Alice",
        strava_id=1,
        refresh_token="tok1",
        segment_team=None,
        distance_team="DTeam",
    )
    r2 = Runner(
        name="Ben",
        strava_id=2,
        refresh_token="tok2",
        segment_team=None,
        distance_team="DTeam",
    )
    return [r1, r2]


@pytest.fixture
def distance_windows() -> list[tuple[datetime, datetime, float]]:
    start = datetime(2025, 1, 1)
    end = datetime(2025, 1, 31)
    return [(start, end, 5.0)]


@pytest.fixture
def distance_activity_cache() -> dict[int, list[dict[str, Any]]]:
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
def assert_summary_columns() -> Any:
    def _assert(df: Any) -> None:
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
