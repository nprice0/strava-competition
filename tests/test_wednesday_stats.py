"""Tests for wednesday_stats end-date handling."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

import strava_competition.tools.wednesday_stats as ws
from strava_competition.models import Runner


def test_parse_end_date_clamps_to_end_of_day() -> None:
    assert ws._parse_end_date("2026-04-08") == datetime(2026, 4, 8, 23, 59, 59, 999999)


def test_activity_on_final_day_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """An activity at 10:00 on the end date must pass the fetch filter.

    ``get_activities`` filters with ``start <= t <= end``; the fake below
    mirrors that contract to prove the end-of-day clamp includes the final
    target day.
    """
    runner = Runner(name="R", strava_id="1", refresh_token="rt")
    start_date = ws._parse_date("2026-01-01")
    end_date = ws._parse_end_date("2026-04-08")  # a Wednesday

    final_day_activity = {
        "start_date_local": "2026-04-08T10:00:00Z",
        "distance": 5000,
    }

    def fake_get_activities(
        _runner: Runner, start: datetime, end: datetime, **_kw: Any
    ) -> list[dict[str, Any]]:
        activity_start = datetime(2026, 4, 8, 10, 0, 0)
        if start <= activity_start <= end:
            return [final_day_activity]
        return []

    monkeypatch.setattr(ws, "get_activities", fake_get_activities)

    class _FakeClient:
        def ensure_runner_token(self, _runner: Runner) -> None:
            return None

    matched = ws.fetch_matching_activities(
        runner,
        start_date,
        end_date,
        weekday=2,  # Wednesday
        client=_FakeClient(),  # type: ignore[arg-type]
    )

    assert matched == [final_day_activity]


def test_midnight_end_would_exclude_final_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: the unclamped midnight end excluded the final day."""
    runner = Runner(name="R", strava_id="1", refresh_token="rt")
    start_date = ws._parse_date("2026-01-01")
    midnight_end = ws._parse_date("2026-04-08")

    def fake_get_activities(
        _runner: Runner, start: datetime, end: datetime, **_kw: Any
    ) -> list[dict[str, Any]]:
        activity_start = datetime(2026, 4, 8, 10, 0, 0)
        if start <= activity_start <= end:
            return [{"start_date_local": "2026-04-08T10:00:00Z"}]
        return []

    monkeypatch.setattr(ws, "get_activities", fake_get_activities)

    class _FakeClient:
        def ensure_runner_token(self, _runner: Runner) -> None:
            return None

    matched = ws.fetch_matching_activities(
        runner,
        start_date,
        midnight_end,
        weekday=2,
        client=_FakeClient(),  # type: ignore[arg-type]
    )

    assert matched == []
