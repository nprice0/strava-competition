"""Tests for DistanceService resilience when fetchers raise exceptions."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pytest

from strava_competition.models import Runner
from strava_competition.services.distance_service import (
    DistanceService,
    DistanceServiceConfig,
)


def _runner(name: str, runner_id: int, team: str = "Team") -> Runner:
    return Runner(
        name=name,
        strava_id=runner_id,
        refresh_token="rt",
        distance_team=team,
    )


def test_distance_service_continues_on_generic_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ok_runner = _runner("OK", 1)
    failing_runner = _runner("Boom", 2)
    window = (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 10, tzinfo=timezone.utc),
        None,
    )
    sample_activity = {
        "distance": 5000,
        "total_elevation_gain": 50,
        "start_date_local": "2024-01-05T08:00:00Z",
    }

    def fetcher(runner: Runner, *_args: Any, **_kwargs: Any) -> Any:
        if runner.strava_id == failing_runner.strava_id:
            raise ValueError("bad json")
        return [sample_activity]

    service = DistanceService(DistanceServiceConfig(fetcher=fetcher))

    with caplog.at_level(logging.ERROR, logger="DistanceService"):
        outputs = service.process([ok_runner, failing_runner], [window])

    assert outputs[-1][0] == "Distance_Summary"
    summary_rows = {row["Runner"]: row for row in outputs[-1][1]}
    assert summary_rows[ok_runner.name]["Total Distance (km)"] == pytest.approx(5.0)
    assert summary_rows[failing_runner.name]["Total Distance (km)"] == 0

    # Ensure the error was logged but did not abort processing
    assert "distance fetch failed" in caplog.text.lower()
