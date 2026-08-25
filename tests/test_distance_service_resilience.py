"""Tests for DistanceService resilience when fetchers raise exceptions."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pytest

from strava_competition.models import Runner
from strava_competition.distance_aggregation import FETCH_FAILED_MARKER
from strava_competition.services.distance_service import (
    DistanceService,
    DistanceServiceConfig,
)


def _runner(name: str, runner_id: int, team: str = "Team") -> Runner:
    return Runner(
        name=name,
        strava_id=str(runner_id),
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
    # Failed runner is visibly marked, not shown as an inactive 0-run row.
    assert summary_rows[failing_runner.name]["Total Runs"] == FETCH_FAILED_MARKER
    window_rows = {row["Runner"]: row for row in outputs[0][1]}
    assert window_rows[failing_runner.name]["Runs"] == FETCH_FAILED_MARKER

    # Ensure the error was logged but did not abort processing
    assert "distance fetch failed" in caplog.text.lower()


def test_none_from_fetcher_is_treated_as_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A None fetch result (get_activities failure contract) must surface as
    FETCH FAILED, not as a silent 0 km runner."""
    ok_runner = _runner("OK", 1)
    failing_runner = _runner("NoData", 2)
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
            return None
        return [sample_activity]

    service = DistanceService(DistanceServiceConfig(fetcher=fetcher))

    with caplog.at_level(logging.ERROR, logger="DistanceService"):
        outputs = service.process([ok_runner, failing_runner], [window])

    summary_rows = {row["Runner"]: row for row in outputs[-1][1]}
    assert summary_rows[ok_runner.name]["Total Distance (km)"] == pytest.approx(5.0)
    assert summary_rows[failing_runner.name]["Total Runs"] == FETCH_FAILED_MARKER
    window_rows = {row["Runner"]: row for row in outputs[0][1]}
    assert window_rows[failing_runner.name]["Runs"] == FETCH_FAILED_MARKER
    assert "distance fetch failed" in caplog.text.lower()


def test_empty_list_from_fetcher_is_not_a_failure() -> None:
    """A genuinely empty activity list stays a normal 0 km result."""
    idle_runner = _runner("Idle", 1)
    window = (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 10, tzinfo=timezone.utc),
        None,
    )

    service = DistanceService(DistanceServiceConfig(fetcher=lambda *_a, **_k: []))
    outputs = service.process([idle_runner], [window])

    summary_rows = {row["Runner"]: row for row in outputs[-1][1]}
    assert summary_rows[idle_runner.name]["Total Runs"] == 0
    assert summary_rows[idle_runner.name]["Total Runs"] != FETCH_FAILED_MARKER


def test_cancel_event_aborts_before_fetching() -> None:
    """A pre-set cancel event must stop batches before any fetch happens."""
    import threading

    runner = _runner("Runner", 1)
    window = (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 10, tzinfo=timezone.utc),
        None,
    )
    calls: list[str] = []

    def fetcher(r: Runner, *_args: Any, **_kwargs: Any) -> Any:
        calls.append(r.name)
        return []

    cancel_event = threading.Event()
    cancel_event.set()

    service = DistanceService(DistanceServiceConfig(fetcher=fetcher))
    outputs = service.process([runner], [window], cancel_event=cancel_event)

    assert calls == []
    # Outputs are still produced (from an empty cache) rather than crashing.
    assert outputs[-1][0] == "Distance_Summary"
