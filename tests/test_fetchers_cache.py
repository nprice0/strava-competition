"""Tests for the activity stream cache in matching.fetchers."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from strava_competition.tools.geometry.fetchers import fetch_activity_stream
from strava_competition.tools.geometry.models import ActivityTrack
from strava_competition.models import Runner


class _RunnerFactory:
    """Simple runner factory that returns distinct runners for tests."""

    def __init__(self) -> None:
        self._counter = 0

    def make(self) -> Runner:
        """Create a runner instance with a unique Strava identifier."""
        self._counter += 1
        return Runner(
            name=f"Runner {self._counter}",
            strava_id=str(self._counter),
            refresh_token="token",
            segment_team="A",
        )


def _fake_payload(points: List[List[float]], times: List[float]) -> Dict[str, Any]:
    """Build a minimal API payload compatible with ActivityTrack.from_payload."""
    return {
        "latlng": points,
        "time": times,
        "activity_id": 123,
        "metadata": {"source": "test"},
    }


def test_fetch_activity_stream_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure consecutive requests reuse the cached activity stream."""
    calls: List[int] = []

    def fake_api_fetch_activity_stream(runner: Runner, activity_id: int) -> dict:
        calls.append(activity_id)
        points = [[0.0, 0.0], [1.0, 1.0]]
        times = [0.0, 10.0]
        return _fake_payload(points, times)

    monkeypatch.setattr(
        "strava_competition.tools.geometry.fetchers.api_fetch_activity_stream",
        fake_api_fetch_activity_stream,
    )

    runner_factory = _RunnerFactory()
    runner = runner_factory.make()

    first = fetch_activity_stream(runner, 999)
    second = fetch_activity_stream(runner, 999)

    assert isinstance(first, ActivityTrack)
    assert second is first
    assert calls == [999]


def test_fetch_activity_stream_cache_is_keyed_by_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate cache keys include the runner id to avoid cross-runner reuse."""
    calls: List[int] = []

    def fake_api_fetch_activity_stream(runner: Runner, activity_id: int) -> dict:
        calls.append(activity_id)
        points = [[0.0, 0.0], [1.0, 1.0]]
        times = [0.0, 10.0]
        payload = _fake_payload(points, times)
        payload["activity_id"] = activity_id
        return payload

    monkeypatch.setattr(
        "strava_competition.tools.geometry.fetchers.api_fetch_activity_stream",
        fake_api_fetch_activity_stream,
    )

    runner_factory = _RunnerFactory()
    runner_one = runner_factory.make()
    runner_two = runner_factory.make()

    _ = fetch_activity_stream(runner_one, 100)
    _ = fetch_activity_stream(runner_two, 100)

    assert calls == [100, 100]
