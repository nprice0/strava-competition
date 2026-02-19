"""Contract tests for Strava client capture/replay helpers."""

from __future__ import annotations

from typing import Any, List

import pytest
import requests

from strava_competition.models import Runner
from strava_competition.strava_client import pagination


class StubLimiter:
    """Record rate-limiter invocations for assertions."""

    def __init__(self) -> None:
        self.before_calls = 0
        self.after_calls: list[tuple[dict[str, object] | None, int | None]] = []

    def before_request(self) -> None:
        """Track limiter acquisitions."""

        self.before_calls += 1

    def after_response(
        self,
        headers: dict[str, object] | None,
        status_code: int | None,
    ) -> tuple[bool, str]:
        """Track limiter releases and metadata."""

        self.after_calls.append((headers, status_code))
        return False, ""


class FakeResponse:
    """Minimal Response stub implementing the methods under test."""

    def __init__(
        self,
        status_code: int,
        payload: List[dict[str, Any]],
        *,
        content_type: str = "application/json",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = {"Content-Type": content_type}
        self.text = ""
        self.url = "https://example.test/resource"

    def raise_for_status(self) -> None:
        """Mimic requests.Response behaviour for status >= 400."""

        if self.status_code >= 400:
            exc = requests.HTTPError(f"status={self.status_code}")
            exc.response = self
            raise exc

    def json(self) -> List[dict[str, Any]]:
        """Return the scripted JSON payload."""

        return self._payload


class ScriptedSession:
    """Session stub that replays a scripted sequence of actions."""

    def __init__(self, actions: List[Any]) -> None:
        self._actions = list(actions)
        self.calls = 0

    def get(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - stub
        if self.calls >= len(self._actions):
            raise AssertionError("No scripted action remaining")
        action = self._actions[self.calls]
        self.calls += 1
        if isinstance(action, Exception):
            raise action
        return action


@pytest.fixture
def runner() -> Runner:
    """Return a baseline runner for Strava client tests."""

    return Runner(name="Contract Runner", strava_id="42", refresh_token="rt")


def test_fetch_page_with_retries_recovers_from_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """Pagination helper should retry request errors and HTTP 5xx responses."""

    runner.access_token = "token"
    sequence = [
        requests.RequestException("boom"),
        FakeResponse(500, []),
        FakeResponse(200, [{"id": 1}]),
    ]
    session = ScriptedSession(sequence)
    limiter = StubLimiter()
    sleeps: list[float] = []

    monkeypatch.setattr(pagination, "STRAVA_MAX_RETRIES", 5)
    monkeypatch.setattr(pagination, "STRAVA_BACKOFF_MAX_SECONDS", 4)
    monkeypatch.setattr(
        pagination.time,
        "sleep",
        lambda value: sleeps.append(value),
    )

    result = pagination.fetch_page_with_retries(
        runner=runner,
        url="https://example.test/segment_efforts",
        params={"page": 1},
        context_label="segment_efforts",
        page=1,
        session=session,
        limiter=limiter,
    )

    assert result == [{"id": 1}]
    assert session.calls == 3
    assert limiter.before_calls == 3
    assert len(limiter.after_calls) == 3
    assert sleeps == [1.0, 2.0]


def test_fetch_page_with_retries_returns_empty_after_exhausting_errors(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """Persistent transport errors should gracefully yield an empty page."""

    runner.access_token = "token"
    sequence = [
        requests.RequestException("boom"),
        requests.RequestException("boom-again"),
    ]
    session = ScriptedSession(sequence)
    limiter = StubLimiter()

    monkeypatch.setattr(pagination, "STRAVA_MAX_RETRIES", 2)
    monkeypatch.setattr(
        pagination.time,
        "sleep",
        lambda *_: None,
    )

    result = pagination.fetch_page_with_retries(
        runner=runner,
        url="https://example.test/segment_efforts",
        params={"page": 1},
        context_label="segment_efforts",
        page=1,
        session=session,
        limiter=limiter,
    )

    assert result == []
    assert session.calls == 2
    assert limiter.before_calls == 2
    assert limiter.after_calls[-1] == (None, None)
