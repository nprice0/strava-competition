"""Contract tests for Strava client capture/replay helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, List

import pytest
import requests

from strava_competition.api_capture import CaptureRecord
from strava_competition.models import Runner
from strava_competition.strava_client import pagination
from strava_competition.strava_client import segment_efforts


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
    ) -> None:
        """Track limiter releases and metadata."""

        self.after_calls.append((headers, status_code))


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


def test_segment_efforts_switches_to_live_when_cache_stale(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """Stale replay payloads should trigger live fetch + capture."""

    now = datetime.now(timezone.utc)
    stale_record = CaptureRecord(
        signature="abc",
        response=[{"id": "cached"}],
        captured_at=now - timedelta(days=8),
        source="base",
    )

    replay_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_replay(*args: Any, **kwargs: Any) -> CaptureRecord | None:
        replay_calls.append((args, kwargs))
        return stale_record

    monkeypatch.setattr(
        segment_efforts,
        "replay_list_response_with_meta",
        fake_replay,
    )

    live_payload = [{"id": "live"}]
    fetch_calls: list[dict[str, Any]] = []

    def fake_fetch_page_with_retries(**kwargs: Any) -> List[dict[str, Any]]:
        fetch_calls.append(kwargs)
        return live_payload

    monkeypatch.setattr(
        segment_efforts,
        "fetch_page_with_retries",
        fake_fetch_page_with_retries,
    )

    recorded_pages: list[tuple[dict[str, Any], List[dict[str, Any]]]] = []

    def fake_record(
        runner_arg: Runner,
        url: str,
        params: dict[str, Any],
        data: List[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        recorded_pages.append((params, data))

    monkeypatch.setattr(
        segment_efforts,
        "record_list_response",
        fake_record,
    )
    monkeypatch.setattr(
        segment_efforts,
        "ensure_runner_token",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        segment_efforts,
        "STRAVA_API_REPLAY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        segment_efforts,
        "STRAVA_API_CAPTURE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        segment_efforts,
        "STRAVA_OFFLINE_MODE",
        False,
    )
    monkeypatch.setattr(
        segment_efforts,
        "REPLAY_CACHE_TTL_DAYS",
        7,
    )

    api = segment_efforts.SegmentEffortsAPI(
        session=object(),
        limiter=StubLimiter(),
    )
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    results = api.get_segment_efforts(
        runner,
        segment_id=11,
        start_date=start,
        end_date=end,
    )

    assert results == live_payload
    assert fetch_calls, "expected live pagination after stale replay"
    assert recorded_pages and recorded_pages[0][1] == live_payload
    assert replay_calls and len(replay_calls) == 1


def test_segment_efforts_returns_cached_page_when_ttl_valid(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """Fresh replay payloads should bypass live pagination entirely."""

    now = datetime.now(timezone.utc)
    cached_record = CaptureRecord(
        signature="def",
        response=[{"id": "cached"}],
        captured_at=now - timedelta(days=1),
        source="base",
    )

    monkeypatch.setattr(
        segment_efforts,
        "replay_list_response_with_meta",
        lambda *_, **__: cached_record,
    )

    def fail_fetch(**_: Any) -> List[dict[str, Any]]:
        raise AssertionError("Live fetch should not run when cache is fresh")

    monkeypatch.setattr(
        segment_efforts,
        "fetch_page_with_retries",
        fail_fetch,
    )

    recorded_pages: list[tuple[dict[str, Any], List[dict[str, Any]]]] = []

    def fake_record(
        runner_arg: Runner,
        url: str,
        params: dict[str, Any],
        data: List[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        recorded_pages.append((params, data))

    monkeypatch.setattr(
        segment_efforts,
        "record_list_response",
        fake_record,
    )
    monkeypatch.setattr(
        segment_efforts,
        "ensure_runner_token",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        segment_efforts,
        "STRAVA_API_REPLAY_ENABLED",
        True,
    )
    monkeypatch.setattr(
        segment_efforts,
        "STRAVA_OFFLINE_MODE",
        False,
    )

    api = segment_efforts.SegmentEffortsAPI(
        session=object(),
        limiter=StubLimiter(),
    )
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    results = api.get_segment_efforts(
        runner,
        segment_id=22,
        start_date=start,
        end_date=end,
    )

    assert results == cached_record.response
    assert not recorded_pages


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
