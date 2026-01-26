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


def test_segment_efforts_fetches_tail_after_cached_runs(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """Cached results trigger incremental fetch only for newer efforts."""

    cached_effort = {"id": "cached", "start_date": "2025-01-01T00:00:00Z"}
    cached_record = CaptureRecord(
        signature="abc",
        response=[cached_effort],
        captured_at=datetime.now(timezone.utc),
        source="base",
    )

    def fake_replay(
        runner_arg: Runner,
        url: str,
        params: dict[str, Any],
        **_: Any,
    ) -> CaptureRecord | None:
        return cached_record if params.get("page") == 1 else None

    monkeypatch.setattr(
        segment_efforts,
        "get_cached_list_with_meta",
        fake_replay,
    )

    live_effort = {"id": "live", "start_date": "2025-01-02T00:00:00Z"}
    fetch_calls: list[dict[str, Any]] = []

    def fake_fetch_page_with_retries(**kwargs: Any) -> List[dict[str, Any]]:
        fetch_calls.append(kwargs)
        if kwargs["context_label"] == "segment_efforts_tail":
            return [live_effort]
        raise AssertionError("unexpected live fetch outside tail refresh")

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
        "save_list_to_cache",
        fake_record,
    )

    overlays: list[tuple[dict[str, Any] | None, List[dict[str, Any]] | None]] = []
    monkeypatch.setattr(segment_efforts, "STRAVA_CACHE_OVERWRITE", False)
    monkeypatch.setattr(
        segment_efforts,
        "save_overlay_to_cache",
        lambda *_args, params=None, response=None, **__: overlays.append(
            (params, response)
        ),
    )
    monkeypatch.setattr(
        segment_efforts,
        "save_response_to_cache",
        lambda *_args, params=None, response=None, **__: overlays.append(
            (params, response)
        ),
    )
    monkeypatch.setattr(
        segment_efforts,
        "ensure_runner_token",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        segment_efforts,
        "_cache_mode_reads",
        True,
    )
    monkeypatch.setattr(
        segment_efforts,
        "_cache_mode_saves",
        True,
    )
    monkeypatch.setattr(
        segment_efforts,
        "_cache_mode_offline",
        False,
    )

    api = segment_efforts.SegmentEffortsAPI(
        session=object(),
        limiter=StubLimiter(),
    )
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=2)

    results = api.get_segment_efforts(
        runner,
        segment_id=11,
        start_date=start,
        end_date=end,
    )

    assert results == [live_effort, cached_effort]
    assert fetch_calls and fetch_calls[0]["context_label"] == "segment_efforts_tail"
    assert recorded_pages and recorded_pages[0][0]["page"] == 1
    assert overlays, "expected overlay persistence for enriched cache"


def test_segment_efforts_returns_cached_page_when_no_tail_needed(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """Cached payloads satisfy the full window when no newer data exists."""

    now = datetime.now(timezone.utc)
    cached_record = CaptureRecord(
        signature="def",
        response=[{"id": "cached", "start_date": now.isoformat()}],
        captured_at=now - timedelta(days=1),
        source="base",
    )

    monkeypatch.setattr(
        segment_efforts,
        "get_cached_list_with_meta",
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
        "save_list_to_cache",
        fake_record,
    )
    monkeypatch.setattr(
        segment_efforts,
        "ensure_runner_token",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        segment_efforts,
        "_cache_mode_reads",
        True,
    )
    monkeypatch.setattr(
        segment_efforts,
        "_cache_mode_offline",
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
