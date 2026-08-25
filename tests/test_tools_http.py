"""Tests for the shared tools HTTP helper's 429 retry behaviour."""

from __future__ import annotations

from typing import Any

import pytest
import requests

import strava_competition.tools._http as http_mod


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
        body: Any = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    def json(self) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def _patch_get(
    monkeypatch: pytest.MonkeyPatch, responses: list[_FakeResponse]
) -> list[Any]:
    calls: list[Any] = []

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append(url)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(http_mod.requests, "get", fake_get)
    return calls


def test_retries_429_honouring_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _FakeResponse(429, headers={"Retry-After": "7"}),
        _FakeResponse(200, body={"id": 1}),
    ]
    calls = _patch_get(monkeypatch, responses)
    sleeps: list[float] = []
    monkeypatch.setattr(http_mod.time, "sleep", sleeps.append)

    result = http_mod.http_get("https://example.com/api", "token")

    assert result == {"id": 1}
    assert len(calls) == 2
    assert sleeps == [7.0]


def test_retries_429_with_backoff_when_header_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        _FakeResponse(429),
        _FakeResponse(429, headers={"Retry-After": "not-a-number"}),
        _FakeResponse(200, body=[]),
    ]
    calls = _patch_get(monkeypatch, responses)
    sleeps: list[float] = []
    monkeypatch.setattr(http_mod.time, "sleep", sleeps.append)

    result = http_mod.http_get("https://example.com/api", "token")

    assert result == []
    assert len(calls) == 3
    # Exponential backoff: base * 2**attempt.
    assert sleeps == [2.0, 4.0]


def test_persistent_429_raises_after_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [_FakeResponse(429)]
    calls = _patch_get(monkeypatch, responses)
    monkeypatch.setattr(http_mod.time, "sleep", lambda _s: None)

    with pytest.raises(requests.HTTPError):
        http_mod.http_get("https://example.com/api", "token")

    # Initial attempt + 3 retries.
    assert len(calls) == http_mod._RATE_LIMIT_MAX_RETRIES + 1


def test_retry_after_wait_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _FakeResponse(429, headers={"Retry-After": "3600"}),
        _FakeResponse(200, body={}),
    ]
    _patch_get(monkeypatch, responses)
    sleeps: list[float] = []
    monkeypatch.setattr(http_mod.time, "sleep", sleeps.append)

    http_mod.http_get("https://example.com/api", "token")

    assert sleeps == [http_mod._RATE_LIMIT_MAX_WAIT_SECONDS]


def test_non_429_error_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [_FakeResponse(403)]
    calls = _patch_get(monkeypatch, responses)

    with pytest.raises(requests.HTTPError):
        http_mod.http_get("https://example.com/api", "token")

    assert len(calls) == 1
