"""Tests for offline-mode enforcement in ``strava_client.cache_helpers``."""

from __future__ import annotations

from typing import Any

import pytest

from strava_competition.errors import StravaAPIError
from strava_competition.models import Runner
from strava_competition.strava_client import cache_helpers


@pytest.fixture
def runner() -> Runner:
    """Return a baseline runner for cache-helper tests."""

    return Runner(name="Cache Runner", strava_id="11", refresh_token="rt")


URL = "https://www.strava.com/api/v3/athlete/activities"
PARAMS = {"page": 1}


def test_get_cached_list_type_mismatch_raises_offline(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """Offline mode must raise on a wrong-typed cached payload, not fall through."""

    monkeypatch.setattr(
        cache_helpers,
        "get_cached_response",
        lambda *args, **kwargs: {"unexpected": "dict"},
    )

    with pytest.raises(StravaAPIError, match="cache miss"):
        cache_helpers.get_cached_list(
            runner,
            URL,
            PARAMS,
            context_label="activities",
            page=1,
            use_cache=True,
            require_cache=True,
        )


def test_get_cached_list_with_meta_type_mismatch_raises_offline(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """Offline mode must raise on a wrong-typed cached record payload."""

    record = cache_helpers.CaptureRecord(
        signature="sig",
        response={"unexpected": "dict"},
        captured_at=None,
        source="base",
    )
    monkeypatch.setattr(
        cache_helpers,
        "get_cached_response_with_meta",
        lambda *args, **kwargs: record,
    )

    with pytest.raises(StravaAPIError, match="cache miss"):
        cache_helpers.get_cached_list_with_meta(
            runner,
            URL,
            PARAMS,
            context_label="activities",
            page=1,
            use_cache=True,
            require_cache=True,
        )


def test_get_cached_list_type_mismatch_returns_none_when_online(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """Outside offline mode a wrong-typed payload degrades to a cache miss."""

    monkeypatch.setattr(
        cache_helpers,
        "get_cached_response",
        lambda *args, **kwargs: {"unexpected": "dict"},
    )

    result: Any = cache_helpers.get_cached_list(
        runner,
        URL,
        PARAMS,
        context_label="activities",
        page=1,
        use_cache=True,
        require_cache=False,
    )

    assert result is None
