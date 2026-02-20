from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from strava_competition.api_capture import CaptureRecord
from strava_competition.models import Runner
from strava_competition.strava_client.activities import (
    ACTIVITY_PAGE_SIZE,
    CachedPage,
    _maybe_refresh_cache_tail,
)


@pytest.fixture()
def runner() -> Runner:
    return Runner(
        name="Test Runner",
        strava_id="123",
        refresh_token="token",
        segment_team="Sheriffs",
    )


def _make_cached_page(age: timedelta) -> CachedPage:
    captured_at = datetime.now(timezone.utc) - age
    record = CaptureRecord(
        signature="sig",
        response=[],
        captured_at=captured_at,
        source="base",
    )
    return CachedPage(
        params={"page": 1},
        data=[],
        record=record,
    )


def test_empty_activity_cache_triggers_refresh_when_stale(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    cached_pages = [_make_cached_page(timedelta(days=1))]
    base_params = {"after": 0, "before": 0, "per_page": ACTIVITY_PAGE_SIZE}
    start = datetime(2025, 12, 9, tzinfo=timezone.utc)
    end = datetime(2025, 12, 11, tzinfo=timezone.utc)
    fetched = [[{"id": 999, "start_date_local": "2025-12-10T09:34:12Z"}]]

    monkeypatch.setattr(
        "strava_competition.strava_client.activities.CACHE_EMPTY_REFRESH_SECONDS",
        60,
        raising=False,
    )
    monkeypatch.setattr(
        "strava_competition.strava_client.activities._fetch_tail_pages",
        lambda *_args, **_kwargs: fetched,
    )
    monkeypatch.setattr(
        "strava_competition.strava_client.activities._persist_enriched_pages",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "strava_competition.strava_client.activities._runner_refresh_deadline",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "strava_competition.strava_client.activities._mark_runner_refreshed",
        lambda *_args, **_kwargs: None,
    )

    merged, refreshed = _maybe_refresh_cache_tail(
        runner,
        "https://www.strava.com/api/v3/athlete/activities",
        base_params,
        cached_pages,
        [],
        session=None,  # type: ignore[arg-type]
        limiter=None,  # type: ignore[arg-type]
        start_date=start,
        end_date=end,
    )

    assert refreshed is True
    assert merged == fetched[0]


def test_empty_activity_cache_skips_refresh_when_recent(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    cached_pages = [_make_cached_page(timedelta(seconds=5))]
    base_params = {"after": 0, "before": 0, "per_page": ACTIVITY_PAGE_SIZE}
    start = datetime(2025, 12, 9, tzinfo=timezone.utc)
    end = datetime(2025, 12, 11, tzinfo=timezone.utc)

    monkeypatch.setattr(
        "strava_competition.strava_client.activities.CACHE_EMPTY_REFRESH_SECONDS",
        3600,
        raising=False,
    )

    def _fail_fetch(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("should not fetch tail for fresh cache")

    monkeypatch.setattr(
        "strava_competition.strava_client.activities._fetch_tail_pages",
        _fail_fetch,
    )

    merged, refreshed = _maybe_refresh_cache_tail(
        runner,
        "https://www.strava.com/api/v3/athlete/activities",
        base_params,
        cached_pages,
        [],
        session=None,  # type: ignore[arg-type]
        limiter=None,  # type: ignore[arg-type]
        start_date=start,
        end_date=end,
    )

    assert refreshed is False
    assert merged == []
