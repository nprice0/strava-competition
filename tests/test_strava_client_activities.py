from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from strava_competition.api_capture import CaptureRecord
from strava_competition.errors import StravaAPIError
from strava_competition.models import Runner
from strava_competition.strava_client.activities import (
    ACTIVITY_PAGE_SIZE,
    CachedPage,
    _maybe_refresh_cache_tail,
)


@pytest.fixture
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


URL = "https://www.strava.com/api/v3/athlete/activities"


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _activity(act_id: int, start: datetime) -> dict[str, Any]:
    return {"id": act_id, "type": "Run", "start_date": _iso(start)}


def test_tail_refresh_failure_degrades_to_cached_data(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """A failing tail refresh must serve cached data, not discard it."""

    import strava_competition.strava_client.activities as acts

    acts._runner_tail_refreshed_until.clear()
    now = datetime.now(timezone.utc)
    cached_act = _activity(1, now - timedelta(days=2))
    cached_pages = [
        CachedPage(params={"page": 1}, data=[cached_act], record=None),
    ]
    base_params = {
        "after": int((now - timedelta(days=10)).timestamp()),
        "before": int(now.timestamp()),
        "per_page": ACTIVITY_PAGE_SIZE,
    }

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise StravaAPIError("tail fetch failed")

    monkeypatch.setattr(acts, "_fetch_tail_pages", _boom)

    merged, refreshed = _maybe_refresh_cache_tail(
        runner,
        URL,
        base_params,
        cached_pages,
        [cached_act],
        session=None,  # type: ignore[arg-type]
        limiter=None,  # type: ignore[arg-type]
        start_date=now - timedelta(days=10),
        end_date=now,
    )

    assert refreshed is False
    assert merged == [cached_act]


def test_tail_refresh_suppression_is_scoped_per_window(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """Refreshing window B must not suppress the refresh for window A."""

    import strava_competition.strava_client.activities as acts

    acts._runner_tail_refreshed_until.clear()
    now = datetime.now(timezone.utc)
    cached_act = _activity(1, now - timedelta(days=2))
    tail_act = _activity(2, now - timedelta(hours=1))
    tail_fetches: list[int] = []

    def _fake_tail(*_args: Any, **_kwargs: Any) -> list[list[dict[str, Any]]]:
        tail_fetches.append(1)
        return [[tail_act]]

    monkeypatch.setattr(acts, "_fetch_tail_pages", _fake_tail)
    monkeypatch.setattr(acts, "_persist_enriched_pages", lambda *a, **k: True)

    def _run_window(days_back: int) -> tuple[list[dict[str, Any]], bool]:
        start = now - timedelta(days=days_back)
        base_params = {
            "after": int(start.timestamp()),
            "before": int(now.timestamp()),
            "per_page": ACTIVITY_PAGE_SIZE,
        }
        cached_pages = [
            CachedPage(params={"page": 1}, data=[cached_act], record=None),
        ]
        return _maybe_refresh_cache_tail(
            runner,
            URL,
            base_params,
            cached_pages,
            [cached_act],
            session=None,  # type: ignore[arg-type]
            limiter=None,  # type: ignore[arg-type]
            start_date=start,
            end_date=now,
        )

    _, refreshed_first = _run_window(10)
    assert refreshed_first is True
    assert len(tail_fetches) == 1

    # A different window for the same runner must still tail-refresh.
    _, refreshed_second = _run_window(5)
    assert refreshed_second is True
    assert len(tail_fetches) == 2

    # The same window is suppressed until the TTL entry expires.
    _, refreshed_repeat = _run_window(10)
    assert refreshed_repeat is False
    assert len(tail_fetches) == 2


def test_get_activities_dedupes_across_pages(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """Overlapping pages (e.g. after a partial persist crash) count once."""

    import strava_competition.strava_client.activities as acts

    window_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    window_end = datetime(2025, 3, 1, tzinfo=timezone.utc)
    base = datetime(2025, 1, 2, tzinfo=timezone.utc)
    full_page = [
        _activity(idx, base + timedelta(hours=idx)) for idx in range(ACTIVITY_PAGE_SIZE)
    ]
    duplicate = dict(full_page[-1])
    pages = {
        1: full_page,
        2: [duplicate, _activity(9999, base + timedelta(days=30))],
    }

    monkeypatch.setattr(acts, "_cache_mode_reads", False)
    monkeypatch.setattr(acts, "_cache_mode_saves", False)
    monkeypatch.setattr(acts, "ensure_runner_token", lambda r: None)
    monkeypatch.setattr(
        acts,
        "fetch_page_with_retries",
        lambda **kwargs: pages[kwargs["params"]["page"]],
    )

    api = acts.ActivitiesAPI()
    result = api.get_activities(runner, window_start, window_end)

    assert result is not None
    ids = [act["id"] for act in result]
    assert len(ids) == len(set(ids))
    assert ids.count(full_page[-1]["id"]) == 1
    assert len(ids) == ACTIVITY_PAGE_SIZE + 1


def test_get_activities_returns_none_and_caches_nothing_on_failure(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """A terminal page failure yields None and never poisons the cache."""

    import strava_competition.strava_client.activities as acts

    saves: list[Any] = []

    def _boom(**_kwargs: Any) -> None:
        raise StravaAPIError("page fetch failed")

    monkeypatch.setattr(acts, "_cache_mode_reads", False)
    monkeypatch.setattr(acts, "_cache_mode_saves", True)
    monkeypatch.setattr(acts, "ensure_runner_token", lambda r: None)
    monkeypatch.setattr(acts, "fetch_page_with_retries", _boom)
    monkeypatch.setattr(acts, "save_list_to_cache", lambda *a, **k: saves.append(a))

    api = acts.ActivitiesAPI()
    result = api.get_activities(
        runner,
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 2, 1, tzinfo=timezone.utc),
    )

    assert result is None
    assert saves == []
