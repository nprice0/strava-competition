"""Tests for Strava API usage telemetry and the end-of-run summary line."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import pytest

from strava_competition.main import _format_api_usage_summary
from strava_competition.models import Runner
from strava_competition.strava_api import _valid_activity_detail
from strava_competition.strava_client import cache_helpers, telemetry
from strava_competition.strava_client import rate_limiter as rl_mod
from strava_competition.strava_client import resources
from strava_competition.strava_client.resources import ResourceAPI

URL = "https://example.test/activities/123"
PARAMS = {"include_all_efforts": "true"}


@pytest.fixture(autouse=True)
def _clean_counters() -> Any:
    """Isolate every test from telemetry accumulated elsewhere."""
    telemetry.reset()
    yield
    telemetry.reset()


@pytest.fixture
def runner() -> Runner:
    """Return a baseline runner for telemetry tests."""
    return Runner(name="Telemetry Runner", strava_id="9", refresh_token="rt")


# ---------------------------------------------------------------------------
# Counter registry
# ---------------------------------------------------------------------------


def test_increment_and_snapshot() -> None:
    """Increments are reflected in the snapshot; unknown names are created."""
    telemetry.increment(telemetry.LIVE_CALLS)
    telemetry.increment(telemetry.LIVE_CALLS)
    telemetry.increment(telemetry.CACHE_HITS)
    telemetry.increment("custom")
    snap = telemetry.snapshot()
    assert snap[telemetry.LIVE_CALLS] == 2
    assert snap[telemetry.CACHE_HITS] == 1
    assert snap[telemetry.VALIDATION_REFETCHES] == 0
    assert snap["custom"] == 1


def test_reset_restores_known_counters_at_zero() -> None:
    """Reset zeroes the known counters and drops ad-hoc ones."""
    telemetry.increment(telemetry.LIVE_CALLS)
    telemetry.increment("custom")
    telemetry.reset()
    assert telemetry.snapshot() == {
        telemetry.LIVE_CALLS: 0,
        telemetry.CACHE_HITS: 0,
        telemetry.VALIDATION_REFETCHES: 0,
    }


def test_snapshot_returns_copy() -> None:
    """Mutating a snapshot must not affect the registry."""
    snap = telemetry.snapshot()
    snap[telemetry.LIVE_CALLS] = 999
    assert telemetry.snapshot()[telemetry.LIVE_CALLS] == 0


def test_increment_is_thread_safe() -> None:
    """Concurrent increments from several threads must not lose updates."""
    threads_count, per_thread = 8, 500

    def worker() -> None:
        for _ in range(per_thread):
            telemetry.increment(telemetry.LIVE_CALLS)

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert telemetry.snapshot()[telemetry.LIVE_CALLS] == threads_count * per_thread


# ---------------------------------------------------------------------------
# fetch_with_capture / fetch_json wiring
# ---------------------------------------------------------------------------


class FakeResp:
    """Minimal Response stand-in for ResourceAPI.fetch_json."""

    def __init__(self, body: Any, headers: dict[str, str] | None = None) -> None:
        self.status_code = 200
        self.headers = headers or {}
        self._body = body

    def json(self) -> Any:
        return self._body


class FakeSession:
    """Session stub returning a scripted 200 response."""

    def __init__(self, body: Any, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self._headers = headers or {}
        self.calls = 0

    def get(self, *_a: Any, **_kw: Any) -> FakeResp:
        self.calls += 1
        return FakeResp(self._body, self._headers)


def _patch_cache(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cached: Any,
) -> None:
    """Route the resources module cache reads/writes through stubs."""
    monkeypatch.setattr(resources, "get_cached_response", lambda *a, **kw: cached)
    monkeypatch.setattr(resources, "save_response_to_cache", lambda *a, **kw: None)
    monkeypatch.setattr(resources, "save_overlay_to_cache", lambda *a, **kw: None)
    monkeypatch.setattr(resources, "_cache_mode_saves", False)
    monkeypatch.setattr(resources, "_cache_mode_offline", False)
    monkeypatch.setattr(resources, "ensure_runner_token", lambda r: None)


def test_fetch_with_capture_cache_hit_increments_cache_hits(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """Serving a cached payload counts one cache hit and no live calls."""
    cached = {"id": 123, "segment_efforts": []}
    _patch_cache(monkeypatch, cached=cached)

    result = ResourceAPI().fetch_with_capture(runner, URL, PARAMS, "ctx")

    assert result == cached
    snap = telemetry.snapshot()
    assert snap[telemetry.CACHE_HITS] == 1
    assert snap[telemetry.LIVE_CALLS] == 0
    assert snap[telemetry.VALIDATION_REFETCHES] == 0


def test_live_fetch_increments_live_calls(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """A cache miss falls through to HTTP and counts one live call."""
    _patch_cache(monkeypatch, cached=None)
    session = FakeSession({"id": 42})
    runner.access_token = "valid"
    limiter = rl_mod.RateLimiter(max_concurrent=1, jitter_range=(0, 0))
    api = ResourceAPI(session=session, limiter=limiter, timeout=5)  # type: ignore[arg-type]

    result = api.fetch_with_capture(runner, URL, PARAMS, "ctx")

    assert result == {"id": 42}
    snap = telemetry.snapshot()
    assert snap[telemetry.LIVE_CALLS] == 1
    assert snap[telemetry.CACHE_HITS] == 0
    assert snap[telemetry.VALIDATION_REFETCHES] == 0


def test_invalid_cached_refetch_increments_validation_refetches(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """An invalid cached payload counts a refetch plus its live call."""
    _patch_cache(monkeypatch, cached={"id": 123})  # missing segment_efforts
    live = {"id": 123, "segment_efforts": []}
    session = FakeSession(live)
    runner.access_token = "valid"
    limiter = rl_mod.RateLimiter(max_concurrent=1, jitter_range=(0, 0))
    api = ResourceAPI(session=session, limiter=limiter, timeout=5)  # type: ignore[arg-type]

    result = api.fetch_with_capture(
        runner, URL, PARAMS, "ctx", validate=_valid_activity_detail
    )

    assert result == live
    snap = telemetry.snapshot()
    assert snap[telemetry.VALIDATION_REFETCHES] == 1
    assert snap[telemetry.LIVE_CALLS] == 1
    assert snap[telemetry.CACHE_HITS] == 0


# ---------------------------------------------------------------------------
# Cached list reads
# ---------------------------------------------------------------------------


def test_get_cached_list_increments_cache_hits(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """A cached list payload counts one cache hit."""
    monkeypatch.setattr(
        cache_helpers, "get_cached_response", lambda *a, **kw: [{"id": 1}]
    )

    result = cache_helpers.get_cached_list(
        runner,
        URL,
        {"page": 1},
        context_label="activities",
        page=1,
        use_cache=True,
    )

    assert result == [{"id": 1}]
    assert telemetry.snapshot()[telemetry.CACHE_HITS] == 1


def test_get_cached_list_with_meta_increments_cache_hits(
    monkeypatch: pytest.MonkeyPatch, runner: Runner
) -> None:
    """A cached list record counts one cache hit."""
    record = cache_helpers.CaptureRecord(
        signature="sig",
        response=[{"id": 1}],
        captured_at=None,
        source="base",
    )
    monkeypatch.setattr(
        cache_helpers, "get_cached_response_with_meta", lambda *a, **kw: record
    )

    result = cache_helpers.get_cached_list_with_meta(
        runner,
        URL,
        {"page": 1},
        context_label="activities",
        page=1,
        use_cache=True,
    )

    assert result is record
    assert telemetry.snapshot()[telemetry.CACHE_HITS] == 1


# ---------------------------------------------------------------------------
# Rate limiter last-seen usage
# ---------------------------------------------------------------------------


def test_rate_limiter_snapshot_records_last_seen_usage() -> None:
    """The limiter snapshot exposes the last-seen usage/limit headers."""
    limiter = rl_mod.RateLimiter(max_concurrent=1, jitter_range=(0, 0))
    headers = {
        "X-RateLimit-Usage": "143,1002",
        "X-RateLimit-Limit": "200,2000",
    }
    limiter.after_response(headers, 200)
    snap = limiter.snapshot()
    assert snap["short_used"] == 143
    assert snap["short_limit"] == 200
    assert snap["daily_used"] == 1002
    assert snap["daily_limit"] == 2000


def test_rate_limiter_snapshot_defaults_to_none() -> None:
    """A limiter that never saw headers reports None usage."""
    limiter = rl_mod.RateLimiter(max_concurrent=1, jitter_range=(0, 0))
    snap = limiter.snapshot()
    assert snap["short_used"] is None
    assert snap["short_limit"] is None
    assert snap["daily_used"] is None
    assert snap["daily_limit"] is None


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------


def _limiter_snapshot(
    **overrides: Optional[int],
) -> Dict[str, float | int | None]:
    base: Dict[str, float | int | None] = {
        "max_allowed": 5,
        "in_flight": 0,
        "throttle_until": 0.0,
        "short_used": None,
        "short_limit": None,
        "daily_used": None,
        "daily_limit": None,
    }
    base.update(overrides)
    return base


def test_summary_with_rate_limit_data() -> None:
    """The summary renders live/cached counts and both rate-limit windows."""
    counters = {"live_calls": 143, "cache_hits": 892, "validation_refetches": 3}
    snap = _limiter_snapshot(
        short_used=143, short_limit=200, daily_used=143, daily_limit=2000
    )
    assert _format_api_usage_summary(counters, snap) == (
        "API usage: live=143 cached=892 validation_refetches=3 | "
        "rate limit: 143/200 (15min), 143/2000 (daily)"
    )


def test_summary_without_rate_limit_data() -> None:
    """A fully cached run renders rate limit as n/a, not zeros."""
    counters = {"live_calls": 0, "cache_hits": 500, "validation_refetches": 0}
    assert _format_api_usage_summary(counters, _limiter_snapshot()) == (
        "API usage: live=0 cached=500 validation_refetches=0 | rate limit: n/a"
    )


def test_summary_with_short_window_only() -> None:
    """Missing daily figures omit the daily clause rather than showing zeros."""
    counters = {"live_calls": 7, "cache_hits": 0, "validation_refetches": 0}
    snap = _limiter_snapshot(short_used=7, short_limit=200)
    assert _format_api_usage_summary(counters, snap) == (
        "API usage: live=7 cached=0 validation_refetches=0 | rate limit: 7/200 (15min)"
    )
