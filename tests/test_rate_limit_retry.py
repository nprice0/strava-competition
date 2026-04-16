"""Tests for rate-limit retry behaviour in resources.py and segment_service.py."""

from __future__ import annotations

import threading
import warnings
from datetime import datetime, timedelta, timezone
from types import MethodType
from typing import Any

import pytest

from strava_competition.errors import StravaRateLimitError
from strava_competition.models import Runner, Segment, SegmentResult
from strava_competition.services.segment_service import SegmentService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runner(name: str, runner_id: int, team: str = "A") -> Runner:
    return Runner(
        name=name,
        strava_id=str(runner_id),
        refresh_token="rt",
        segment_team=team,
    )


def _segment(name: str = "Test Segment") -> Segment:
    now = datetime.now(timezone.utc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Segment(
            id=123,
            name=name,
            start_date=now - timedelta(days=1),
            end_date=now,
            default_time_seconds=999.0,
        )


# ---------------------------------------------------------------------------
# ResourceAPI.fetch_json tests
# ---------------------------------------------------------------------------


class TestResourceAPI429Retries:
    """Verify fetch_json uses a dedicated 429 retry budget with escalating backoff."""

    def test_recovers_after_transient_429(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transient 429 followed by success should return data."""
        from strava_competition.strava_client.resources import ResourceAPI
        from strava_competition.strava_client import rate_limiter as rl_mod

        # Minimal rate-limit headers
        HEADERS_429: dict[str, str] = {
            "X-RateLimit-Usage": "300,500",
            "X-RateLimit-Limit": "300,3000",
            "X-ReadRateLimit-Usage": "300,500",
            "X-ReadRateLimit-Limit": "300,3000",
        }
        HEADERS_OK: dict[str, str] = {
            "X-RateLimit-Usage": "100,501",
            "X-RateLimit-Limit": "300,3000",
            "X-ReadRateLimit-Usage": "100,501",
            "X-ReadRateLimit-Limit": "300,3000",
        }

        call_count = 0

        class FakeResp:
            def __init__(self, status: int, headers: dict[str, str], body: Any):
                self.status_code = status
                self.headers = headers
                self._body = body

            def json(self) -> Any:
                return self._body

        class FakeSession:
            def get(self, *_a: Any, **_kw: Any) -> FakeResp:
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return FakeResp(
                        429, HEADERS_429, {"message": "Rate Limit Exceeded"}
                    )
                return FakeResp(200, HEADERS_OK, {"id": 42})

        runner = _runner("Test", 1)
        runner.access_token = "valid"

        # Patch out token refresh
        monkeypatch.setattr(
            "strava_competition.strava_client.resources.ensure_runner_token",
            lambda r: None,
        )
        # Use very short throttle for test speed — must patch in both modules
        monkeypatch.setattr(
            "strava_competition.strava_client.resources.RATE_LIMIT_THROTTLE_SECONDS", 0
        )
        monkeypatch.setattr(
            "strava_competition.strava_client.rate_limiter.RATE_LIMIT_THROTTLE_SECONDS",
            0,
        )
        monkeypatch.setattr(
            "strava_competition.strava_client.resources.RATE_LIMIT_429_BACKOFF_MAX_SECONDS",
            0.01,
        )

        limiter = rl_mod.RateLimiter(max_concurrent=1, jitter_range=(0, 0))
        api = ResourceAPI(session=FakeSession(), limiter=limiter, timeout=5)  # type: ignore[arg-type]

        result = api.fetch_json(runner, "https://example.com/api", None, "test_ctx")
        assert result == {"id": 42}
        assert call_count == 3

    def test_raises_rate_limit_error_after_max_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Persistent 429s should raise StravaRateLimitError, not generic StravaAPIError."""
        from strava_competition.strava_client.resources import ResourceAPI
        from strava_competition.strava_client import rate_limiter as rl_mod

        HEADERS_429: dict[str, str] = {
            "X-RateLimit-Usage": "300,500",
            "X-RateLimit-Limit": "300,3000",
            "X-ReadRateLimit-Usage": "300,500",
            "X-ReadRateLimit-Limit": "300,3000",
        }

        class FakeResp:
            def __init__(self) -> None:
                self.status_code = 429
                self.headers = HEADERS_429

            def json(self) -> Any:
                return {"message": "Rate Limit Exceeded"}

        class FakeSession:
            def get(self, *_a: Any, **_kw: Any) -> FakeResp:
                return FakeResp()

        runner = _runner("Test", 1)
        runner.access_token = "valid"

        monkeypatch.setattr(
            "strava_competition.strava_client.resources.ensure_runner_token",
            lambda r: None,
        )
        monkeypatch.setattr(
            "strava_competition.strava_client.resources.RATE_LIMIT_THROTTLE_SECONDS", 0
        )
        monkeypatch.setattr(
            "strava_competition.strava_client.rate_limiter.RATE_LIMIT_THROTTLE_SECONDS",
            0,
        )
        monkeypatch.setattr(
            "strava_competition.strava_client.resources.RATE_LIMIT_429_MAX_RETRIES", 3
        )
        monkeypatch.setattr(
            "strava_competition.strava_client.resources.RATE_LIMIT_429_BACKOFF_MAX_SECONDS",
            0.01,
        )

        limiter = rl_mod.RateLimiter(max_concurrent=1, jitter_range=(0, 0))
        api = ResourceAPI(session=FakeSession(), limiter=limiter, timeout=5)  # type: ignore[arg-type]

        with pytest.raises(
            StravaRateLimitError, match="rate limited.*429.*after 3 retries"
        ):
            api.fetch_json(runner, "https://example.com/api", None, "test_ctx")


# ---------------------------------------------------------------------------
# SegmentService runner-level retry tests
# ---------------------------------------------------------------------------


class TestSegmentServiceRateLimitRetry:
    """Verify that rate-limited runners are retried instead of silently penalised."""

    def test_rate_limited_runner_retried_and_succeeds(self) -> None:
        """A runner that fails with StravaRateLimitError on the first round
        should be retried and eventually get their real result."""
        service = SegmentService(max_workers=1)
        seg = _segment()
        alice = _runner("Alice", 1)
        bob = _runner("Bob", 2)

        call_counts: dict[str, int] = {}

        def fake_scan(
            self: SegmentService,
            runner: Runner,
            segment: Segment,
            cancel_event: threading.Event | None = None,
        ) -> SegmentResult | None:
            count = call_counts.get(runner.name, 0) + 1
            call_counts[runner.name] = count

            if runner.name == "Bob" and count == 1:
                raise StravaRateLimitError("429 after max retries for runner Bob")

            return SegmentResult(
                runner=runner.name,
                team=runner.segment_team or "",
                segment=segment.name,
                attempts=1,
                fastest_time=120.0 if runner.name == "Alice" else 130.0,
                fastest_date=None,
                source="activity_scan",
            )

        service._result_from_activity_scan = MethodType(fake_scan, service)  # type: ignore[method-assign]

        # Patch cooldown to 0 for test speed
        import strava_competition.services.segment_service as svc_mod

        orig_cooldown = svc_mod._RATE_LIMIT_RUNNER_COOLDOWN
        svc_mod._RATE_LIMIT_RUNNER_COOLDOWN = 0
        try:
            results = service._process_segment(seg, [alice, bob], 1, 1, None, None)
        finally:
            svc_mod._RATE_LIMIT_RUNNER_COOLDOWN = orig_cooldown

        # Both runners should have real results
        all_results = [r for team in results.values() for r in team]
        names = {r.runner for r in all_results}
        assert "Alice" in names
        assert "Bob" in names
        bob_result = next(r for r in all_results if r.runner == "Bob")
        assert bob_result.fastest_time == 130.0
        assert bob_result.source == "activity_scan"

        # Bob should have been called twice (first fail, then retry)
        assert call_counts["Bob"] == 2
        # Alice should have been called once
        assert call_counts["Alice"] == 1

    def test_persistent_rate_limit_excluded_from_defaults(self) -> None:
        """A runner that stays rate-limited across all retry rounds
        should NOT get default_time — we never confirmed they have no effort."""
        service = SegmentService(max_workers=1)
        seg = _segment()
        runner = _runner("LimitedRunner", 1)

        def always_rate_limited(
            self: SegmentService,
            r: Runner,
            s: Segment,
            cancel_event: threading.Event | None = None,
        ) -> SegmentResult | None:
            raise StravaRateLimitError("429 persistent")

        service._result_from_activity_scan = MethodType(always_rate_limited, service)  # type: ignore[method-assign]

        import strava_competition.services.segment_service as svc_mod

        orig_cooldown = svc_mod._RATE_LIMIT_RUNNER_COOLDOWN
        svc_mod._RATE_LIMIT_RUNNER_COOLDOWN = 0
        try:
            results = service._process_segment(seg, [runner], 1, 1, None, None)
        finally:
            svc_mod._RATE_LIMIT_RUNNER_COOLDOWN = orig_cooldown

        # Runner should have NO result — not even default_time.
        # We never successfully scanned them, so we can't assume they
        # have no effort.
        all_results = [r for team in results.values() for r in team]
        assert len(all_results) == 0

    def test_non_rate_limit_api_error_not_retried(self) -> None:
        """A runner that fails with a generic StravaAPIError (e.g. 403)
        should NOT be retried — only rate-limit errors trigger retry."""

        service = SegmentService(max_workers=1)
        seg = _segment()
        runner = _runner("Forbidden", 1)

        call_count = 0

        def permission_denied(
            self: SegmentService,
            r: Runner,
            s: Segment,
            cancel_event: threading.Event | None = None,
        ) -> SegmentResult | None:
            nonlocal call_count
            call_count += 1
            # _result_from_activity_scan catches non-rate-limit StravaAPIError
            # and returns None. So simulate that:
            return None

        service._result_from_activity_scan = MethodType(permission_denied, service)  # type: ignore[method-assign]

        import strava_competition.services.segment_service as svc_mod

        orig_cooldown = svc_mod._RATE_LIMIT_RUNNER_COOLDOWN
        svc_mod._RATE_LIMIT_RUNNER_COOLDOWN = 0
        try:
            service._process_segment(seg, [runner], 1, 1, None, None)
        finally:
            svc_mod._RATE_LIMIT_RUNNER_COOLDOWN = orig_cooldown

        # Should only be called once — not retried
        assert call_count == 1

    def test_scanned_runner_with_no_effort_gets_default_time(self) -> None:
        """A runner successfully scanned but with no matching effort should
        get the default_time penalty — this is the legitimate 'no effort' case."""
        service = SegmentService(max_workers=1)
        seg = _segment()
        runner = _runner("NoEffort", 1)

        def no_effort(
            self: SegmentService,
            r: Runner,
            s: Segment,
            cancel_event: threading.Event | None = None,
        ) -> SegmentResult | None:
            return None  # successfully scanned, found nothing

        service._result_from_activity_scan = MethodType(no_effort, service)  # type: ignore[method-assign]

        results = service._process_segment(seg, [runner], 1, 1, None, None)

        all_results = [r for team in results.values() for r in team]
        assert len(all_results) == 1
        assert all_results[0].runner == "NoEffort"
        assert all_results[0].fastest_time == 999.0
        assert all_results[0].source == "default_time"

    def test_mixed_rate_limited_and_scanned_runners(self) -> None:
        """When some runners are rate-limited and others scan OK, only the
        scanned runners with no effort get default_time."""
        service = SegmentService(max_workers=1)
        seg = _segment()
        alice = _runner("Alice", 1)  # scans OK, has an effort
        bob = _runner("Bob", 2)  # rate-limited
        carol = _runner("Carol", 3)  # scans OK, no effort → default_time

        def fake_scan(
            self: SegmentService,
            runner: Runner,
            segment: Segment,
            cancel_event: threading.Event | None = None,
        ) -> SegmentResult | None:
            if runner.name == "Alice":
                return SegmentResult(
                    runner="Alice",
                    team="A",
                    segment=segment.name,
                    attempts=1,
                    fastest_time=120.0,
                    fastest_date=None,
                    source="activity_scan",
                )
            if runner.name == "Bob":
                raise StravaRateLimitError("429 for Bob")
            return None  # Carol: no effort found

        service._result_from_activity_scan = MethodType(fake_scan, service)  # type: ignore[method-assign]

        import strava_competition.services.segment_service as svc_mod

        orig_cooldown = svc_mod._RATE_LIMIT_RUNNER_COOLDOWN
        svc_mod._RATE_LIMIT_RUNNER_COOLDOWN = 0
        try:
            results = service._process_segment(
                seg, [alice, bob, carol], 1, 1, None, None
            )
        finally:
            svc_mod._RATE_LIMIT_RUNNER_COOLDOWN = orig_cooldown

        all_results = [r for team in results.values() for r in team]
        result_map = {r.runner: r for r in all_results}

        # Alice: real effort
        assert result_map["Alice"].fastest_time == 120.0
        assert result_map["Alice"].source == "activity_scan"

        # Carol: no effort → default_time
        assert result_map["Carol"].fastest_time == 999.0
        assert result_map["Carol"].source == "default_time"

        # Bob: rate-limited → excluded entirely, no result
        assert "Bob" not in result_map
