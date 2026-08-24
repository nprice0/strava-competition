"""Tests for payload validation in ``ResourceAPI.fetch_with_capture``."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import pytest

from strava_competition.errors import StravaAPIError
from strava_competition.models import Runner
from strava_competition.strava_api import _valid_activity_detail
from strava_competition.strava_client import resources
from strava_competition.strava_client.resources import ResourceAPI

URL = "https://example.test/activities/123"
PARAMS = {"include_all_efforts": "true"}


@pytest.fixture
def runner() -> Runner:
    """Return a baseline runner for validation tests."""
    return Runner(name="Validator Runner", strava_id="7", refresh_token="rt")


class CaptureSpy:
    """Record cache interactions and script the cached/live payloads."""

    def __init__(
        self,
        *,
        cached: Any = None,
        live: Any = None,
    ) -> None:
        self.cached = cached
        self.live = live
        self.live_calls = 0
        self.saved_base: list[Any] = []
        self.saved_overlay: list[Any] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch the resources module to route cache traffic through the spy."""
        monkeypatch.setattr(
            resources,
            "get_cached_response",
            lambda *args, **kwargs: self.cached,
        )
        monkeypatch.setattr(
            resources,
            "save_response_to_cache",
            lambda *args, response, **kwargs: self.saved_base.append(response),
        )
        monkeypatch.setattr(
            resources,
            "save_overlay_to_cache",
            lambda *args, response, **kwargs: self.saved_overlay.append(response),
        )
        monkeypatch.setattr(resources, "_cache_mode_saves", True)
        monkeypatch.setattr(resources, "_cache_mode_offline", False)

        spy = self

        def fake_fetch_json(
            api: ResourceAPI,
            runner: Runner,
            url: str,
            params: Optional[Dict[str, Any]],
            context: str,
        ) -> Any:
            spy.live_calls += 1
            return spy.live

        monkeypatch.setattr(ResourceAPI, "fetch_json", fake_fetch_json, raising=False)


def test_invalid_cached_payload_is_refetched_and_overlaid(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """A cached payload missing segment_efforts triggers a refetch + overlay."""
    live = {"id": 123, "segment_efforts": []}
    spy = CaptureSpy(cached={"id": 123}, live=live)
    spy.install(monkeypatch)

    result = ResourceAPI().fetch_with_capture(
        runner,
        URL,
        PARAMS,
        "activity_detail",
        validate=_valid_activity_detail,
    )

    assert result == live
    assert spy.live_calls == 1
    assert spy.saved_overlay == [live]
    assert spy.saved_base == []


def test_valid_cached_payload_with_empty_efforts_served_from_cache(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """An empty segment_efforts list is valid and served without a live call."""
    cached = {"id": 123, "segment_efforts": []}
    spy = CaptureSpy(cached=cached)
    spy.install(monkeypatch)

    result = ResourceAPI().fetch_with_capture(
        runner,
        URL,
        PARAMS,
        "activity_detail",
        validate=_valid_activity_detail,
    )

    assert result == cached
    assert spy.live_calls == 0
    assert spy.saved_base == []
    assert spy.saved_overlay == []


def test_invalid_live_payload_returned_but_not_cached(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """A live payload failing validation is returned but never persisted."""
    live = {"id": 123}  # no segment_efforts key
    spy = CaptureSpy(cached=None, live=live)
    spy.install(monkeypatch)

    result = ResourceAPI().fetch_with_capture(
        runner,
        URL,
        PARAMS,
        "activity_detail",
        validate=_valid_activity_detail,
    )

    assert result == live
    assert spy.live_calls == 1
    assert spy.saved_base == []
    assert spy.saved_overlay == []


def test_offline_mode_with_invalid_cached_payload_raises(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """Offline mode surfaces an invalid cached payload as a StravaAPIError."""
    spy = CaptureSpy(cached={"id": 123, "segment_efforts": None})
    spy.install(monkeypatch)
    monkeypatch.setattr(resources, "_cache_mode_offline", True)
    api = ResourceAPI()

    with pytest.raises(StravaAPIError) as excinfo:
        api.fetch_with_capture(
            runner,
            URL,
            PARAMS,
            "activity_detail",
            validate=_valid_activity_detail,
        )

    assert "invalid cached payload" in str(excinfo.value)
    assert spy.live_calls == 0


def test_offline_mode_cache_miss_still_reports_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """Offline cache misses keep the original error wording."""
    spy = CaptureSpy(cached=None)
    spy.install(monkeypatch)
    monkeypatch.setattr(resources, "_cache_mode_offline", True)
    api = ResourceAPI()

    with pytest.raises(StravaAPIError) as excinfo:
        api.fetch_with_capture(
            runner,
            URL,
            PARAMS,
            "activity_detail",
            validate=_valid_activity_detail,
        )

    assert "cache miss" in str(excinfo.value)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"segment_efforts": []}, True),
        ({"segment_efforts": [{"id": 1}]}, True),
        ({"segment_efforts": None}, False),
        ({}, False),
        ([], False),
        (None, False),
        ("text", False),
    ],
)
def test_valid_activity_detail(payload: Any, expected: bool) -> None:
    """Validator accepts dicts with a list segment_efforts, rejects the rest."""
    assert _valid_activity_detail(payload) is expected


def _strip_segment_efforts(value: Any) -> Any:
    """Simulate a redaction config that removes a structurally required key."""
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if k != "segment_efforts"}
    return value


@pytest.mark.parametrize("cached", [None, {"id": 123}])
def test_redaction_breaking_validation_skips_persist(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
    caplog: pytest.LogCaptureFixture,
    cached: Any,
) -> None:
    """M1: a redaction that breaks validation blocks both save branches."""
    live = {"id": 123, "segment_efforts": []}
    spy = CaptureSpy(cached=cached, live=live)
    spy.install(monkeypatch)
    monkeypatch.setattr(resources, "_redact_payload", _strip_segment_efforts)

    with caplog.at_level(logging.ERROR, logger=resources.__name__):
        result = ResourceAPI().fetch_with_capture(
            runner,
            URL,
            PARAMS,
            "activity_detail",
            validate=_valid_activity_detail,
        )

    assert result == live
    assert spy.live_calls == 1
    assert spy.saved_base == []
    assert spy.saved_overlay == []
    assert any(
        "Redaction breaks payload validation" in record.message
        for record in caplog.records
    )


def test_concurrent_invalid_cache_refetch_is_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
    runner: Runner,
) -> None:
    """M2: workers hitting the same invalid cached key fetch live only once."""
    invalid = {"id": 123}
    valid = {"id": 123, "segment_efforts": []}
    store: dict[str, Any] = {"payload": invalid}
    # Both threads must complete their initial (invalid) cache read before
    # either proceeds to the locked refetch; the barrier forces the overlap.
    barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    reads = 0
    live_calls = 0

    def fake_get_cached(*args: Any, **kwargs: Any) -> Any:
        nonlocal reads
        with state_lock:
            reads += 1
            initial_read = reads <= 2
        if initial_read:
            barrier.wait(timeout=5)
        return store["payload"]

    def fake_fetch_json(
        api: ResourceAPI,
        fetch_runner: Runner,
        url: str,
        params: Optional[Dict[str, Any]],
        context: str,
    ) -> Any:
        nonlocal live_calls
        with state_lock:
            live_calls += 1
        return valid

    def fake_save_overlay(*args: Any, response: Any, **kwargs: Any) -> None:
        store["payload"] = response

    def fail_base_save(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("base save must not be used for refetch healing")

    monkeypatch.setattr(resources, "get_cached_response", fake_get_cached)
    monkeypatch.setattr(resources, "save_overlay_to_cache", fake_save_overlay)
    monkeypatch.setattr(resources, "save_response_to_cache", fail_base_save)
    monkeypatch.setattr(resources, "_cache_mode_saves", True)
    monkeypatch.setattr(resources, "_cache_mode_offline", False)
    monkeypatch.setattr(ResourceAPI, "fetch_json", fake_fetch_json, raising=False)

    api = ResourceAPI()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                api.fetch_with_capture,
                runner,
                URL,
                PARAMS,
                "activity_detail",
                validate=_valid_activity_detail,
            )
            for _ in range(2)
        ]
        results = [future.result(timeout=10) for future in futures]

    assert live_calls == 1
    assert results == [valid, valid]
    assert store["payload"] == valid
