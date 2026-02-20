"""Load/smoke tests for concurrency and capture replay behaviour."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import Any, Dict, Mapping, Tuple

from strava_competition.models import Runner
from strava_competition.strava_client.rate_limiter import RateLimiter
from strava_competition.strava_client.resources import ResourceAPI
from strava_competition.strava_client import resources
import pytest


class RecordingLimiter(RateLimiter):
    """Rate limiter that records observed concurrency for assertions."""

    def __init__(self, max_concurrent: int) -> None:
        super().__init__(max_concurrent=max_concurrent, jitter_range=(0.0, 0.0))
        self._observed_lock = threading.Lock()
        self._current = 0
        self.max_observed = 0

    def before_request(self) -> None:
        super().before_request()
        with self._observed_lock:
            self._current += 1
            self.max_observed = max(self.max_observed, self._current)

    def after_response(
        self,
        headers: Mapping[str, object] | None,
        status_code: int | None,
    ) -> tuple[bool, str]:
        result = super().after_response(headers, status_code)
        with self._observed_lock:
            self._current = max(0, self._current - 1)
        return result


class ParallelSession:
    """Minimal requests.Session stand-in that tracks call count."""

    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay
        self.call_count = 0
        self._lock = threading.Lock()

    def get(
        self, url: str, headers: Dict[str, str], params: Dict[str, Any], timeout: int
    ) -> Any:
        with self._lock:
            self.call_count += 1
        time.sleep(self.delay)
        return _FakeResponse(payload={"url": url, "params": params})


class _FakeResponse:
    """Tiny Response-like object for ResourceAPI.fetch_json."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.status_code = 200
        self._payload = payload
        self.headers = {
            "X-RateLimit-Usage": "10,0",
            "X-RateLimit-Limit": "100,0",
            "Content-Type": "application/json",
        }
        self.text = ""
        self.url = payload.get("url", "https://example.test")

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


def _new_runner(idx: int) -> Runner:
    runner = Runner(name=f"Runner {idx}", strava_id=idx, refresh_token="rt")
    runner.access_token = "token"
    return runner


def _capture_key(
    method: str,
    url: str,
    identity: str,
    params: Dict[str, Any] | None,
    body: Dict[str, Any] | None,
) -> Tuple[
    str,
    str,
    str,
    Tuple[Tuple[str, Any], ...] | None,
    Tuple[Tuple[str, Any], ...] | None,
]:
    def _serialise(value: Dict[str, Any] | None) -> Tuple[Tuple[str, Any], ...] | None:
        if value is None:
            return None
        return tuple(sorted(value.items()))

    return (method, url, identity, _serialise(params), _serialise(body))


def test_resource_api_rate_limiter_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure multiple runners never exceed the configured concurrency cap."""

    limiter = RecordingLimiter(max_concurrent=3)
    session = ParallelSession(delay=0.02)
    monkeypatch.setattr(resources, "ensure_runner_token", lambda *_: None)

    api = ResourceAPI(session=session, limiter=limiter, timeout=1)  # type: ignore[arg-type]

    runners = [_new_runner(idx) for idx in range(8)]
    barrier = threading.Barrier(len(runners))

    def _fetch(runner: Runner) -> Any:
        barrier.wait()
        return api.fetch_json(
            runner,
            url="https://example.test/activities",
            params={"page": 1, "runner": runner.strava_id},
            context="load_smoke",
        )

    with ThreadPoolExecutor(max_workers=len(runners)) as executor:
        results = list(executor.map(_fetch, runners))

    assert len(results) == len(runners)
    assert session.call_count == len(runners)
    assert limiter.max_observed == 3


def test_capture_replay_smoke_multiple_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay hits should satisfy offline runs after an initial capture warm-up."""

    store: Dict[
        Tuple[
            str,
            str,
            str,
            Tuple[Tuple[str, Any], ...] | None,
            Tuple[Tuple[str, Any], ...] | None,
        ],
        Any,
    ] = {}
    store_lock = threading.Lock()
    live_fetch_enabled = {"value": True}
    seq_lock = threading.Lock()
    seq = {"value": 0}

    def fake_record_response(
        method: str,
        url: str,
        identity: str,
        response: Any,
        *,
        params: Dict[str, Any] | None = None,
        body: Dict[str, Any] | None = None,
    ) -> None:
        key = _capture_key(method, url, identity, params, body)
        with store_lock:
            store[key] = response

    def fake_replay_response(
        method: str,
        url: str,
        identity: str,
        *,
        params: Dict[str, Any] | None = None,
        body: Dict[str, Any] | None = None,
    ) -> Any:
        key = _capture_key(method, url, identity, params, body)
        with store_lock:
            return store.get(key)

    def fake_fetch_json(
        self: ResourceAPI,
        runner: Runner,
        url: str,
        params: Dict[str, Any] | None,
        context: str,
    ) -> Dict[str, Any]:
        if not live_fetch_enabled["value"]:
            raise AssertionError("Live fetch should not be invoked while offline")
        with seq_lock:
            seq["value"] += 1
            idx = seq["value"]
        return {"runner": runner.name, "seq": idx, "context": context}

    monkeypatch.setattr(resources, "save_response_to_cache", fake_record_response)
    monkeypatch.setattr(resources, "get_cached_response", fake_replay_response)
    monkeypatch.setattr(resources, "_cache_mode_saves", True)
    monkeypatch.setattr(resources, "_cache_mode_offline", False)
    monkeypatch.setattr(ResourceAPI, "fetch_json", fake_fetch_json, raising=False)

    api = ResourceAPI()
    runners = [_new_runner(idx) for idx in range(6)]
    url = "https://example.test/resource"

    def _fetch_with_capture(runner: Runner) -> Any:
        return api.fetch_with_capture(
            runner, url, params={"runner": runner.strava_id}, context="smoke"
        )

    with ThreadPoolExecutor(max_workers=len(runners)) as executor:
        first_pass = list(executor.map(_fetch_with_capture, runners))

    assert len(first_pass) == len(runners)
    assert len(store) == len(runners)

    live_fetch_enabled["value"] = False
    monkeypatch.setattr(resources, "_cache_mode_offline", True)

    with ThreadPoolExecutor(max_workers=len(runners)) as executor:
        replay_pass = list(executor.map(_fetch_with_capture, runners))

    assert replay_pass == first_pass
