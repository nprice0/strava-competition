from datetime import datetime

import pytest

from strava_competition import config, strava_api
from strava_competition.errors import StravaAPIError
from strava_competition.models import Runner
from strava_competition.strava_client import resources as resource_client
from strava_competition.strava_client import segment_efforts as segment_client

from conftest import FakeResp, _patch_session


@pytest.fixture(autouse=True)
def reset_default_client(monkeypatch):
    """Reset the default client before each test to avoid cross-test pollution."""
    monkeypatch.setattr(strava_api, "_default_client", None)


@pytest.fixture(autouse=True)
def no_sleep_rate_limiter(monkeypatch):
    class NoopLimiter:
        def before_request(self):
            return

        def after_response(self, headers, status_code):
            return

    monkeypatch.setattr(strava_api, "_limiter", NoopLimiter())


@pytest.fixture(autouse=True)
def disable_capture(monkeypatch):
    """Force cache misses and prevent filesystem writes during unit tests."""

    monkeypatch.setattr(resource_client, "_cache_mode_offline", False)
    monkeypatch.setattr(segment_client, "_cache_mode_offline", False)
    monkeypatch.setattr(
        resource_client, "get_cached_response", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        resource_client, "save_response_to_cache", lambda *args, **kwargs: None
    )
    # Also disable segment_efforts replay
    monkeypatch.setattr(
        segment_client, "get_cached_list_with_meta", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        segment_client, "save_list_to_cache", lambda *args, **kwargs: None
    )


@pytest.fixture(autouse=True)
def fake_token_refresh(monkeypatch):
    """Avoid hitting the real OAuth token exchange during tests."""

    monkeypatch.setattr(
        "strava_competition.strava_client.base.get_access_token",
        lambda rt, runner_name=None: ("stub_access", rt),
    )


def test_get_segment_efforts_pagination(monkeypatch):
    """Full first page (200) then empty second page triggers exactly 2 calls."""
    calls = {"count": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        calls["count"] += 1
        page = int(params.get("page", 1))
        if page == 1:
            data = [
                {
                    "elapsed_time": 100 + i,
                    "start_date_local": f"2024-01-02T10:00:{i:02d}Z",
                }
                for i in range(200)
            ]
            return FakeResp(
                200,
                data=data,
                headers={
                    "X-RateLimit-Usage": "10,100",
                    "X-RateLimit-Limit": "100,1000",
                },
            )
        return FakeResp(
            200,
            data=[],
            headers={"X-RateLimit-Usage": "11,100", "X-RateLimit-Limit": "100,1000"},
        )

    mock_session = type("MockSession", (), {"get": staticmethod(fake_get)})()
    _patch_session(monkeypatch, mock_session)
    monkeypatch.setattr(
        "strava_competition.strava_client.base.get_access_token",
        lambda rt, runner_name=None: ("at1", rt),
    )

    runner = Runner(name="Alice", strava_id=1, refresh_token="rt1", segment_team="Red")
    efforts = strava_api.get_segment_efforts(
        runner,
        segment_id=123,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
    )
    assert isinstance(efforts, list)
    assert len(efforts) == 200
    assert calls["count"] == 2


def test_get_segment_efforts_refresh_on_401(monkeypatch):
    """First call 401 then success after token refresh."""
    state = {"call": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        if state["call"] == 0:
            state["call"] += 1
            return FakeResp(401, data=[])
        return FakeResp(
            200, data=[{"elapsed_time": 95, "start_date_local": "2024-01-04T12:00:00Z"}]
        )

    mock_session = type("MockSession", (), {"get": staticmethod(fake_get)})()
    _patch_session(monkeypatch, mock_session)
    monkeypatch.setattr(
        "strava_competition.strava_client.base.get_access_token",
        lambda rt, runner_name=None: ("new_access", rt),
    )

    runner = Runner(name="Ben", strava_id=2, refresh_token="rt2", segment_team="Blue")
    efforts = strava_api.get_segment_efforts(
        runner,
        segment_id=456,
        start_date=datetime(2024, 2, 1),
        end_date=datetime(2024, 2, 28),
    )
    assert len(efforts) == 1


def test_get_segment_efforts_402_json_error(monkeypatch, caplog):
    """402 JSON error returns None and logs message + code detail."""
    error_payload = {
        "message": "Payment Required",
        "errors": [
            {"resource": "segment", "field": "efforts", "code": "payment_required"}
        ],
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        return FakeResp(
            402, data=error_payload, headers={"Content-Type": "application/json"}
        )

    mock_session = type("MockSession", (), {"get": staticmethod(fake_get)})()
    _patch_session(monkeypatch, mock_session)
    monkeypatch.setattr(
        "strava_competition.strava_client.base.get_access_token",
        lambda rt, runner_name=None: ("tok", rt),
    )

    runner = Runner(name="Cara", strava_id=3, refresh_token="rt3", segment_team="Green")
    caplog.set_level("INFO")
    efforts = strava_api.get_segment_efforts(
        runner,
        segment_id=789,
        start_date=datetime(2024, 3, 1),
        end_date=datetime(2024, 3, 31),
    )
    assert efforts is None
    combined = "\n".join(caplog.messages)
    assert "Payment Required" in combined
    assert "segment/efforts:payment_required" in combined


def test_fetch_segment_geometry_offline_requires_capture(monkeypatch):
    runner = Runner(
        name="Offline", strava_id=99, refresh_token="rt", segment_team="Solo"
    )

    monkeypatch.setattr(resource_client, "_cache_mode_offline", True)
    monkeypatch.setattr(
        resource_client, "get_cached_response", lambda *args, **kwargs: None
    )

    def fail_get(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("HTTP call performed in offline mode")

    mock_session = type("MockSession", (), {"get": staticmethod(fail_get)})()
    _patch_session(monkeypatch, mock_session)

    with pytest.raises(StravaAPIError) as excinfo:
        strava_api.fetch_segment_geometry(runner, segment_id=111)

    assert "cache miss" in str(excinfo.value)


def test_get_segment_efforts_offline_cache_miss(monkeypatch):
    """Verify get_segment_efforts raises StravaAPIError on cache miss in offline mode."""
    runner = Runner(
        name="Offline", strava_id=42, refresh_token="rt", segment_team="Solo"
    )

    monkeypatch.setattr(segment_client, "_cache_mode_offline", True)
    monkeypatch.setattr(config, "_cache_mode_offline", True, raising=False)
    # Mock the underlying cache function to return None (cache miss)
    monkeypatch.setattr(
        segment_client,
        "get_cached_list_with_meta",
        lambda *args, **kwargs: None,
    )

    def fail_get(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("HTTP call performed in offline mode")

    mock_session = type("MockSession", (), {"get": staticmethod(fail_get)})()
    _patch_session(monkeypatch, mock_session)

    with pytest.raises(StravaAPIError) as excinfo:
        strava_api.get_segment_efforts(
            runner,
            segment_id=321,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
        )

    assert "miss" in str(excinfo.value).lower()
