from datetime import datetime

import pytest

from strava_competition.models import Runner
from strava_competition import strava_api, auth
from strava_competition.strava_client import session as session_module
from strava_competition.strava_client import segment_efforts as segment_client
from strava_competition.strava_client import resources as resource_client

from conftest import FakeResp


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
            return False, ""

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


def _patch_all_sessions(monkeypatch, mock_session):
    """Patch get_default_session in all modules that import it.

    Also resets the default client to force re-creation with the mocked session.
    """
    # First patch the get_default_session functions
    monkeypatch.setattr(session_module, "get_default_session", lambda: mock_session)
    monkeypatch.setattr(strava_api, "get_default_session", lambda: mock_session)
    monkeypatch.setattr(segment_client, "get_default_session", lambda: mock_session)
    monkeypatch.setattr(auth, "_get_session", lambda: mock_session)
    # Reset the module-level cached client so get_default_client creates a fresh one
    monkeypatch.setattr(strava_api, "_default_client", None)
    # Force recreation and update the module-level reference
    strava_api.get_default_client()


def test_integration_token_refresh_and_segment_fetch(monkeypatch):
    """End-to-end: initial token fetch, 401 triggers refresh, then successful efforts page.

    Verifies interaction between auth.get_access_token (real implementation) and
    strava_api.get_segment_efforts without stubbing get_access_token directly.
    """
    # Patch Strava token endpoint POSTs (auth module session)
    token_calls = {"count": 0}
    token_sequence = [
        {"access_token": "A1", "refresh_token": "R1"},  # initial ensure_token
        {"access_token": "A2", "refresh_token": "R2"},  # after 401 refresh
    ]

    def fake_post(url, data=None, timeout=None):
        idx = token_calls["count"]
        token_calls["count"] += 1
        return FakeResp(200, data=token_sequence[idx])

    monkeypatch.setattr(auth, "CLIENT_ID", "cid")
    monkeypatch.setattr(auth, "CLIENT_SECRET", "csec")

    # Patch GET efforts sequence: first 401, then success page (<200 to stop)
    get_calls = {"count": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        get_calls["count"] += 1
        if get_calls["count"] == 1:
            # 401 triggers refresh
            return FakeResp(401, data=[])
        # Successful page (single effort)
        return FakeResp(
            200,
            data=[{"elapsed_time": 123, "start_date_local": "2024-01-05T09:00:00Z"}],
        )

    # Create a mock session with both get and post methods
    mock_session = type(
        "MockSession",
        (),
        {"get": staticmethod(fake_get), "post": staticmethod(fake_post)},
    )()
    _patch_all_sessions(monkeypatch, mock_session)

    runner = Runner(
        name="Alice", strava_id=1, refresh_token="origRT", segment_team="Red"
    )

    efforts = strava_api.get_segment_efforts(
        runner,
        segment_id=42,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
    )

    # Assertions
    assert isinstance(efforts, list)
    assert len(efforts) == 1
    assert token_calls["count"] == 2, "Expected initial token + refresh token calls"
    # Runner tokens updated to last response
    assert runner.access_token == "A2"
    assert runner.refresh_token == "R2"
    assert get_calls["count"] == 2, "First 401 + second success"
