import pytest

from strava_competition import strava_api, auth
from strava_competition.strava_client import session as session_module
from strava_competition.strava_client import resources as resource_client


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
    monkeypatch.setattr(
        resource_client, "get_cached_response", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        resource_client, "save_response_to_cache", lambda *args, **kwargs: None
    )


def _patch_all_sessions(monkeypatch, mock_session):
    """Patch get_default_session in all modules that import it.

    Also resets the default client to force re-creation with the mocked session.
    """
    # First patch the get_default_session functions
    monkeypatch.setattr(session_module, "get_default_session", lambda: mock_session)
    monkeypatch.setattr(strava_api, "get_default_session", lambda: mock_session)
    monkeypatch.setattr(auth, "_get_session", lambda: mock_session)
    # Reset the module-level cached client so get_default_client creates a fresh one
    monkeypatch.setattr(strava_api, "_default_client", None)
    # Force recreation and update the module-level reference
    strava_api.get_default_client()
