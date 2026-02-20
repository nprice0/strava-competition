import pytest

from strava_competition import strava_api
from strava_competition.errors import StravaAPIError
from strava_competition.models import Runner
from strava_competition.strava_client import resources as resource_client

from conftest import _patch_session
from typing import Any


@pytest.fixture(autouse=True)
def reset_default_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the default client before each test to avoid cross-test pollution."""
    monkeypatch.setattr(strava_api, "_default_client", None)


@pytest.fixture(autouse=True)
def no_sleep_rate_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoopLimiter:
        def before_request(self) -> None:
            return

        def after_response(self, headers: Any, status_code: Any) -> tuple[bool, str]:
            return False, ""

    monkeypatch.setattr(strava_api, "_limiter", NoopLimiter())


@pytest.fixture(autouse=True)
def disable_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force cache misses and prevent filesystem writes during unit tests."""

    monkeypatch.setattr(resource_client, "_cache_mode_offline", False)
    monkeypatch.setattr(
        resource_client, "get_cached_response", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        resource_client, "save_response_to_cache", lambda *args, **kwargs: None
    )


@pytest.fixture(autouse=True)
def fake_token_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid hitting the real OAuth token exchange during tests."""

    monkeypatch.setattr(
        "strava_competition.strava_client.base.get_access_token",
        lambda rt, runner_name=None: ("stub_access", rt),
    )


def test_fetch_segment_geometry_offline_requires_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner(
        name="Offline", strava_id="99", refresh_token="rt", segment_team="Solo"
    )

    monkeypatch.setattr(resource_client, "_cache_mode_offline", True)
    monkeypatch.setattr(
        resource_client, "get_cached_response", lambda *args, **kwargs: None
    )

    def fail_get(
        *args: Any, **kwargs: Any
    ) -> None:  # pragma: no cover - should never run
        raise AssertionError("HTTP call performed in offline mode")

    mock_session = type("MockSession", (), {"get": staticmethod(fail_get)})()
    _patch_session(monkeypatch, mock_session)

    with pytest.raises(StravaAPIError) as excinfo:
        strava_api.fetch_segment_geometry(runner, segment_id=111)

    assert "cache miss" in str(excinfo.value)
