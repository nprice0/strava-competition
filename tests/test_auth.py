import pytest

from strava_competition import auth
from conftest import FakeResp
from typing import Any


def _set_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "CLIENT_ID", "cid")
    monkeypatch.setattr(auth, "CLIENT_SECRET", "csec")


def _mock_session_with_post(monkeypatch: pytest.MonkeyPatch, fake_post: Any) -> None:
    """Patch _get_session() to return a mock with the given post function."""
    mock_session = type("MockSession", (), {"post": staticmethod(fake_post)})()
    monkeypatch.setattr(auth, "_get_session", lambda: mock_session)


def test_get_access_token_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_creds(monkeypatch)

    def fake_post(url: Any, data: Any = None, timeout: Any = None) -> Any:
        assert data["grant_type"] == "refresh_token"
        return FakeResp(200, data={"access_token": "AAA", "refresh_token": "BBB"})

    _mock_session_with_post(monkeypatch, fake_post)
    at, rt = auth.get_access_token("refresh123", runner_name="Runner1")
    assert at == "AAA"
    assert rt == "BBB"


def test_get_access_token_http_error_with_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_creds(monkeypatch)

    def fake_post(url: Any, data: Any = None, timeout: Any = None) -> Any:
        return FakeResp(
            400,
            data={
                "message": "Bad Request",
                "errors": [{"field": "refresh_token", "code": "invalid"}],
            },
        )

    _mock_session_with_post(monkeypatch, fake_post)
    with pytest.raises(auth.TokenError):
        auth.get_access_token("badtoken")


def test_get_access_token_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_creds(monkeypatch)

    class BadJSONResp(FakeResp):
        def json(self) -> Any:  # force JSON decode error path
            raise ValueError("invalid json")

    def fake_post(url: Any, data: Any = None, timeout: Any = None) -> Any:
        return BadJSONResp(200, data=None, text="not-json")

    _mock_session_with_post(monkeypatch, fake_post)
    with pytest.raises(auth.TokenError):
        auth.get_access_token("refresh123")


def test_get_access_token_missing_access_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_creds(monkeypatch)

    def fake_post(url: Any, data: Any = None, timeout: Any = None) -> Any:
        return FakeResp(200, data={"refresh_token": "NEW"})

    _mock_session_with_post(monkeypatch, fake_post)
    with pytest.raises(auth.TokenError):
        auth.get_access_token("refresh123")


def test_get_access_token_missing_client_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure credentials appear missing inside auth
    monkeypatch.setattr(auth, "CLIENT_ID", "")
    monkeypatch.setattr(auth, "CLIENT_SECRET", "")
    with pytest.raises(auth.TokenError):
        auth.get_access_token("refresh123")


def test_mask_tail_fully_masks_short_values() -> None:
    assert auth._mask_tail("abcd", visible=4) == "****"
    assert auth._mask_tail("ab", visible=4) == "**"
    assert auth._mask_tail("abcdefgh", visible=4) == "****efgh"
    assert auth._mask_tail("", visible=4) == ""
    assert auth._mask_tail(None, visible=4) == ""


def _setup_429_env(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Common 429-test setup: creds, no cooldown carry-over, recorded sleeps."""
    _set_creds(monkeypatch)
    monkeypatch.setattr(auth, "_rate_limit_cooldown_until", 0.0)
    sleeps: list[float] = []
    monkeypatch.setattr(auth.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def test_429_honors_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = _setup_429_env(monkeypatch)
    # Computed backoff would be 50s; Retry-After must win.
    monkeypatch.setattr(auth, "RATE_LIMIT_THROTTLE_SECONDS", 50)
    calls = {"n": 0}

    def fake_post(url: Any, data: Any = None, timeout: Any = None) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResp(429, data={}, headers={"Retry-After": "3"})
        return FakeResp(200, data={"access_token": "AAA", "refresh_token": "BBB"})

    _mock_session_with_post(monkeypatch, fake_post)
    at, _ = auth.get_access_token("refresh123")
    assert at == "AAA"
    assert 3.0 in sleeps
    assert 50 not in sleeps


def test_429_invalid_retry_after_falls_back_to_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = _setup_429_env(monkeypatch)
    monkeypatch.setattr(auth, "RATE_LIMIT_THROTTLE_SECONDS", 7)
    calls = {"n": 0}

    def fake_post(url: Any, data: Any = None, timeout: Any = None) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResp(429, data={}, headers={"Retry-After": "soon"})
        return FakeResp(200, data={"access_token": "AAA"})

    _mock_session_with_post(monkeypatch, fake_post)
    auth.get_access_token("refresh123")
    assert 7 in sleeps


def test_429_retry_after_capped_at_max_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = _setup_429_env(monkeypatch)
    monkeypatch.setattr(auth, "RATE_LIMIT_THROTTLE_SECONDS", 1)
    calls = {"n": 0}

    def fake_post(url: Any, data: Any = None, timeout: Any = None) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResp(429, data={}, headers={"Retry-After": "500"})
        return FakeResp(200, data={"access_token": "AAA"})

    _mock_session_with_post(monkeypatch, fake_post)
    auth.get_access_token("refresh123")
    assert auth._RATE_LIMIT_MAX_BACKOFF_SECONDS in sleeps
    assert 500.0 not in sleeps


def test_429_sets_shared_cooldown_other_calls_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = _setup_429_env(monkeypatch)
    monkeypatch.setattr(auth, "RATE_LIMIT_THROTTLE_SECONDS", 1)
    calls = {"n": 0}

    def fake_post(url: Any, data: Any = None, timeout: Any = None) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResp(429, data={}, headers={"Retry-After": "30"})
        return FakeResp(200, data={"access_token": "AAA"})

    _mock_session_with_post(monkeypatch, fake_post)
    auth.get_access_token("refresh123")
    # The 429 recorded a process-wide cooldown in the future.
    assert auth._rate_limit_cooldown_until > auth.time.monotonic()

    # A subsequent refresh waits out the shared cooldown before posting.
    sleeps.clear()
    auth.get_access_token("refresh456")
    assert sleeps, "second refresh should wait out the shared cooldown"
    assert sleeps[0] == pytest.approx(30.0, abs=1.0)
