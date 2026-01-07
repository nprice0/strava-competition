import pytest

from strava_competition import auth
from conftest import FakeResp


def _set_creds(monkeypatch):
    monkeypatch.setattr(auth, "CLIENT_ID", "cid")
    monkeypatch.setattr(auth, "CLIENT_SECRET", "csec")


def _mock_session_with_post(monkeypatch, fake_post):
    """Patch _get_session() to return a mock with the given post function."""
    mock_session = type("MockSession", (), {"post": staticmethod(fake_post)})()
    monkeypatch.setattr(auth, "_get_session", lambda: mock_session)


def test_get_access_token_success(monkeypatch):
    _set_creds(monkeypatch)

    def fake_post(url, data=None, timeout=None):
        assert data["grant_type"] == "refresh_token"
        return FakeResp(200, data={"access_token": "AAA", "refresh_token": "BBB"})

    _mock_session_with_post(monkeypatch, fake_post)
    at, rt = auth.get_access_token("refresh123", runner_name="Runner1")
    assert at == "AAA"
    assert rt == "BBB"


def test_get_access_token_http_error_with_json(monkeypatch):
    _set_creds(monkeypatch)

    def fake_post(url, data=None, timeout=None):
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


def test_get_access_token_invalid_json(monkeypatch):
    _set_creds(monkeypatch)

    class BadJSONResp(FakeResp):
        def json(self):  # force JSON decode error path
            raise ValueError("invalid json")

    def fake_post(url, data=None, timeout=None):
        return BadJSONResp(200, data=None, text="not-json")

    _mock_session_with_post(monkeypatch, fake_post)
    with pytest.raises(auth.TokenError):
        auth.get_access_token("refresh123")


def test_get_access_token_missing_access_token(monkeypatch):
    _set_creds(monkeypatch)

    def fake_post(url, data=None, timeout=None):
        return FakeResp(200, data={"refresh_token": "NEW"})

    _mock_session_with_post(monkeypatch, fake_post)
    with pytest.raises(auth.TokenError):
        auth.get_access_token("refresh123")


def test_get_access_token_missing_client_creds(monkeypatch):
    # Ensure credentials appear missing inside auth
    monkeypatch.setattr(auth, "CLIENT_ID", "")
    monkeypatch.setattr(auth, "CLIENT_SECRET", "")
    with pytest.raises(auth.TokenError):
        auth.get_access_token("refresh123")
