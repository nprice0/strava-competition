import json
import pytest

from strava_competition import auth


class FakeResp:
    def __init__(self, status_code=200, data=None, text=None):
        self.status_code = status_code
        self._data = data
        self._text = text

    def json(self):
        if isinstance(self._data, Exception):  # force JSON error
            raise self._data
        return self._data

    @property
    def text(self):  # emulate .text attribute for snippet logging
        if self._text is not None:
            return self._text
        try:
            return json.dumps(self._data)
        except Exception:
            return str(self._data)


def _set_creds(monkeypatch):
    monkeypatch.setattr(auth, "CLIENT_ID", "cid")
    monkeypatch.setattr(auth, "CLIENT_SECRET", "csec")


def test_get_access_token_success(monkeypatch):
    _set_creds(monkeypatch)

    def fake_post(url, data=None, timeout=None):
        assert data["grant_type"] == "refresh_token"
        return FakeResp(200, data={"access_token": "AAA", "refresh_token": "BBB"})

    monkeypatch.setattr(auth, "_session", type("S", (), {"post": staticmethod(fake_post)})())
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

    monkeypatch.setattr(auth, "_session", type("S", (), {"post": staticmethod(fake_post)})())
    with pytest.raises(auth.TokenError):
        auth.get_access_token("badtoken")


def test_get_access_token_invalid_json(monkeypatch):
    _set_creds(monkeypatch)

    class BadJSONResp(FakeResp):
        def json(self):  # force JSON decode error path
            raise ValueError("invalid json")

    def fake_post(url, data=None, timeout=None):
        return BadJSONResp(200, data=None, text="not-json")

    monkeypatch.setattr(auth, "_session", type("S", (), {"post": staticmethod(fake_post)})())
    with pytest.raises(auth.TokenError):
        auth.get_access_token("refresh123")


def test_get_access_token_missing_access_token(monkeypatch):
    _set_creds(monkeypatch)

    def fake_post(url, data=None, timeout=None):
        return FakeResp(200, data={"refresh_token": "NEW"})

    monkeypatch.setattr(auth, "_session", type("S", (), {"post": staticmethod(fake_post)})())
    with pytest.raises(auth.TokenError):
        auth.get_access_token("refresh123")


def test_get_access_token_missing_client_creds(monkeypatch):
    # Ensure credentials appear missing inside auth
    monkeypatch.setattr(auth, "CLIENT_ID", "")
    monkeypatch.setattr(auth, "CLIENT_SECRET", "")
    with pytest.raises(auth.TokenError):
        auth.get_access_token("refresh123")
