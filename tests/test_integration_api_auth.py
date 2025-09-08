from datetime import datetime
import json

from strava_competition.models import Runner
from strava_competition import strava_api, auth


class FakeResp:
    def __init__(self, status_code=200, data=None, headers=None):
        self.status_code = status_code
        self._data = data if data is not None else []
        self.headers = headers or {}

    def json(self):
        return self._data

    @property
    def text(self):
        try:
            return json.dumps(self._data)
        except Exception:
            return str(self._data)

    @property
    def content(self):
        return self.text.encode()

    def raise_for_status(self):
        if 400 <= self.status_code:
            import requests
            raise requests.exceptions.HTTPError(response=self)


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
    monkeypatch.setattr(auth, "_session", type("S", (), {"post": staticmethod(fake_post)})())

    # Patch GET efforts sequence: first 401, then success page (<200 to stop)
    get_calls = {"count": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        get_calls["count"] += 1
        if get_calls["count"] == 1:
            # 401 triggers refresh
            return FakeResp(401, data=[])
        # Successful page (single effort)
        return FakeResp(200, data=[{"elapsed_time": 123, "start_date_local": "2024-01-05T09:00:00Z"}])

    monkeypatch.setattr(strava_api._session, "get", fake_get)

    runner = Runner(name="Alice", strava_id=1, refresh_token="origRT", segment_team="Red")

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
