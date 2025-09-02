import types
from datetime import datetime

import pytest

from strava_competition import strava_api
from strava_competition.models import Runner


class FakeResp:
    def __init__(self, status_code=200, data=None, headers=None):
        self.status_code = status_code
        self._data = data or []
        self.headers = headers or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if 400 <= self.status_code:
            import requests

            raise requests.exceptions.HTTPError(response=self)


@pytest.fixture(autouse=True)
def no_sleep_rate_limiter(monkeypatch):
    class NoopLimiter:
        def before_request(self):
            return

        def after_response(self, headers, status_code):
            return

    monkeypatch.setattr(strava_api, "_limiter", NoopLimiter())


def test_get_segment_efforts_pagination(monkeypatch):
    calls = {"count": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        calls["count"] += 1
        page = int(params.get("page", 1))
        if page == 1:
            # Return a full page of 200 items to trigger pagination
            data = [
                {"elapsed_time": 100 + i, "start_date_local": f"2024-01-02T10:00:{i:02d}Z"}
                for i in range(200)
            ]
            return FakeResp(200, data=data, headers={"X-RateLimit-Usage": "10,100", "X-RateLimit-Limit": "100,1000"})
        else:
            # Second page empty -> stop
            return FakeResp(200, data=[], headers={"X-RateLimit-Usage": "11,100", "X-RateLimit-Limit": "100,1000"})

    # Patch session.get and token refresh
    monkeypatch.setattr(strava_api._session, "get", fake_get)
    monkeypatch.setattr(strava_api, "get_access_token", lambda rt, runner_name=None: ("at1", rt))

    runner = Runner(name="Alice", strava_id=1, refresh_token="rt1", team="Red")
    efforts = strava_api.get_segment_efforts(
        runner,
        segment_id=123,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
    )
    assert isinstance(efforts, list)
    assert len(efforts) == 200
    assert calls["count"] == 2  # 2 pages


def test_get_segment_efforts_refresh_on_401(monkeypatch):
    state = {"call": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        # First call returns 401, next returns data
        if state["call"] == 0:
            state["call"] += 1
            return FakeResp(401, data=[])
        else:
            return FakeResp(200, data=[{"elapsed_time": 95, "start_date_local": "2024-01-04T12:00:00Z"}])

    def fake_refresh(rt, runner_name=None):
        return ("new_access", rt)

    monkeypatch.setattr(strava_api._session, "get", fake_get)
    monkeypatch.setattr(strava_api, "get_access_token", fake_refresh)

    runner = Runner(name="Bob", strava_id=2, refresh_token="rt2", team="Blue")
    efforts = strava_api.get_segment_efforts(
        runner,
        segment_id=456,
        start_date=datetime(2024, 2, 1),
        end_date=datetime(2024, 2, 28),
    )
    # Should recover after a refresh and return one effort
    assert len(efforts) == 1