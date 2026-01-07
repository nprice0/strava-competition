from datetime import datetime, timezone
from types import SimpleNamespace

from strava_competition import strava_api
from strava_competition.models import Runner


def _runner() -> Runner:
    return Runner(name="Test", strava_id=123, refresh_token="rt", segment_team="A")


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


def test_strava_client_injects_session_and_limiter(monkeypatch):
    calls = {}

    def fake_segment_impl(*args, **kwargs):
        calls["segment_session"] = kwargs["session"]
        calls["segment_limiter"] = kwargs["limiter"]
        return []

    def fake_activities_impl(*args, **kwargs):
        calls["activities_session"] = kwargs["session"]
        calls["activities_limiter"] = kwargs["limiter"]
        return []

    monkeypatch.setattr(strava_api, "_get_segment_efforts_impl", fake_segment_impl)
    monkeypatch.setattr(strava_api, "_get_activities_impl", fake_activities_impl)

    dummy_session = object()
    dummy_limiter = SimpleNamespace(resize=lambda *_args, **_kwargs: None)
    client = strava_api.StravaClient(session=dummy_session, limiter=dummy_limiter)

    runner = _runner()
    client.get_segment_efforts(runner, 1, _utcnow(), _utcnow())
    client.get_activities(runner, _utcnow(), _utcnow())

    assert calls["segment_session"] is dummy_session
    assert calls["segment_limiter"] is dummy_limiter
    assert calls["activities_session"] is dummy_session
    assert calls["activities_limiter"] is dummy_limiter


def test_module_wrappers_delegate_to_default_client(monkeypatch):
    runner = _runner()
    called = {"efforts": False, "activities": False, "resized": False}

    def fake_efforts(*args, **kwargs):
        called["efforts"] = True
        return ["ok"]

    def fake_activities(*args, **kwargs):
        called["activities"] = True
        return ["activity"]

    def fake_resize(value):
        called["resized"] = value

    client = strava_api.get_default_client()
    monkeypatch.setattr(client, "get_segment_efforts", fake_efforts)
    monkeypatch.setattr(client, "get_activities", fake_activities)
    monkeypatch.setattr(client, "set_rate_limiter", fake_resize)

    assert strava_api.get_segment_efforts(runner, 1, _utcnow(), _utcnow()) == ["ok"]
    assert strava_api.get_activities(runner, _utcnow(), _utcnow()) == ["activity"]
    strava_api.set_rate_limiter(5)
    assert called["efforts"]
    assert called["activities"]
    assert called["resized"] == 5
