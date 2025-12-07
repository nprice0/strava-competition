"""Tests covering capture overlays used by replay-tail."""

from __future__ import annotations

from strava_competition import api_capture


def test_overlay_preferred_over_base(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_capture, "STRAVA_API_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(api_capture, "STRAVA_API_CAPTURE_ENABLED", True)
    monkeypatch.setattr(api_capture, "STRAVA_API_REPLAY_ENABLED", True)
    monkeypatch.setattr(api_capture, "STRAVA_API_CAPTURE_OVERWRITE", True)

    capture = api_capture.APICapture()
    key = api_capture.CaptureKey(
        method="GET",
        url="https://test.example/activities",
        identity="runner-1",
        params={"page": 1},
        body=None,
    )

    capture.store(key, ["base"])
    base_record = capture.fetch_record(key)
    assert base_record is not None
    assert base_record.response == ["base"]
    assert base_record.source == "base"

    capture.store_overlay(key, ["overlay"])
    overlay_record = capture.fetch_record(key)
    assert overlay_record is not None
    assert overlay_record.response == ["overlay"]
    assert overlay_record.source == "overlay"
