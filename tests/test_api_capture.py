"""Tests for the Strava API capture layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from strava_competition import api_capture


@pytest.fixture(name="temp_capture")
def fixture_temp_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> api_capture.APICapture:
    """Return an APICapture configured to read/write within a temp directory."""

    capture_dir = tmp_path / "captures"
    monkeypatch.setattr(api_capture, "STRAVA_API_CAPTURE_DIR", str(capture_dir))
    monkeypatch.setattr(api_capture, "STRAVA_API_CAPTURE_ENABLED", True)
    monkeypatch.setattr(api_capture, "STRAVA_API_REPLAY_ENABLED", True)
    monkeypatch.setattr(api_capture, "STRAVA_API_CAPTURE_OVERWRITE", True)
    return api_capture.APICapture()


def test_roundtrip_store_and_replay(temp_capture: api_capture.APICapture) -> None:
    """Recorded payloads should be retrievable with the same key."""

    key = api_capture.CaptureKey(
        method="GET",
        url="https://example.com/athlete/activities",
        identity="runner-123",
        params={"page": 1, "per_page": 200},
        body=None,
    )

    assert temp_capture.fetch(key) is None
    payload = {"items": [1, 2, 3]}
    temp_capture.store(key, payload)
    assert temp_capture.fetch(key) == payload


def test_capture_creates_subdirectories(
    temp_capture: api_capture.APICapture, tmp_path: Path
) -> None:
    """Capture files are stored within hashed subdirectories for scalability."""

    key = api_capture.CaptureKey(
        method="GET",
        url="https://example.com/resource",
        identity="runner-456",
        params={"page": 5},
        body=None,
    )
    temp_capture.store(key, {"value": 42})
    base_dir = Path(api_capture.STRAVA_API_CAPTURE_DIR)
    # The capture builds a nested folder structure using the signature prefix.
    files = list(base_dir.rglob("*.json"))
    assert files, "expected at least one capture file"
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        assert "captured_at" in data
        assert data["request"]["identity"] == key.identity
        assert data["response"], "response payload missing"
