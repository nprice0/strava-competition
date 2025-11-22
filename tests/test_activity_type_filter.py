"""Tests for Strava activity type filtering helpers."""

from __future__ import annotations

import os
from pathlib import Path

TEST_CAPTURE_DIR = Path(__file__).resolve().parent / "strava_api_capture"
os.environ.setdefault("STRAVA_API_CAPTURE_DIR", str(TEST_CAPTURE_DIR))
os.environ.setdefault("STRAVA_API_REPLAY_ENABLED", "true")
os.environ.setdefault("STRAVA_API_CAPTURE_ENABLED", "false")

from strava_competition.strava_api import _activity_type_matches


def test_activity_matches_when_type_only() -> None:
    activity = {"type": "Run"}
    assert _activity_type_matches(activity, {"run"}) is True


def test_activity_matches_when_sport_type_is_more_specific() -> None:
    activity = {"sport_type": "TrailRun", "type": "Run"}
    assert _activity_type_matches(activity, {"run"}) is True


def test_activity_requires_allowed_type() -> None:
    activity = {"sport_type": "Walk", "type": "Walk"}
    assert _activity_type_matches(activity, {"run"}) is False


def test_activity_matches_specific_sport_type() -> None:
    activity = {"sport_type": "TrailRun"}
    assert _activity_type_matches(activity, {"trailrun"}) is True
