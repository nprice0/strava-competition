"""Tests for Strava activity type filtering helpers."""

from __future__ import annotations

from strava_competition.activity_types import activity_type_matches


def test_activity_matches_when_type_only() -> None:
    activity = {"type": "Run"}
    assert activity_type_matches(activity, {"run"}) is True


def test_activity_matches_when_sport_type_is_more_specific() -> None:
    activity = {"sport_type": "TrailRun", "type": "Run"}
    assert activity_type_matches(activity, {"run"}) is True


def test_activity_requires_allowed_type() -> None:
    activity = {"sport_type": "Walk", "type": "Walk"}
    assert activity_type_matches(activity, {"run"}) is False


def test_activity_matches_specific_sport_type() -> None:
    activity = {"sport_type": "TrailRun"}
    assert activity_type_matches(activity, {"trailrun"}) is True
