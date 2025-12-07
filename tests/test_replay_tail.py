"""Tests for replay_tail utility helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from strava_competition.replay_tail import (
    ActivityStats,
    cache_is_stale,
    chunk_activities,
    dedupe_activities,
    merge_activity_lists,
    summarize_activities,
)


def _activity(activity_id: int, start: datetime) -> dict:
    return {
        "id": activity_id,
        "start_date": start.isoformat().replace("+00:00", "Z"),
    }


def test_summarize_activities_tracks_extremes() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    acts = [_activity(1, base), _activity(2, base + timedelta(days=1))]
    stats = summarize_activities(acts)
    assert stats == ActivityStats(count=2, latest=base + timedelta(days=1), oldest=base)


def test_dedupe_preserves_order() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    activities = [
        _activity(1, base),
        _activity(2, base + timedelta(minutes=5)),
        _activity(1, base + timedelta(minutes=10)),
    ]
    deduped = dedupe_activities(activities)
    assert [item["id"] for item in deduped] == [1, 2]


def test_merge_activity_lists_gives_precedence_to_first() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    tail = [_activity(3, base + timedelta(hours=1))]
    cached = [_activity(3, base), _activity(4, base + timedelta(hours=2))]
    merged = merge_activity_lists(tail, cached)
    assert [item["id"] for item in merged] == [3, 4]


@pytest.mark.parametrize(
    "delta_days,expected",
    [(-1, False), (0, False), (8, True)],
)
def test_cache_is_stale(delta_days: int, expected: bool) -> None:
    captured = datetime.now(timezone.utc) - timedelta(days=max(delta_days, 0))
    assert cache_is_stale(captured, ttl_days=7) is expected


def test_chunk_activities_handles_arbitrary_lengths() -> None:
    acts = [
        _activity(idx, datetime(2024, 1, 1, tzinfo=timezone.utc)) for idx in range(5)
    ]
    chunks = chunk_activities(acts, chunk_size=2)
    assert len(chunks) == 3
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
