"""Unit tests for SegmentService prefetch optimisation."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from strava_competition.services.segment_service import (
    SegmentService,
    PrefetchedActivity,
    RunnerActivityCache,
)


class TestComputeUnionDateRange:
    """Tests for _compute_union_date_range."""

    def test_single_segment(self):
        """Union of one segment is its own range."""
        service = SegmentService()
        seg = Mock(start_date=datetime(2026, 1, 1), end_date=datetime(2026, 1, 15))

        start, end = service._compute_union_date_range(segments=[seg])

        assert start == datetime(2026, 1, 1)
        assert end == datetime(2026, 1, 15)

    def test_overlapping_segments(self):
        """Union covers earliest to latest across overlapping segments."""
        service = SegmentService()
        seg1 = Mock(start_date=datetime(2026, 1, 1), end_date=datetime(2026, 1, 15))
        seg2 = Mock(start_date=datetime(2026, 1, 10), end_date=datetime(2026, 1, 25))

        start, end = service._compute_union_date_range(segments=[seg1, seg2])

        assert start == datetime(2026, 1, 1)
        assert end == datetime(2026, 1, 25)

    def test_segment_groups_with_windows(self):
        """Union covers all windows across groups."""
        service = SegmentService()
        window1 = Mock(start_date=datetime(2026, 1, 1), end_date=datetime(2026, 1, 10))
        window2 = Mock(start_date=datetime(2026, 2, 1), end_date=datetime(2026, 2, 28))
        group = Mock(windows=[window1, window2])

        start, end = service._compute_union_date_range(segment_groups=[group])

        assert start == datetime(2026, 1, 1)
        assert end == datetime(2026, 2, 28)

    def test_empty_raises(self):
        """Empty input raises ValueError."""
        service = SegmentService()

        with pytest.raises(ValueError, match="No segments"):
            service._compute_union_date_range()


class TestFindEffortsFromCache:
    """Tests for _find_efforts_from_cache."""

    def test_finds_matching_efforts(self):
        """Returns efforts matching segment ID within date window."""
        service = SegmentService()
        runner = Mock(strava_id=123)

        cache = {
            123: RunnerActivityCache(
                runner_id=123,
                activities=[
                    PrefetchedActivity(
                        activity_id=1,
                        start_date=datetime(2026, 1, 10),
                        activity_type="Run",
                        raw_data={},
                        segment_efforts=(
                            {"segment": {"id": 999}, "elapsed_time": 300},
                            {"segment": {"id": 888}, "elapsed_time": 200},
                        ),
                    )
                ],
            )
        }

        efforts = service._find_efforts_from_cache(
            runner, 999, datetime(2026, 1, 1), datetime(2026, 1, 15), cache
        )

        assert len(efforts) == 1
        assert efforts[0]["segment"]["id"] == 999

    def test_filters_by_date_window(self):
        """Excludes activities outside the date window."""
        service = SegmentService()
        runner = Mock(strava_id=123)

        cache = {
            123: RunnerActivityCache(
                runner_id=123,
                activities=[
                    PrefetchedActivity(
                        activity_id=1,
                        start_date=datetime(2026, 2, 1),  # Outside window
                        activity_type="Run",
                        raw_data={},
                        segment_efforts=(
                            {"segment": {"id": 999}, "elapsed_time": 300},
                        ),
                    )
                ],
            )
        }

        efforts = service._find_efforts_from_cache(
            runner, 999, datetime(2026, 1, 1), datetime(2026, 1, 15), cache
        )

        assert len(efforts) == 0

    def test_handles_missing_runner(self):
        """Returns empty list for runner not in cache."""
        service = SegmentService()
        runner = Mock(strava_id=999)
        cache = {}

        efforts = service._find_efforts_from_cache(
            runner, 999, datetime(2026, 1, 1), datetime(2026, 1, 15), cache
        )

        assert efforts == []

    def test_handles_runner_with_fetch_error(self):
        """Returns empty list for runner with fetch error."""
        service = SegmentService()
        runner = Mock(strava_id=123)

        cache = {
            123: RunnerActivityCache(
                runner_id=123,
                activities=[],
                fetch_error="Rate Limit Exceeded",
            )
        }

        efforts = service._find_efforts_from_cache(
            runner, 999, datetime(2026, 1, 1), datetime(2026, 1, 15), cache
        )

        assert efforts == []


class TestPrefetchIntegration:
    """Integration tests for prefetch flow."""

    @patch("strava_competition.services.segment_service.get_activities")
    def test_prefetch_populates_cache(self, mock_activities):
        """Prefetch retrieves activity list (details fetched lazily later)."""
        mock_activities.return_value = [
            {"id": 1, "start_date": "2026-01-10T10:00:00Z", "type": "Run"},
            {"id": 2, "start_date": "2026-01-15T10:00:00Z", "type": "Run"},
        ]

        service = SegmentService(max_workers=1)
        runner = Mock(strava_id=123, segment_team="Team A", name="Test Runner")

        cache = service._prefetch_all_activities(
            [runner],
            datetime(2026, 1, 1),
            datetime(2026, 1, 31),
        )

        assert 123 in cache
        assert len(cache[123].activities) == 2  # Two activities
        mock_activities.assert_called_once()
        # Details are NOT fetched upfront (lazy loading)
        # segment_efforts will be empty until _find_efforts_from_cache fetches them

    @patch("strava_competition.services.segment_service.get_activities")
    @patch("strava_competition.services.segment_service.time.sleep")
    def test_prefetch_uses_disk_cache_on_rate_limit(self, mock_sleep, mock_activities):
        """Falls back to disk cache when API returns 429 after retries."""
        from strava_competition.errors import StravaAPIError

        # First 3 calls raise 429, simulating rate limit
        mock_activities.side_effect = [
            StravaAPIError("429 Rate Limit Exceeded"),
            StravaAPIError("429 Rate Limit Exceeded"),
            StravaAPIError("429 Rate Limit Exceeded"),
            # Fourth call (from _load_from_disk_cache) returns cached data
            [{"id": 1, "start_date": "2026-01-10T10:00:00Z", "type": "Run"}],
        ]

        service = SegmentService(max_workers=1)
        runner = Mock(strava_id=123, segment_team="Team A", name="Test Runner")

        cache = service._prefetch_all_activities(
            [runner],
            datetime(2026, 1, 1),
            datetime(2026, 1, 31),
        )

        assert 123 in cache
        assert cache[123].from_stale_cache is True
        assert cache[123].fetch_error is None  # Has data, no error
        assert len(cache[123].activities) == 1
        # Verify retries happened with backoff
        assert mock_activities.call_count == 4  # 3 retries + 1 cache lookup
        assert mock_sleep.call_count == 2  # Sleep between retries 1→2 and 2→3

    @patch("strava_competition.services.segment_service.get_activities")
    @patch("strava_competition.services.segment_service.time.sleep")
    def test_prefetch_records_error_when_no_cache(self, mock_sleep, mock_activities):
        """Records fetch_error when API fails and no disk cache available."""
        from strava_competition.errors import StravaAPIError

        # All calls fail
        mock_activities.side_effect = StravaAPIError("429 Rate Limit Exceeded")

        service = SegmentService(max_workers=1)
        runner = Mock(strava_id=123, segment_team="Team A", name="Test Runner")

        cache = service._prefetch_all_activities(
            [runner],
            datetime(2026, 1, 1),
            datetime(2026, 1, 31),
        )

        assert 123 in cache
        assert cache[123].fetch_error is not None
        assert "429" in cache[123].fetch_error or "Rate Limit" in cache[123].fetch_error
        assert len(cache[123].activities) == 0


class TestProcessSegmentWithCache:
    """Tests for _process_segment_with_cache."""

    @patch(
        "strava_competition.services.segment_service.FORCE_ACTIVITY_SCAN_FALLBACK",
        False,
    )
    def test_processes_cached_efforts(self):
        """Processes runners from prefetch cache without API calls."""
        service = SegmentService(max_workers=1)

        runner = Mock(
            strava_id=123,
            segment_team="Team A",
            name="Test Runner",
            birthday=None,
        )
        segment = Mock(
            id=999,
            name="Test Segment",
            start_date=datetime(2026, 1, 1),
            end_date=datetime(2026, 1, 31),
            default_time_seconds=None,
            min_distance_meters=None,
            birthday_bonus_seconds=0,
        )

        cache = {
            123: RunnerActivityCache(
                runner_id=123,
                activities=[
                    PrefetchedActivity(
                        activity_id=1,
                        start_date=datetime(2026, 1, 10),
                        activity_type="Run",
                        raw_data={},
                        segment_efforts=(
                            {
                                "segment": {"id": 999},
                                "elapsed_time": 300,
                                "start_date": "2026-01-10T10:00:00Z",
                            },
                        ),
                    )
                ],
            )
        }

        results = service._process_segment_with_cache(
            segment,
            [runner],
            cache,
            1,
            1,
            None,
            None,
        )

        assert "Team A" in results
        assert len(results["Team A"]) == 1
        assert results["Team A"][0].fastest_time == 300

    def test_skips_runners_with_fetch_error(self):
        """Runners with fetch errors get default results if configured."""
        service = SegmentService(max_workers=1)

        runner = Mock(
            strava_id=123,
            segment_team="Team A",
            name="Test Runner",
            birthday=None,
        )
        segment = Mock(
            id=999,
            name="Test Segment",
            start_date=datetime(2026, 1, 1),
            end_date=datetime(2026, 1, 31),
            default_time_seconds=9999,
            min_distance_meters=None,
            birthday_bonus_seconds=0,
        )

        cache = {
            123: RunnerActivityCache(
                runner_id=123,
                activities=[],
                fetch_error="Rate Limit Exceeded",
            )
        }

        results = service._process_segment_with_cache(
            segment,
            [runner],
            cache,
            1,
            1,
            None,
            None,
        )

        # Should have default result
        assert "Team A" in results
        assert results["Team A"][0].fastest_time == 9999
        assert results["Team A"][0].source == "default_time"
