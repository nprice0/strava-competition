"""Benchmark tests comparing old vs new approach."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from strava_competition.services.segment_service import SegmentService


@pytest.mark.benchmark
class TestAPICallReduction:
    """Verify API call reduction."""

    @patch("strava_competition.services.segment_service.SEGMENT_PREFETCH_ENABLED", True)
    @patch("strava_competition.services.segment_service.get_activities")
    @patch("strava_competition.services.segment_service.get_activity_with_efforts")
    def test_api_calls_scale_with_runners_not_segments(
        self, mock_details, mock_activities
    ):
        """New approach: API calls = O(runners), not O(runners × segments)."""
        # Track API calls
        call_count = {"activities": 0, "details": 0}

        def count_activities(*args, **kwargs):
            call_count["activities"] += 1
            return [{"id": 1}]

        def count_details(*args, **kwargs):
            call_count["details"] += 1
            return {
                "id": 1,
                "start_date": "2026-01-10T10:00:00Z",
                "type": "Run",
                "segment_efforts": [],
            }

        mock_activities.side_effect = count_activities
        mock_details.side_effect = count_details

        service = SegmentService(max_workers=1)

        # 5 runners, 4 segments
        runners = [
            Mock(strava_id=i, segment_team="A", name=f"R{i}", birthday=None)
            for i in range(5)
        ]
        segments = [
            Mock(
                id=i,
                name=f"S{i}",
                start_date=datetime(2026, 1, 1),
                end_date=datetime(2026, 1, 31),
                default_time_seconds=None,
                min_distance_meters=None,
                birthday_bonus_seconds=0,
            )
            for i in range(4)
        ]

        service.process(segments, runners)

        # Old approach would be: 5 runners × 4 segments = 20 activity calls
        # New approach: 5 runners × 1 = 5 activity calls
        assert call_count["activities"] == 5


@pytest.mark.benchmark
class TestPerformanceComparison:
    """Compare performance characteristics of prefetch vs legacy."""

    @patch("strava_competition.services.segment_service.SEGMENT_PREFETCH_ENABLED", False)
    @patch("strava_competition.services.segment_service.get_segment_efforts")
    @patch("strava_competition.services.segment_service.FORCE_ACTIVITY_SCAN_FALLBACK", False)
    def test_legacy_mode_still_works(self, mock_efforts):
        """Legacy mode functions correctly when prefetch disabled."""
        mock_efforts.return_value = [
            {"id": 1, "elapsed_time": 300, "start_date": "2026-01-10T10:00:00Z"}
        ]

        service = SegmentService(max_workers=1)

        runner = Mock(
            strava_id=123,
            segment_team="Team A",
            birthday=None,
            payment_required=False,
        )
        runner.name = "Test Runner"

        segment = Mock(
            id=999,
            start_date=datetime(2026, 1, 1),
            end_date=datetime(2026, 1, 31),
            default_time_seconds=None,
            min_distance_meters=None,
            birthday_bonus_seconds=0,
        )
        segment.name = "Test Segment"

        results = service.process([segment], [runner])

        assert "Test Segment" in results
        assert "Team A" in results["Test Segment"]
        assert len(results["Test Segment"]["Team A"]) == 1
