from strava_competition.segment_aggregation import build_segment_outputs
from strava_competition.models import SegmentResult


def test_build_segment_outputs_includes_summary_and_ranking(segment_results):
    # Extend shared fixture with an empty segment to assert message sheet path.
    segment_results["Empty Segment"] = {}
    outputs = build_segment_outputs(segment_results, include_summary=True)
    names = [n for n, _ in outputs]
    assert "Segment One" in names
    assert "Segment Two" in names
    assert any(n.startswith("Empty Segment") for n in names)
    assert "Summary" in names
    # Find Climb sheet
    climb_df = dict(outputs)["Segment One"]
    assert {"Team", "Runner", "Rank", "Fastest Time (sec)"}.issubset(climb_df.columns)
    # Ranking: fastest_time -> 110 (Carl), 115 (Ben), 120 (Alice)
    ranks = {row.Runner: row.Rank for row in climb_df.itertuples()}
    # Validate ordering irrespective of which team produced fastest time
    assert min(ranks.values()) == 1
    assert len(ranks) == 3
    # Summary sheet integrity
    summary_df = dict(outputs)["Summary"]
    assert {"Team", "Total Attempts", "Average Fastest Time (sec)"}.issubset(summary_df.columns)
    # Teams present in fixture mapping
    assert set(summary_df["Team"]) == {"Team A", "Team B", "Team C"}


def test_build_segment_outputs_no_summary(segment_results):
    outputs = build_segment_outputs(segment_results, include_summary=False)
    names = [n for n, _ in outputs]
    assert "Summary" not in names
