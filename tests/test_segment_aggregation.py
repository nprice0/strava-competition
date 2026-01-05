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
    assert {
        "Team",
        "Runner",
        "Rank",
        "Fastest Time (sec)",
        "Fastest Distance (m)",
    }.issubset(climb_df.columns)
    assert climb_df["Fastest Distance (m)"].notna().any()
    # Ranking: fastest_time -> 110 (Carl), 115 (Ben), 120 (Alice)
    ranks = {row.Runner: row.Rank for row in climb_df.itertuples() if row.Runner}
    # Validate ordering irrespective of which team produced fastest time
    assert min(ranks.values()) == 1
    assert len(ranks) == 3
    # Blank spacer rows exist between team groupings
    assert climb_df["Team"].isna().sum() == 1
    # Segment summary metadata attached to dataframe
    seg_summary = climb_df.attrs.get("segment_summary")
    assert seg_summary is not None
    expected_cols = [
        "Team",
        "Total Fastest time",
        "Gap to Leader",
        "Total Rank",
        "Fastest (Runner)",
        "Avg. Fastest Time",
        "Median Fastest Time",
        "Total Attempts",
        "Avg. Attempts",
    ]
    assert list(seg_summary.columns) == expected_cols
    assert list(seg_summary["Team"]) == ["Team B", "Team A"]
    assert seg_summary.loc[0, "Gap to Leader"] == "0:00:00"
    assert seg_summary.loc[1, "Gap to Leader"] == "0:02:10"
    assert seg_summary.loc[0, "Total Rank"] == 1
    assert seg_summary.loc[1, "Total Rank"] == 5
    # Summary sheet integrity
    summary_df = dict(outputs)["Summary"]
    assert {"Team", "Total Attempts", "Average Fastest Time (sec)"}.issubset(
        summary_df.columns
    )
    # Teams present in fixture mapping
    assert set(summary_df["Team"]) == {"Team A", "Team B", "Team C"}


def test_build_segment_outputs_no_summary(segment_results):
    outputs = build_segment_outputs(segment_results, include_summary=False)
    names = [n for n, _ in outputs]
    assert "Summary" not in names


def test_fastest_distance_zero_for_default_results():
    zero_segment = {
        "Team Z": [
            SegmentResult(
                runner="No Attempts",
                team="Team Z",
                segment="Zero Seg",
                attempts=0,
                fastest_time=150.0,
                fastest_date=None,
                fastest_distance_m=None,
            )
        ]
    }
    outputs = build_segment_outputs({"Zero Seg": zero_segment}, include_summary=False)
    df = dict(outputs)["Zero Seg"]
    row = df[df["Runner"] == "No Attempts"].iloc[0]
    assert row["Fastest Distance (m)"] == 0
