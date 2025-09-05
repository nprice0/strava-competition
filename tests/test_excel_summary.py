import pandas as pd
from pathlib import Path

from strava_competition.excel_io import write_results
from strava_competition.models import SegmentResult

# Helper to build results mapping structure expected by write_results
# { segment_name: { team_name: [SegmentResult, ...] } }

def _build_results():
    return {
        "Segment One": {
            "Team A": [
                SegmentResult(runner="Runner1", team="Team A", segment="Segment One", attempts=2, fastest_time=100, fastest_date="2025-01-01T10:00:00"),
                SegmentResult(runner="Runner2", team="Team A", segment="Segment One", attempts=1, fastest_time=120, fastest_date="2025-01-01T10:05:00"),
            ],
            "Team B": [
                SegmentResult(runner="Runner3", team="Team B", segment="Segment One", attempts=3, fastest_time=90, fastest_date="2025-01-01T09:50:00"),
            ],
        },
        "Segment Two": {
            "Team A": [
                SegmentResult(runner="Runner1", team="Team A", segment="Segment Two", attempts=1, fastest_time=110, fastest_date="2025-01-02T11:00:00"),
            ],
            # Team B has no participation on this segment (omitted entirely)
            "Team C": [
                SegmentResult(runner="Runner4", team="Team C", segment="Segment Two", attempts=2, fastest_time=80, fastest_date="2025-01-02T11:15:00"),
            ],
        },
    }


def test_summary_sheet_aggregates(tmp_path: Path):
    results = _build_results()
    out = tmp_path / "output.xlsx"
    write_results(out, results, include_summary=True)

    # Load all sheets
    book = pd.read_excel(out, sheet_name=None)
    assert "Summary" in book, "Summary sheet missing when include_summary=True"
    summary = book["Summary"].copy()
    # Make sure expected columns present
    expected_cols = {
        "Team",
        "Runners Participating",
        "Segments With Participation",
        "Total Attempts",
        "Average Fastest Time (sec)",
        "Total Fastest Times (sec)",
    }
    missing = expected_cols - set(summary.columns)
    assert not missing, f"Missing columns in summary: {missing}"

    summary.set_index("Team", inplace=True)

    # Team A expectations
    assert summary.loc["Team A", "Runners Participating"] == 2
    assert summary.loc["Team A", "Segments With Participation"] == 2
    assert summary.loc["Team A", "Total Attempts"] == 4  # 2 + 1 + 1
    assert summary.loc["Team A", "Average Fastest Time (sec)"] == 110.0
    # Sum of fastest times for Team A: 100 + 120 + 110 = 330
    assert summary.loc["Team A", "Total Fastest Times (sec)"] == 330.0

    # Team B expectations
    assert summary.loc["Team B", "Runners Participating"] == 1
    assert summary.loc["Team B", "Segments With Participation"] == 1
    assert summary.loc["Team B", "Total Attempts"] == 3
    assert summary.loc["Team B", "Average Fastest Time (sec)"] == 90.0
    assert summary.loc["Team B", "Total Fastest Times (sec)"] == 90.0

    # Team C expectations
    assert summary.loc["Team C", "Runners Participating"] == 1
    assert summary.loc["Team C", "Segments With Participation"] == 1
    assert summary.loc["Team C", "Total Attempts"] == 2
    assert summary.loc["Team C", "Average Fastest Time (sec)"] == 80.0
    assert summary.loc["Team C", "Total Fastest Times (sec)"] == 80.0


def test_no_summary_when_disabled(tmp_path: Path):
    results = _build_results()
    out = tmp_path / "output_no_summary.xlsx"
    write_results(out, results, include_summary=False)
    book = pd.read_excel(out, sheet_name=None)
    assert "Summary" not in book, "Summary sheet should not be present when include_summary=False"
