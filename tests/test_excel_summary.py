import pandas as pd
from pathlib import Path

from strava_competition.excel_writer import write_results

# Helper to build results mapping structure expected by write_results
# { segment_name: { team_name: [SegmentResult, ...] } }

def test_summary_sheet_aggregates(tmp_path: Path, segment_results, assert_summary_columns):
    results = segment_results
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


def test_no_summary_when_disabled(tmp_path: Path, segment_results):
    results = segment_results
    out = tmp_path / "output_no_summary.xlsx"
    write_results(out, results, include_summary=False)
    book = pd.read_excel(out, sheet_name=None)
    assert "Summary" not in book, "Summary sheet should not be present when include_summary=False"
