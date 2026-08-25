from strava_competition.distance_aggregation import (
    FETCH_FAILED_MARKER,
    build_distance_outputs,
)
from typing import Any


def test_distance_summary_includes_all_distance_runners(
    distance_runners: Any, distance_windows: Any, distance_activity_cache: Any
) -> None:
    outputs = build_distance_outputs(
        distance_runners, distance_windows, distance_activity_cache
    )
    # Last sheet is Distance_Summary
    sheet_name, summary_rows = outputs[-1]
    assert sheet_name == "Distance_Summary"
    # Should have both runners
    names = {r["Runner"] for r in summary_rows}
    assert names == {"Alice", "Ben"}
    # Check aggregate correctness for Alice
    alice_row = next(r for r in summary_rows if r["Runner"] == "Alice")
    assert alice_row["Total Runs"] == 2
    assert alice_row["Total Distance (km)"] == round((6000.0 + 4000.0) / 1000.0, 2)
    # Check threshold count appears in window sheet
    window_rows = outputs[0][1]
    assert any(k.startswith("Runs >=") for k in window_rows[0].keys())


def test_failed_runners_marked_in_window_and_summary_sheets(
    distance_runners: Any, distance_windows: Any, distance_activity_cache: Any
) -> None:
    """Runners in the failed set show FETCH FAILED instead of 0 runs."""
    outputs = build_distance_outputs(
        distance_runners,
        distance_windows,
        distance_activity_cache,
        failed_runner_names={"Ben"},
    )
    for sheet_name, rows in outputs:
        by_runner = {r["Runner"]: r for r in rows}
        runs_key = "Total Runs" if sheet_name == "Distance_Summary" else "Runs"
        assert by_runner["Ben"][runs_key] == FETCH_FAILED_MARKER
        assert isinstance(by_runner["Alice"][runs_key], int)


def test_failed_runner_set_defaults_to_empty(
    distance_runners: Any, distance_windows: Any, distance_activity_cache: Any
) -> None:
    """Omitting the failed set keeps the legacy behaviour (no markers)."""
    outputs = build_distance_outputs(
        distance_runners, distance_windows, distance_activity_cache
    )
    for _sheet_name, rows in outputs:
        assert all(row.get("Runs") != FETCH_FAILED_MARKER for row in rows)
        assert all(row.get("Total Runs") != FETCH_FAILED_MARKER for row in rows)
