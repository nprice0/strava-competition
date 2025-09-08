from strava_competition.distance_aggregation import build_distance_outputs


def test_distance_summary_includes_all_distance_runners(distance_runners, distance_windows, distance_activity_cache):
    outputs = build_distance_outputs(distance_runners, distance_windows, distance_activity_cache)
    # Last sheet is Distance_Summary
    sheet_name, summary_rows = outputs[-1]
    assert sheet_name == "Distance_Summary"
    # Should have both runners
    names = {r["Runner"] for r in summary_rows}
    assert names == {"Alice", "Ben"}
    # Check aggregate correctness for Alice
    alice_row = next(r for r in summary_rows if r["Runner"] == "Alice")
    assert alice_row["Total Runs"] == 2
    assert alice_row["Total Distance (km)"] == round((6000.0 + 4000.0)/1000.0, 2)
    # Check threshold count appears in window sheet
    window_rows = outputs[0][1]
    assert any(k.startswith("Runs >=") for k in window_rows[0].keys())
