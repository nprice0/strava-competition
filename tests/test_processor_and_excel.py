from datetime import datetime
import os
import tempfile

import pandas as pd
import pytest

from strava_competition.models import Runner, Segment
from strava_competition import excel_io
from strava_competition import processor


def make_input_workbook(path):
    segs = pd.DataFrame(
        [
            {
                "Segment ID": 101,
                "Segment Name": "Hill Climb",
                "Start Date": datetime(2024, 1, 1),
                "End Date": datetime(2024, 1, 31),
            }
        ]
    )
    runners = pd.DataFrame(
        [
            {"Name": "Alice", "Strava ID": 1, "Refresh Token": "rt1", "Team": "Red"},
            {"Name": "Ben", "Strava ID": 2, "Refresh Token": "rt2", "Team": "Blue"},
            {"Name": "Carl", "Strava ID": 3, "Refresh Token": "rt3", "Team": "Red"},
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        segs.to_excel(w, sheet_name="Segments", index=False)
        runners.to_excel(w, sheet_name="Runners", index=False)


def test_read_and_write_roundtrip_and_ranks(monkeypatch):
    # Create a temporary input workbook
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.xlsx")
        out_path = os.path.join(tmpdir, "out.xlsx")
        make_input_workbook(in_path)

        # Read inputs using our helpers
        segments = excel_io.read_segments(in_path)
        runners = excel_io.read_runners(in_path)
        assert len(segments) == 1 and len(runners) == 3

        # Mock the API to return deterministic efforts per runner
        def fake_get_efforts(runner, segment_id, start_date, end_date):
            data = {
                "Alice": [
                    {"elapsed_time": 120, "start_date_local": "2024-01-10T09:00:00Z"},
                    {"elapsed_time": 115, "start_date_local": "2024-01-15T09:00:00Z"},
                ],
                "Ben": [
                    {"elapsed_time": 118, "start_date_local": "2024-01-12T09:00:00Z"},
                ],
                "Carl": [
                    {"elapsed_time": 115, "start_date_local": "2024-01-14T09:00:00Z"},
                    {"elapsed_time": 112, "start_date_local": "2024-01-20T09:00:00Z"},
                ],
            }
            return data.get(runner.name, [])

        # Patch the symbol used in processor module
        monkeypatch.setattr(processor, "get_segment_efforts", fake_get_efforts)

        # Process and write results (remain inside temp dir context)
        results = processor.process_segments(segments, runners, max_workers=2)
        assert "Hill Climb" in results
        # Write results to Excel
        excel_io.write_results(out_path, results)
        assert os.path.exists(out_path)

        # Read back the written Excel to check ranks and content
        df = pd.read_excel(out_path, sheet_name="Hill Climb")
        # Expect 3 rows, with Rank based on Fastest Time (sec)
        assert set(["Team", "Runner", "Rank", "Team Rank", "Attempts", "Fastest Time (sec)", "Fastest Date"]).issubset(
            set(df.columns)
        )
        assert len(df) == 3
        # Fastest times: Carl=112, Ben=118, Alice=115 -> ranks 1,2,2 or 1,3,2 depending on ties
        # Since Carl 112 (Red), Alice 115 (Red), Ben 118 (Blue)
        # Overall ranks should be 1 (Carl), 2 (Alice), 3 (Ben)
        df_sorted = df.sort_values("Runner").reset_index(drop=True)
        runner_to_time = dict(zip(df_sorted["Runner"], df_sorted["Fastest Time (sec)"]))
        assert runner_to_time["Carl"] == 112
        assert runner_to_time["Alice"] == 115
        assert runner_to_time["Ben"] == 118
        runner_to_rank = dict(zip(df_sorted["Runner"], df_sorted["Rank"]))
        assert runner_to_rank["Carl"] == 1
        assert runner_to_rank["Alice"] == 2
        assert runner_to_rank["Ben"] == 3

        # Team ranks: Red team has Carl(1) and Alice(2) within team; Blue has Ben(1) within team
        team_rank = dict(zip(df_sorted["Runner"], df_sorted["Team Rank"]))
        assert team_rank["Carl"] == 1
        assert team_rank["Alice"] == 2
        assert team_rank["Ben"] == 1


def test_update_runner_refresh_tokens_roundtrip(tmp_path):
    in_path = tmp_path / "input.xlsx"
    segs = pd.DataFrame([
        {"Segment ID": 1, "Segment Name": "S1", "Start Date": datetime(2024,1,1), "End Date": datetime(2024,1,2)}
    ])
    runners_df = pd.DataFrame([
        {"Name": "A", "Strava ID": 10, "Refresh Token": "rt-old", "Team": "X"}
    ])
    with pd.ExcelWriter(in_path, engine="openpyxl") as w:
        segs.to_excel(w, sheet_name="Segments", index=False)
        runners_df.to_excel(w, sheet_name="Runners", index=False)

    runners_objs = excel_io.read_runners(str(in_path))
    assert runners_objs[0].refresh_token == "rt-old"
    runners_objs[0].refresh_token = "rt-new"
    excel_io.update_runner_refresh_tokens(str(in_path), runners_objs)
    df_after = pd.read_excel(in_path, sheet_name="Runners")
    assert df_after.loc[df_after["Strava ID"] == 10, "Refresh Token"].iloc[0] == "rt-new"
