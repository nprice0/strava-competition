from datetime import datetime
import pandas as pd
import os
import tempfile

from strava_competition.excel_reader import read_distance_windows


def test_read_distance_windows_basic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "input.xlsx")
        df_dist = pd.DataFrame(
            [
                {
                    "Start Date": datetime(2024, 1, 1),
                    "End Date": datetime(2024, 1, 7),
                    "Distance Threshold (km)": 5,
                },
                {
                    "Start Date": datetime(2024, 1, 5),
                    "End Date": datetime(2024, 1, 10),
                    "Distance Threshold (km)": None,
                },
                {
                    "Start Date": datetime(2024, 1, 11),
                    "End Date": datetime(2024, 1, 9),
                    "Distance Threshold (km)": 10,
                },  # invalid inverted -> skipped
            ]
        )
        # minimal required sheets
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            pd.DataFrame(
                columns=[
                    "Segment ID",
                    "Segment Name",
                    "Start Date",
                    "End Date",
                    "Default Time",
                    "Minimum Distance (m)",
                ]
            ).to_excel(w, sheet_name="Segment Series", index=False)
            pd.DataFrame(
                columns=[
                    "Name",
                    "Strava ID",
                    "Refresh Token",
                    "Segment Series Team",
                    "Distance Series Team",
                ]
            ).to_excel(w, sheet_name="Runners", index=False)
            df_dist.to_excel(w, sheet_name="Distance Series", index=False)
        windows = read_distance_windows(path)
        assert len(windows) == 2
        assert windows[0][0] == datetime(2024, 1, 1)
        assert windows[1][1] == datetime(2024, 1, 10)
