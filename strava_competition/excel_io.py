import pandas as pd

from .models import Runner, Segment


def read_segments(filepath):
    df = pd.read_excel(filepath, sheet_name="Segments")
    segments = []
    for _, row in df.iterrows():
        segments.append(
            Segment(
                id=int(row["Segment ID"]),
                name=row["Segment Name"],
                start_date=pd.to_datetime(row["Start Date"]),
                end_date=pd.to_datetime(row["End Date"]),
            )
        )
    return segments


def read_runners(filepath):
    df = pd.read_excel(filepath, sheet_name="Runners")
    runners = []
    for _, row in df.iterrows():
        runners.append(
            Runner(
                name=row["Name"],
                strava_id=int(row["Strava ID"]),
                refresh_token=row["Refresh Token"],
                team=row["Team"],
            )
        )
    return runners


def write_results(filepath, results):
    with pd.ExcelWriter(
        filepath, engine="openpyxl", datetime_format="YYYY-MM-DD HH:MM:SS"
    ) as writer:
        if not results:
            # Create a default sheet if there are no results
            df = pd.DataFrame({"Message": ["No results to display."]})
            df.to_excel(writer, sheet_name="Summary", index=False)
        else:
            used_sheet_names = set()
            for segment_name, team_data in results.items():
                rows = []
                for team, runners in team_data.items():
                    for r in runners:
                        rows.append(
                            {
                                "Team": r.team,
                                "Runner": r.runner,
                                "Attempts": r.attempts,
                                "Fastest Time (sec)": r.fastest_time,
                                "Fastest Date": r.fastest_date,
                            }
                        )
                if rows:
                    df = pd.DataFrame(rows)
                    # Normalize dates (remove timezone to satisfy Excel)
                    if "Fastest Date" in df.columns:
                        dt = pd.to_datetime(df["Fastest Date"], utc=True, errors="coerce")
                        df["Fastest Date"] = dt.dt.tz_localize(None)
                    # Ensure time column is numeric for ranking
                    if "Fastest Time (sec)" in df.columns:
                        df["Fastest Time (sec)"] = pd.to_numeric(
                            df["Fastest Time (sec)"], errors="coerce"
                        )
                        # Overall rank across the whole segment (fastest = 1)
                        # Ties share rank (competition ranking)
                        df["Rank"] = (
                            df["Fastest Time (sec)"].rank(method="min", ascending=True).astype(int)
                        )
                        # Per-team rank (useful when viewing by team); ties share rank
                        if "Team" in df.columns:
                            df["Team Rank"] = (
                                df.groupby("Team")["Fastest Time (sec)"]
                                .rank(method="min", ascending=True)
                                .astype(int)
                            )
                    # Keep existing sort: by Team, then by fastest time within the team
                    df.sort_values(by=["Team", "Fastest Time (sec)"], inplace=True)
                    # Order columns for readability if present
                    preferred_cols = [
                        col
                        for col in [
                            "Team",
                            "Runner",
                            "Rank",
                            "Team Rank",
                            "Attempts",
                            "Fastest Time (sec)",
                            "Fastest Date",
                        ]
                        if col in df.columns
                    ]
                    remaining = [c for c in df.columns if c not in preferred_cols]
                    df = df[preferred_cols + remaining]
                    # Ensure unique sheet name within 31 char limit
                    base = segment_name[:31]
                    name = base
                    i = 1
                    while name in used_sheet_names:
                        suffix = f"_{i}"
                        name = base[: 31 - len(suffix)] + suffix
                        i += 1
                    used_sheet_names.add(name)
                    df.to_excel(writer, sheet_name=name, index=False)
                else:
                    # No data for this segment, write a message
                    df = pd.DataFrame({"Message": ["No results for this segment."]})
                    base = segment_name[:31]
                    name = base if base not in used_sheet_names else f"{base[:28]}_msg"
                    used_sheet_names.add(name)
                    df.to_excel(writer, sheet_name=name, index=False)


def update_runner_refresh_tokens(filepath, runners):
    # Read the runners sheet
    df = pd.read_excel(filepath, sheet_name="Runners")
    # Update the refresh token for each runner
    for runner in runners:
        df.loc[df["Strava ID"] == runner.strava_id, "Refresh Token"] = runner.refresh_token
    # Write back to the same sheet
    with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="Runners", index=False)
