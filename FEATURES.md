# Features Guide

[README](README.md) · **[Features Guide](FEATURES.md)**

---

This guide explains each feature in detail and how to configure them in your workbook.

---

## Contents

- [Segment Series](#segment-series)
  - [Split Windows](#split-windows)
  - [Birthday Bonus](#birthday-bonus)
  - [Time Bonus](#time-bonus)
  - [Minimum Distance Filter](#minimum-distance-filter)
  - [Default Time](#default-time)
- [Distance Series](#distance-series)
- [CLI Tools](#cli-tools)
- [Workbook Reference](#workbook-reference)

---

## Segment Series

Segment competitions track who's fastest on your club's favourite Strava segments. For each segment you define, the app fetches every runner's efforts within the date window and ranks them by time.

### How it works

1. Define segments in the `Segment Series` sheet with a Strava segment ID and date range
2. The app fetches each runner's efforts for that segment within the window
3. Results are ranked by elapsed time (fastest first)
4. Team totals are calculated if runners have teams assigned

The output includes one sheet per segment showing:

- Runner name and team
- Elapsed time (with any bonuses applied)
- Number of attempts
- Whether a birthday bonus was applied

---

### Split Windows

Run one segment across multiple date windows and show each runner's best effort across all windows.

#### When to use this

- **Multi-week competitions** – e.g. "Hill Climb Challenge" running for 6 weeks, with runners able to attempt any week
- **Seasonal events** – compare best times across spring, summer, and autumn windows
- **Promotional periods** – give runners multiple chances to post their best time

#### How to configure

Add multiple rows with the same **Segment ID**. Each row becomes a separate window.

| Segment ID | Segment Name      | Start Date | End Date   | Window Label |
| ---------- | ----------------- | ---------- | ---------- | ------------ |
| 12345678   | Hill Climb Sprint | 2026-01-01 | 2026-01-15 | Week 1       |
| 12345678   | Hill Climb Sprint | 2026-01-16 | 2026-01-31 | Week 2       |
| 12345678   | Hill Climb Sprint | 2026-02-01 | 2026-02-14 | Final Push   |

**Key points:**

- Rows are grouped by Segment ID (not name)
- All rows with the same ID must have the same Segment Name
- The runner's fastest time across all windows appears in the output
- Total attempts across all windows are shown
- Window Label is optional but helps identify periods

#### Configuration toggle

Set `SEGMENT_SPLIT_WINDOWS_ENABLED` in your environment:

- **Enabled (default)**: Best time across all windows → one output sheet per segment
- **Disabled**: Each window → separate sheet

---

### Birthday Bonus

Give runners a time advantage on their birthday. If a runner completes a segment effort on their birthday, the configured bonus is deducted from their elapsed time.

#### How to configure

1. In the `Runners` sheet, add each runner's birthday in `dd-MMM` format (e.g. `15-Jan`, `07-May`)
2. In the `Segment Series` sheet, set `Birthday Bonus (secs)` to the number of seconds to deduct

#### Example

| Runner | Birthday |
| ------ | -------- |
| Alice  | 15-Jan   |

| Segment ID | Birthday Bonus (secs) |
| ---------- | --------------------- |
| 12345678   | 30                    |

If Alice completes the segment in 120 seconds on 15th January, her adjusted time becomes **90 seconds**.

#### Notes

- The bonus is based on the effort's start time (when the runner entered the segment)
- Different windows can have different bonus values
- Leave blank or set to 0 for no bonus
- The output shows whether a birthday bonus was applied

---

### Time Bonus

Apply a time adjustment to **all runners** who complete an effort within a specific window. Unlike birthday bonus, this applies to everyone—not just those on their birthday.

#### How to configure

In the `Segment Series` sheet, add the `Time Bonus (secs)` column:

| Segment ID | Window Label  | Time Bonus (secs) |
| ---------- | ------------- | ----------------- |
| 12345678   | Christmas Day | 30                |
| 12345678   | New Year      | -10               |

- **Positive values** subtract time (reward/bonus)
- **Negative values** add time (penalty)
- Leave empty for no adjustment

#### Use cases

- **Incentivise unlikely days** – 30-second bonus for running on Christmas Day
- **Short promotional windows** – bonus for a 1-hour window on New Year's Eve
- **Compensate for conditions** – penalty for a course diversion or adverse weather

#### Example

A window with `Time Bonus (secs) = 15` means all runners in that window get 15 seconds deducted. A value of `-5` adds 5 seconds (penalty).

#### Notes

- Stacks with birthday bonus (time bonus applied after birthday bonus)
- Decimal values supported (e.g. `5.5`)
- Adjusted time floors at 0 seconds (cannot go negative)
- Does **not** apply to default time—only actual efforts

---

### Minimum Distance Filter

Filter out short efforts that don't cover the full segment. Useful when Strava records partial efforts due to GPS issues or cutting corners.

#### How to configure

In the `Segment Series` sheet, set `Minimum Distance (m)`:

| Segment ID | Minimum Distance (m) |
| ---------- | -------------------- |
| 12345678   | 500                  |

Only efforts where the runner covered at least 500 metres count toward rankings.

#### Notes

- Leave blank or set to 0 to disable filtering
- Distance comes from Strava's effort payload
- Useful for segments where GPS drift might create short phantom efforts

---

### Default Time

Assign a fallback time for runners who don't complete the segment. This ensures rankings always account for the full roster.

#### How to configure

In the `Segment Series` sheet, set `Default Time`:

| Segment ID | Default Time |
| ---------- | ------------ |
| 12345678   | 00:10:00     |

Accepts `HH:MM:SS`, Excel time values, or raw seconds.

#### Notes

- Runners with no recorded effort receive this time
- Useful for handicap-style competitions where everyone must be ranked
- Time bonuses do **not** apply to default times

---

## Distance Series

Track cumulative running distance over time windows. This is separate from segment competitions and doesn't require specific Strava segments.

### How it works

1. Define date windows in the `Distance Series` sheet
2. The app fetches each runner's activities within the window
3. Distances from qualifying activity types (runs by default) are totalled
4. Results show per-runner and per-team totals

### How to configure

Create a `Distance Series` sheet:

| Start Date | End Date   | Distance Threshold (km) |
| ---------- | ---------- | ----------------------- |
| 2026-01-01 | 2026-01-31 | 50                      |
| 2026-02-01 | 2026-02-28 |                         |

- **Distance Threshold** is optional—only runners who meet or exceed this distance appear in results
- Leave threshold blank to include all runners

### Team participation

Runners need the `Distance Series Team` column populated in the `Runners` sheet to participate. Leave it blank and that runner is excluded from distance competitions.

---

## CLI Tools

The app includes several helper tools for debugging, exporting data, and maintenance. See the [README](README.md) for command examples.

### fetch_runner_segment_efforts

View a runner's activities with full segment effort details. Useful for debugging why an effort isn't appearing in results or inspecting the raw data Strava returns for a specific date range.

### fetch_activity_gps

Download the GPS track for any Strava activity and save it as a GPX file. The exported file can be opened in mapping applications, uploaded to GPS devices, or used for route analysis. Includes altitude, time, and distance data by default.

### fetch_segment_gpx

Export a Strava segment's route as a GPX file. Useful for sharing segment routes with others or importing into GPS devices and mapping apps for navigation.

### clip_activity_segment

Extract just the segment portion from a full activity's GPS track. Given an activity and segment, this tool slices out the relevant track points and saves them as a new GPX file. Supports multiple input methods: automatic lookup from Strava, manual index positions, timestamps, or cached data.

### deviation_map

Generate an interactive HTML map showing how a runner's actual route compares to a segment. Highlights gate crossings and any large deviations from the expected path. Useful for understanding why times differ between runners or checking segment accuracy.

### capture_gc

Clean up old cached API responses to free disk space. Deletes cache files older than a specified retention window while preserving recent data. Supports a dry-run mode to preview what would be deleted before committing.

---

## Workbook Reference

Quick reference for all sheet columns. The "Required" column indicates whether the column header must exist in the sheet—values can still be left blank where noted.

### Runners sheet

| Column               | Required | Description                                                       |
| -------------------- | -------- | ----------------------------------------------------------------- |
| Name                 | Yes      | Display name                                                      |
| Strava ID            | Yes      | The athlete's Strava ID                                           |
| Refresh Token        | Yes      | OAuth refresh token                                               |
| Segment Series Team  | Yes      | Team for segment competitions (blank = skip)                      |
| Distance Series Team | Yes      | Team for distance competitions (blank = skip)                     |
| Birthday (dd-MMM)    | Yes      | Birthday for bonus calculations, e.g. `07-May` (blank = no bonus) |

### Segment Series sheet

| Column                | Required | Description                                             |
| --------------------- | -------- | ------------------------------------------------------- |
| Segment ID            | Yes      | Strava segment ID                                       |
| Segment Name          | Yes      | Display name                                            |
| Start Date            | Yes      | Window start datetime (Excel datetime or ISO string)    |
| End Date              | Yes      | Window end datetime                                     |
| Window Label          | No       | Label for split windows (e.g. "Week 1")                 |
| Default Time          | Yes      | Fallback time for runners with no effort (blank = none) |
| Minimum Distance (m)  | Yes      | Minimum effort distance to qualify (blank = 0)          |
| Birthday Bonus (secs) | Yes      | Seconds deducted for birthday efforts (blank = 0)       |
| Time Bonus (secs)     | No       | Seconds added/subtracted for all runners                |

### Distance Series sheet

| Column                  | Required | Description                                       |
| ----------------------- | -------- | ------------------------------------------------- |
| Start Date              | Yes      | Window start datetime                             |
| End Date                | Yes      | Window end datetime                               |
| Distance Threshold (km) | Yes      | Minimum distance to qualify (blank = include all) |

---

## Tips

- **Sheet names are case-sensitive** – use exactly `Runners`, `Segment Series`, and `Distance Series`
- **Datetime formats** – Excel datetimes or ISO strings both work
- **Leave optional columns blank** rather than deleting them
- **Test with a small group first** before running a full competition
