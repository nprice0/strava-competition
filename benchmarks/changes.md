# Investigation: Out-of-Tolerance Segment Timings (±3 s)

## Overview

- Compared `competition_results_actual.xlsx` (baseline) against `competition_results_20251105_223829.xlsx` via `strava_competition.excel_diff`.
- No structural workbook diffs. Five runner/segment timings exceeded the ±3 s tolerance: four on SS25-02 (Fiveways to Hell) and one on "It's a ball that comes out a cannon".
- All other per-runner deltas remain ≤1.5 s.

## Findings by runner & segment

### SS25-02 – Fiveways to Hell (segment id 38987500)

- **Andrew Knott** — baseline 1 898 s vs candidate 1 924.16 s (+26.16 s).

  - Fastest activity (id 14547101577, 2025‑05‑21) from Strava effort fails fallback similarity: discrete Fréchet distance 528.79 m > adaptive threshold 77.93 m despite coverage ratio 1.0 and gate offsets ≤33 m.
  - Fallback therefore accepts slower activity 14576321774 (2025‑05‑24) with matched time 1 924 s.
  - Root cause: single GPS spike (~30 m) near gate inflates the Fréchet score enough to reject the fastest run.

- **Luke Sibieta** — baseline 1 729 s vs candidate 1 769.81 s (+40.81 s).

  - Fastest attempt (activity 14618107187, 2025‑05‑28) rejected: Fréchet 533.35 m with raw offset 404 m caused by a large GPS jump mid-segment.
  - Matcher falls back to activity 14587060735 (2025‑05‑25) at 1 769.8 s.
  - Root cause mirrors Andrew: outlier GPS samples push similarity above threshold.

- **Neil Price** — baseline 1 893 s vs candidate 2 191 s (+298 s).

  - Candidate output comes from activity 14624467801 (2025‑05‑28) where fallback gating spans entry index 7 → exit index 2 198; elapsed 2 191 s.
  - Best Strava effort (activity 14661937369, 2025‑06‑01) does match geometrically, but timing indices 1 883 → 4 549 yield 2 684 s, so the matcher selects the shorter (still slow) 2 191 s attempt.
  - Root cause: gate crossing detection chooses a late finish after the runner loops back over the segment, inflating elapsed time by ~13 min.

- **Ben Wernick** — baseline 1 579 s vs candidate `—` (no fallback match recorded).
  - Fastest attempt (activity 14422381338, 2025‑05‑09) rejected: Fréchet 531.80 m; raw offset 1.5 km indicates a sizeable detour or GPS dropout mid-run.
  - Without a successful match, the sheet renders `—` for the fallback time.
  - Root cause: similarity filter discards the only activity with the 1 579 s effort because of a large off-segment excursion.

### It's a ball that comes out a cannon (segment id 38955187)

- **Ben Wernick** — baseline 619 s vs candidate 1 127.31 s (+508.31 s).
  - Fallback picks activity 14387692537 (2025‑05‑05) with entry 16 → exit 1 142, elapsed 1 127 s.
  - The actual fastest activity (14783453586, 2025‑06‑13) matches geometrically but timing indices 601 → 2 811 produce 3 117 s, so the matcher favors the smaller-yet-still-inflated 1 127 s run.
  - Root cause: the runner re-crosses start/finish multiple times during a long workout; the timing estimator locks onto a later pass rather than the first finish, stretching elapsed time by >8 minutes.

## Recommended remediations

1. **Robust similarity scoring**: before computing Fréchet, drop interior samples with offsets well above `threshold_filtered_max_offset_m` (or clamp with a percentile filter). This salvages otherwise valid attempts with isolated GPS spikes while keeping Luke/Andrew/Ben in tolerance and avoids loosening thresholds globally.
2. **Gate-crossing selection**: enhance `estimate_segment_time` to prefer the first valid start/finish pair after direction gating (e.g. constrain with monotonic coverage span or a max idle gap). This limits elapsed time inflation for Neil and Ben while preserving runners who do not loop.
3. **Regression safeguards**: add unit tests and synthetic GPS fixtures covering (a) large mid-run spikes, (b) multi-pass gate crossings, and (c) treadmill/indoor activities so future tweaks do not push unaffected athletes further from baseline.

Implementing the above should recover the five outliers while keeping the remaining runners at or below their current deltas. No code changes were made in this investigation.
