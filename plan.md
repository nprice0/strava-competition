# Segment Matching Implementation Plan

> Goal: add a reusable GPS-based segment matcher that only runs when Strava does not return official segment effort times (i.e., non-subscribed runners) while keeping the rest of the codebase clean and testable.

## Step 1 — Dependencies and Environment ✅

- Updated `requirements.txt` to include `numpy`, `scipy`, `pyproj`, `shapely`, `geopy`, and `polyline` alongside existing dependencies. Confirmed they are compatible with the current Python tooling.
- Reminder for later steps: consider import guards if we need to keep optional installs lightweight.

## Step 2 — Package Layout ✅

- Added the `strava_competition/matching/` package with module stubs: `__init__.py`, `fetchers.py`, `preprocessing.py`, `similarity.py`, `timing.py`, and `models.py`.
- Defined core dataclasses (`SegmentGeometry`, `ActivityTrack`, `MatchResult`, `Tolerances`) and placeholder functions so later steps can fill in logic without changing imports.
- Existing service modules remain untouched and will integrate with the new package in later steps.

## Step 3 — Segment and Activity Data Acquisition ✅

- Added resilient helpers in `strava_competition/strava_api.py` (`fetch_segment_geometry`, `fetch_activity_stream`) that reuse the shared limiter, surface rich error types, and normalize Strava responses (polyline, distance, lat/lon/time samples).
- Introduced new Strava-specific error classes in `errors.py` to distinguish permission issues, subscription limits, missing resources, and empty streams.
- Implemented `matching/fetchers.py` so callers receive ready-to-use `SegmentGeometry` and `ActivityTrack` dataclasses with sanitized metadata while keeping point decoding for the preprocessing step.

## Step 4 — Preprocessing Pipeline ✅

- Implemented decoding, reprojection, simplification, and distance-based resampling utilities in `matching/preprocessing.py`, using `polyline`, `pyproj`, `shapely`, and `numpy` under robust error handling.
- Added dataclasses (`PreparedSegmentGeometry`, `PreparedActivityTrack`) plus `prepare_geometry` and `prepare_activity` helpers that return reusable metric artefacts, including local CRS transformers for caching.
- Helper functions now produce consistent `(n, 2)` metric arrays, enforce tolerance/interval guards, and gracefully handle empty or degenerate inputs.

## Step 5 — Direction and Coverage Validation ✅

- Implemented `matching/validation.py` with `check_direction` and `compute_coverage` helpers returning structured results and leveraging vectorised numpy operations.
- Added dataclasses (`DirectionCheckResult`, `CoverageResult`) capturing diagnostics like max start distance, direction score, and coverage bounds for downstream decisions.
- Projection helper now maps activity points to cumulative distances along the segment and clamps results for stable diagnostics.

## Step 6 — Similarity Scoring ✅

- Implemented `discrete_frechet_distance` using a dynamic-programming accumulator over metric-space coordinates so we can quantify leash-length deviations deterministically.
- Added `windowed_dtw` with a Sakoe-Chiba style window guard, giving us an optional elasticity measure when Fréchet distance exceeds tolerance but alignment is still plausible.
- Wired `SegmentCache` to store `PreparedSegmentGeometry` alongside a lazily constructed shapely `LineString` via `build_cached_segment`, ensuring downstream orchestration reuses preprocessing artefacts efficiently.

## Step 7 — Segment Time Computation ✅

- Added `SegmentTimingEstimate` dataclass and `estimate_segment_time` implementation in `matching/timing.py`, working off prepared activity/segment geometry for consistent metric projections.
- Project activity samples onto the segment polyline, interpolate entry/exit timestamps against coverage bounds, and produce elapsed seconds along with indices/timestamps for diagnostics.
- Reused shared helpers to keep interpolation and projection logic deterministic and ready for orchestration in the next step.

## Step 8 — Main Matching Orchestration ✅

- Implemented `match_activity_to_segment` in `matching/__init__.py` to orchestrate fetch → preprocessing → validation → similarity → timing, returning a structured `MatchResult`.
- Introduced cache-aware `_get_prepared_segment`, activity preparation helper, and DTW window calculator while surfacing diagnostics (cache hits, direction/coverage metrics, similarity scores, timing breakdown).
- Ensured the function only performs logging side effects and gracefully handles failure cases by returning unmatched results with descriptive diagnostics for integration layers.

## Step 9 — Integration with Existing Services ✅

- Extended `services/segment_service.py` so it still prioritises official Strava efforts but invokes `match_activity_to_segment` for runners who hit 402 responses or are pre-flagged as payment-required, including a follow-up activity fetch window.
- Added helper workflow to queue fallback runners, dedupe retries, and merge matcher-derived timings (with source + diagnostics) into `SegmentResult` objects without disrupting existing aggregation or progress reporting.
- Updated `SegmentResult` model to capture optional fastest dates plus metadata describing the origin of each time so downstream reporting can distinguish Strava vs. fallback matches.

## Step 10 — Configuration Management ✅

- Added matcher-specific constants in `config.py` so simplification, resampling, start/frechet tolerances, and coverage threshold can be tuned via environment variables without code edits.
- Updated `match_activity_to_segment` to source its default `Tolerances` from the configuration module, keeping optional overrides intact but centralising project-wide defaults.
- Documented the new env hooks for future plan steps (testing + docs) to emphasise how operators can experiment safely.

## Step 11 — Testing Strategy ✅

- Added `tests/test_matching_components.py` with unit coverage for preprocessing, validation, similarity metrics, and timing interpolation using synthetic geometries.
- Exercised `match_activity_to_segment` via monkeypatched fetchers to confirm matched and failure paths surface the expected diagnostics.
- Ensured the matcher cache resets between tests and verified the suite passes under the project dependencies.

## Step 12 — Performance and Caching

- Benchmark the matcher with activities containing 10k+ points; adjust resampling interval or introduce multi-stage simplification if runtime exceeds targets.
- Persist segment cache within a request scope (in-memory for now). Optionally add a lightweight disk cache if re-running from CLI.
- Avoid per-point Python loops; rely on `numpy` vectorization where possible.

## Step 13 — Operational Guidance

- Add structured logging around preprocessing counts, Fréchet score, coverage, and elapsed time for debugging.
- Update `README.md` with:
  - Required Strava OAuth scopes (`read_all`, `activity:read_all`, `profile:read_all`).
  - Explanation of the fallback mechanism for non-subscribed athletes.
  - Instructions for enabling/disabling the matcher via configuration.
- Document known limitations (extreme GPS noise, pauses mid-segment, unsupported sports).
