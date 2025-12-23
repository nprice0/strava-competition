# Segment Split Windows Implementation Plan

This document outlines the work required to support "Segment Split Windows": the ability to define multiple date windows for a single segment and compute each runner's best effort across all configured windows.

---

## 1. Requirements Clarification

1. Clarify workbook UX:
   - Multiple windows will be expressed by duplicating segment rows (with optional `Window Label`); ensure workbook templates/docs make this crystal clear.
   - Decide whether split windows share a single segment sheet output or optionally produce per-window sheets plus an aggregate.
2. Confirm scoring rules:
   - Only the best attempt per runner will be visible in the aggregated sheet (raw per-window attempts remain internal diagnostics).
   - Ranks are computed from aggregated best times only (per-window placements remain implicit in diagnostics).
3. Validate interaction with birthday bonus and default times.
4. Determine replay/capture needs when reprocessing the same segment over overlapping windows.

## 2. Workbook Schema Updates

1. Extend the `Segment Series` sheet format to capture multiple windows by duplicating segment rows with the same ID/name (grouping key) and optionally adding a `Window Label` column for human-friendly tags that can be reused across segments (e.g., "Final Push Week").
2. Update `excel_reader.py` validations:
   - Parse new structure into an in-memory model: `SegmentSplitWindow` objects keyed by segment.
   - Enforce contiguous/non-overlapping windows if required.
3. Document the new schema in README and provide sample workbook rows.

## 3. Data Model & Config Changes

1. Introduce new model classes in `models.py`, e.g., `SegmentWindow` and `SegmentSplitConfig`.
2. Update config to toggle the feature (`SEGMENT_SPLIT_WINDOWS_ENABLED`) and optionally cap window counts.
3. Adjust `ResultsMapping` definitions (if necessary) to distinguish base segments vs. split aggregates.

## 4. Service Layer Enhancements

1. Modify `segment_service.SegmentService` to accept multiple windows per segment:
   - Iterate per window, fetching efforts bounded by that window.
   - Cache and reuse activities across windows to avoid duplicate API calls (consider per-runner global cache keyed by date ranges).
2. Combine per-window results:
   - After processing all windows, compute each runner's best adjusted time (apply birthday bonus per effort).
   - Store diagnostics referencing the window that produced the best effort.
3. Provide optional per-window detail rows (for debugging) controlled by config flag.

## 5. Aggregation & Writer Updates

1. Extend `segment_aggregation.build_segment_outputs` to:
   - Generate one primary sheet per split segment with aggregated best times.
   - Optionally append per-window summaries (e.g., sub-table listing each window's fastest runner).
2. Update `excel_writer` to handle new attributes (e.g., include window name/date in summary tables).
3. Ensure birthday highlight logic still applies to the aggregated rows.

## 6. CLI & Tooling Alignment

1. Update CLI helpers (e.g., `fetch_runner_segment_efforts`) to support split window diagnostics.
2. Provide a command to validate workbook split window definitions (dry-run parser that logs window counts and overlaps).

## 7. Backward Compatibility & Migration

1. Ensure feature is opt-in; existing workbooks without split windows continue to run unchanged.
2. Add migration guidance: sample workbook templates, instructions for duplicating rows, etc.
3. Consider auto-detecting split windows to warn if a segment appears multiple times without grouping metadata.

## 8. Testing Strategy

1. Unit tests:
   - `excel_reader` parsing of multiple windows.
   - `segment_service` logic combining efforts across windows (mock Strava data).
   - Aggregation outputs (sheet names, summaries, birthday highlighting).
2. Integration tests:
   - End-to-end scenario with capture data covering two windows and verifying best-time selection.
3. Regression tests to ensure non-split segments behave exactly as before.

## 9. Documentation & Samples

1. Update README with a dedicated "Segment Split Windows" section.
2. Provide a sample workbook (under `samples/` or tests fixtures) demonstrating three windows.
3. Document config flags and CLI helpers supporting the feature.

## 10. Rollout Considerations

1. Provide feature flag toggle for staged rollout.
2. Communicate to users (release notes) about schema change and required workbook updates.
3. Monitor runtime performance—multiple windows increase Strava calls; consider caching or rate-limit adjustments.
