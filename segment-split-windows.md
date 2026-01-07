# Segment Split Windows Implementation Plan

This document outlines the work required to support "Segment Split Windows": the ability to define multiple date windows for a single segment and compute each runner's best effort across all configured windows.

---

## 1. Requirements Summary

1. **Workbook UX**:
   - Multiple windows are expressed by duplicating segment rows with the same **Segment ID** (grouping key).
   - An optional `Window Label` column provides human-friendly tags (e.g., "Week 1", "Final Push").
   - Output remains a single sheet per segment (when enabled), showing each runner's best time across all windows.
2. **Scoring rules**:
   - Each runner's fastest time across all windows is shown in the output sheet.
   - Ranks are computed from these aggregated best times.
   - Per-window attempts remain internal (diagnostics only).
3. **Birthday bonus**: Can be specified per window row. Defaults to 0 if omitted.
4. **Default time**: Applies once per segment group when a runner has no efforts in any window. Only needs to be specified on one row per segment group (if set on multiple rows, values must match).
5. **Minimum distance**: Applies per segment group. Only needs to be specified on one row per group (if set on multiple rows, values must match).
6. **Overlapping windows**: Allowed (e.g., weekly + monthly windows). Log a warning if windows fully overlap (likely user error).
7. **Replay/capture**: Reprocessing the same segment over multiple windows uses cached activities; no special handling required.

---

## 2. Workbook Schema Updates

### 2.1 New Column

Add optional `Window Label` column to the `Segment Series` sheet.

### 2.2 Sample Workbook Rows

| Segment ID | Segment Name      | Start Date | End Date   | Window Label | Default Time | Min Distance (m) | Birthday Bonus (secs) |
| ---------- | ----------------- | ---------- | ---------- | ------------ | ------------ | ---------------- | --------------------- |
| 12345678   | Hill Climb Sprint | 2026-01-01 | 2026-01-15 | Week 1       | 00:10:00     | 500              | 30                    |
| 12345678   | Hill Climb Sprint | 2026-01-16 | 2026-01-31 | Week 2       |              | 500              | 30                    |
| 12345678   | Hill Climb Sprint | 2026-02-01 | 2026-02-14 | Final Push   |              | 500              | 45                    |

Notes:

- Rows with the same `Segment ID` are grouped together.
- Row order does not affect processing (efforts are filtered by date, not row order).
- `Default Time` and `Min Distance (m)` only need to appear on one row per group (if set on multiple rows, values must match).
- `Birthday Bonus (secs)` can vary per window (e.g., 45 seconds in "Final Push"). Defaults to 0 if omitted.

### 2.3 Reader Updates (`excel_reader.py`)

1. Parse rows into `SegmentWindow` objects, grouped by `Segment ID`.
2. Validate:
   - All rows in a group must have the same `Segment Name` (error if mismatch).
   - If `Default Time` appears on multiple rows, values must match.
   - If `Min Distance (m)` appears on multiple rows, values must match.
   - Warn (don't error) if windows fully overlap.
3. Return `List[SegmentGroup]` where each group contains one or more windows.

---

## 3. Data Model Changes (`models.py`)

```python
@dataclass
class SegmentWindow:
    """A single date window within a segment group."""
    start_date: datetime
    end_date: datetime
    label: str | None = None
    birthday_bonus_seconds: float = 0.0


@dataclass
class SegmentGroup:
    """A segment with one or more date windows."""
    id: int
    name: str
    windows: List[SegmentWindow]
    default_time_seconds: float | None = None
    min_distance_meters: float | None = None
```

**Note**: `SegmentGroup` replaces the existing `Segment` dataclass. The old `Segment` class will be deprecated and removed.

### 3.1 Config Toggle

Add `SEGMENT_SPLIT_WINDOWS_ENABLED` to `config.py` (default: `True` once implemented).

When **enabled**: Rows with the same Segment ID are aggregated into a single output sheet showing each runner's best time across all windows.

When **disabled**: Each row produces its own output sheet. If a segment ID appears only once, the sheet is named `{Segment Name}` (unchanged from current behavior). If it appears multiple times, sheets are named `{Segment Name} - {Window Label}` (or `{Segment Name} - {Start Date} to {End Date}` if no label).

---

## 4. Service Layer Enhancements

### 4.1 Cache Strategy

1. Compute union date range per segment group: `(min(window.start), max(window.end))`.
2. Fetch activities/efforts once per runner for the union range.
3. Filter fetched efforts by window boundaries client-side.
4. Existing `_activity_cache` in `SegmentService` already supports this pattern.

### 4.2 Processing Logic

1. For each `SegmentGroup`, iterate through windows and collect efforts bounded by each window's dates.
2. Apply birthday bonus per-window (using that window's `birthday_bonus_seconds`).
3. Select the runner's fastest adjusted time across all windows.
4. Store diagnostics: which window produced the best effort, per-window times if useful for debugging.

### 4.3 Signature Change

```python
def process(
    self,
    segment_groups: Sequence[SegmentGroup],  # was: Sequence[Segment]
    runners: Sequence[Runner],
    ...
) -> ResultsMapping:
```

**Note**: This is a breaking change. Update all callers (`main.py`, CLI tools, tests) in the same PR.

---

## 5. Aggregation & Writer Updates

### 5.1 Output Structure

- One sheet per segment group, named by `segment_name` (unchanged).
- Each runner appears once with their best time across all windows.
- `Fastest Date` column shows the date of the best effort (from whichever window).
- `Attempts` column shows the total attempts across all windows (when enabled).

### 5.2 Birthday Highlighting

Applies to aggregated rows as before—if the best effort had birthday bonus applied, highlight that row.

### 5.3 Optional Diagnostics

Controlled by config flag: optionally include a `Best Window` column showing which window label produced the best time.

---

## 6. CLI & Tooling Alignment

1. Update CLI helpers (e.g., `fetch_runner_segment_efforts`) to accept segment groups.
2. No additional validation command needed—existing workbook parsing will report errors.

---

## 7. Backward Compatibility

1. **Opt-in by structure**: If a segment ID appears only once, it behaves exactly as today (single window = current behavior).
2. Existing workbooks without duplicate segment rows continue to work unchanged.
3. Auto-detect: When a segment ID appears multiple times, treat as split windows automatically.

---

## 8. Testing Strategy

1. **Unit tests**:
   - `excel_reader`: Parse single-window segments (regression), multi-window groups, validation errors.
   - `segment_service`: Combine efforts across windows, select best time, apply per-window birthday bonus.
   - Aggregation: Sheet naming, ranking, birthday highlighting.
2. **Integration tests**:
   - End-to-end with capture data covering two windows; verify best-time selection.
3. **Regression tests**:
   - Ensure single-window segments produce identical output to current behavior.

---

## 9. Documentation & Samples

1. Update README with "Segment Split Windows" section.
2. Provide sample workbook (under `samples/` or test fixtures) demonstrating three windows.
3. Document the `Window Label` column and birthday bonus override behavior.

---

## 10. Rollout Considerations

1. Feature flag (`SEGMENT_SPLIT_WINDOWS_ENABLED`) for staged rollout.
2. Release notes explaining schema change and workbook updates.
3. Performance: With union-range caching, API calls remain ~1 per runner per segment group (not per window).
