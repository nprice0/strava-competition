# Refactor: Remove Segment Efforts API — Use Activity Scan Only

## Overview

This document describes the refactor to **remove the Strava segment efforts API code** entirely and make **activity scan** the only approach for finding segment efforts.

### Current State

The codebase has two paths for finding segment efforts:

1. **Segment Efforts API** — Calls `GET /segment_efforts` endpoint directly
2. **Activity Scan** — Fetches runner activities, then extracts segment efforts from each activity's `segment_efforts` array

Currently controlled by two flags:

- `USE_ACTIVITY_SCAN_FALLBACK = True` — Fall back to activity scan when segment API fails
- `FORCE_ACTIVITY_SCAN_FALLBACK = True` — Always use activity scan, bypass segment API

Since `FORCE_ACTIVITY_SCAN_FALLBACK` defaults to `True`, the segment efforts API code is **never executed** in normal operation.

### Target State

- Activity scan becomes the **only** approach
- Remove all segment efforts API code
- Remove both configuration flags
- Simplify `segment_service.py` significantly

---

## Benefits

1. **Reduced complexity** — Remove ~500-800 lines of dead code
2. **Single code path** — Easier to debug and maintain
3. **No API cost** — Segment efforts API requires Strava subscription; activity scan works for all users
4. **Cleaner config** — Remove two confusing/redundant flags

---

## Files to Modify

### 1. Delete Entirely

| File                               | Lines | Notes                                        |
| ---------------------------------- | ----- | -------------------------------------------- |
| `strava_client/segment_efforts.py` | ~380  | Entire `SegmentEffortsAPI` class and helpers |

### 2. Remove Functions/Code

#### `strava_competition/strava_api.py`

**Remove:**

- `_get_segment_efforts_impl()` function (lines ~43-63)
- `SegmentEffortsAPI` import
- `StravaClient.get_segment_efforts()` method
- `get_segment_efforts()` module-level function (if exists)

**Keep:**

- `_get_activities_impl()`
- `StravaClient.get_activities()`
- All activity/resource methods

#### `strava_competition/strava_client/__init__.py`

**Remove:**

```python
from .segment_efforts import SegmentEffortsAPI  # noqa: F401
```

#### `strava_competition/config.py`

**Remove:**

```python
# Global switch for the activity scan fallback.
USE_ACTIVITY_SCAN_FALLBACK = _env_bool("USE_ACTIVITY_SCAN_FALLBACK", True)

# When True, always bypass Strava efforts and use activity scan. Useful for debugging.
FORCE_ACTIVITY_SCAN_FALLBACK = _env_bool("FORCE_ACTIVITY_SCAN_FALLBACK", True)
```

#### `strava_competition/services/segment_service.py`

This is the largest change. The file contains significant logic for:

- Trying segment API first, then falling back to activity scan
- Conditional checks on `FORCE_ACTIVITY_SCAN_FALLBACK` and `USE_ACTIVITY_SCAN_FALLBACK`

**Remove:**

- Import of `get_segment_efforts`
- Import of `FORCE_ACTIVITY_SCAN_FALLBACK`, `USE_ACTIVITY_SCAN_FALLBACK`
- `_validate_efforts_for_window()` method (only used for segment API efforts)
- `_result_from_efforts()` method (only used for segment API efforts)
- All conditional blocks checking these flags
- The "try segment API first" logic in `_process_runner_for_group()`

**Simplify:**

- `_process_runner_for_group()` — Just call activity scan directly
- `_get_best_result_for_runner()` — Remove segment API path

**Before (conceptual):**

```python
def _process_runner_for_group(...):
    efforts = None
    if not runner.payment_required:
        try:
            efforts = get_segment_efforts(...)  # ← REMOVE
        except:
            efforts = None

    validated = self._validate_efforts_for_window(...)  # ← REMOVE

    if not validated and USE_ACTIVITY_SCAN_FALLBACK:  # ← SIMPLIFY
        scan_result = self._result_from_activity_scan(...)
```

**After (conceptual):**

```python
def _process_runner_for_group(...):
    return self._result_from_activity_scan(...)
```

---

### 3. Test Files to Modify

| File                                            | Action        | Notes                                                                               |
| ----------------------------------------------- | ------------- | ----------------------------------------------------------------------------------- |
| `tests/test_strava_api_mocked.py`               | Remove tests  | `test_get_segment_efforts_*` tests (4-5 tests)                                      |
| `tests/test_strava_client.py`                   | Remove tests  | Tests using `get_segment_efforts`                                                   |
| `tests/test_strava_client_capture_contracts.py` | Remove tests  | `test_segment_efforts_*` tests                                                      |
| `tests/test_segment_split_windows.py`           | Simplify      | Remove `get_segment_efforts` mocking, remove `FORCE_ACTIVITY_SCAN_FALLBACK` patches |
| `tests/test_segment_service_filters.py`         | Simplify      | Remove `FORCE_ACTIVITY_SCAN_FALLBACK` patches                                       |
| `tests/test_excel_integration.py`               | Simplify      | Remove `get_segment_efforts` mock                                                   |
| `tests/test_activity_scan.py`                   | Keep/Simplify | Remove `USE_ACTIVITY_SCAN_FALLBACK` patches (now always active)                     |

---

### 4. Documentation Updates

#### `README.md`

**Update environment variables table:**
Remove:

```markdown
| `USE_ACTIVITY_SCAN_FALLBACK` | `true` | Fall back to activity scan when segment API fails |
| `FORCE_ACTIVITY_SCAN_FALLBACK` | `false` | Always use activity scan, bypassing segment API |
```

**Update explanatory text:**

- Remove references to segment efforts API as a data source
- Simplify the "activity scan fallback" section to just describe activity scan as the primary approach

#### `refactor-cache-mode.md`

**Remove or update:**

- References to `segment_efforts.py`
- Any examples showing segment efforts API usage

---

### 5. Tools

#### `strava_competition/tools/fetch_runner_segment_efforts.py`

**Decision needed:** This tool fetches activities and extracts segment efforts from them (activity scan approach). Despite the name, it does NOT use the segment efforts API.

**Recommended action:** Keep the tool but consider renaming to `fetch_runner_activities.py` or similar for clarity.

---

## Detailed Code Changes

### `strava_client/segment_efforts.py` — DELETE ENTIRE FILE

The entire 380-line file implements `SegmentEffortsAPI` which:

- Paginates through `/segment_efforts` endpoint
- Handles caching/replay of segment effort responses
- Manages tail-fill logic for incremental updates

None of this is needed when using activity scan exclusively.

---

### `strava_api.py` — Remove segment efforts functions

```python
# REMOVE these imports:
from .strava_client.segment_efforts import SegmentEffortsAPI

# REMOVE this function (~20 lines):
def _get_segment_efforts_impl(
    runner: Runner,
    segment_id: int,
    start_date: datetime,
    end_date: datetime,
    *,
    session: Optional[requests.Session] = None,
    limiter: Optional[RateLimiter] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Delegate to :class:`SegmentEffortsAPI` for backwards compatibility."""
    api = SegmentEffortsAPI(...)
    return api.get_segment_efforts(...)

# REMOVE this method from StravaClient class:
def get_segment_efforts(
    self,
    runner: Runner,
    segment_id: int,
    start_date: datetime,
    end_date: datetime,
) -> Optional[List[Dict[str, Any]]]:
    return _get_segment_efforts_impl(...)

# REMOVE module-level function if it exists:
def get_segment_efforts(...):
    ...
```

---

### `services/segment_service.py` — Major Simplification

#### Remove imports

```python
# REMOVE:
from ..strava_api import get_segment_efforts, get_activities
# REPLACE WITH:
from ..strava_api import get_activities

# REMOVE:
from ..config import (
    FORCE_ACTIVITY_SCAN_FALLBACK,
    USE_ACTIVITY_SCAN_FALLBACK,
    ...
)
# KEEP other config imports
```

#### Remove methods

```python
# REMOVE entirely:
def _validate_efforts_for_window(self, runner, group, window, efforts):
    """Validate and filter efforts for a specific window."""
    if FORCE_ACTIVITY_SCAN_FALLBACK:
        return []
    ...

# REMOVE entirely:
def _result_from_efforts(self, runner, segment, efforts):
    """Convert Strava segment efforts into a SegmentResult if available."""
    if FORCE_ACTIVITY_SCAN_FALLBACK:
        return None
    ...
```

#### Simplify `_process_runner_for_group()`

**Current logic (conceptual):**

```python
for window in group.windows:
    # Try segment API first
    efforts = None
    if not runner.payment_required:
        try:
            efforts = get_segment_efforts(...)
        except:
            efforts = None

    # Validate segment API efforts
    validated = self._validate_efforts_for_window(runner, group, window, efforts)

    # Fallback to activity scan
    if not validated and USE_ACTIVITY_SCAN_FALLBACK:
        scan_result = self._result_from_activity_scan(...)
```

**Simplified logic:**

```python
for window in group.windows:
    temp_segment = self._segment_from_group_window(group, window)
    scan_result = self._result_from_activity_scan(runner, temp_segment, cancel_event)
    if scan_result:
        # collect result
```

#### Simplify `_get_best_result_for_runner()`

Similar pattern — remove segment API path, keep only activity scan.

---

## Config Changes Summary

| Setting                        | Current                   | After Refactor |
| ------------------------------ | ------------------------- | -------------- |
| `USE_ACTIVITY_SCAN_FALLBACK`   | `True` (env configurable) | **REMOVED**    |
| `FORCE_ACTIVITY_SCAN_FALLBACK` | `True` (env configurable) | **REMOVED**    |

Activity scan is now the only approach — no configuration needed.

---

## Test Impact

### Tests to Delete

1. `test_strava_api_mocked.py`:

   - `test_get_segment_efforts_pagination`
   - `test_get_segment_efforts_refresh_on_401`
   - `test_get_segment_efforts_402_json_error`
   - `test_get_segment_efforts_offline_cache_miss`

2. `test_strava_client.py`:

   - Tests that mock/call `get_segment_efforts`

3. `test_strava_client_capture_contracts.py`:
   - `test_segment_efforts_fetches_tail_after_cached_runs`
   - Any tests using `SegmentEffortsAPI`

### Tests to Simplify

1. `test_segment_split_windows.py`:

   - Remove `monkeypatch.setattr(mod, "get_segment_efforts", fake_get_efforts)`
   - Remove `monkeypatch.setattr(mod, "FORCE_ACTIVITY_SCAN_FALLBACK", False)`

2. `test_segment_service_filters.py`:

   - Remove `FORCE_ACTIVITY_SCAN_FALLBACK` patches

3. `test_activity_scan.py`:

   - Remove `USE_ACTIVITY_SCAN_FALLBACK` patches (always active now)

4. `test_excel_integration.py`:
   - Remove `get_segment_efforts` mock

---

## Migration Checklist

1. [ ] Delete `strava_client/segment_efforts.py`
2. [ ] Update `strava_client/__init__.py` — remove `SegmentEffortsAPI` export
3. [ ] Update `strava_api.py` — remove segment efforts functions
4. [ ] Update `config.py` — remove both flags
5. [ ] Simplify `services/segment_service.py` — major refactor
6. [ ] Delete segment efforts tests from `test_strava_api_mocked.py`
7. [ ] Delete segment efforts tests from `test_strava_client.py`
8. [ ] Delete segment efforts tests from `test_strava_client_capture_contracts.py`
9. [ ] Simplify `test_segment_split_windows.py`
10. [ ] Simplify `test_segment_service_filters.py`
11. [ ] Simplify `test_activity_scan.py`
12. [ ] Simplify `test_excel_integration.py`
13. [ ] Update `README.md` — remove env vars, update explanations
14. [ ] Run full test suite: `python -m pytest tests/ -q`
15. [ ] Run type check: `python -m mypy strava_competition/`
16. [ ] Optional: Rename `fetch_runner_segment_efforts.py` tool

---

## Config Renames

Since activity scan becomes the only approach, the `ACTIVITY_SCAN_` prefix is redundant. Consider renaming:

| Current Name                              | Proposed Name               | Notes                                                     |
| ----------------------------------------- | --------------------------- | --------------------------------------------------------- |
| `ACTIVITY_SCAN_MAX_ACTIVITY_PAGES`        | `MAX_ACTIVITY_PAGES`        | Limits pages fetched when scanning activities             |
| `ACTIVITY_SCAN_CACHE_INCLUDE_ALL_EFFORTS` | `CACHE_INCLUDE_ALL_EFFORTS` | Whether to request all segment efforts in activity detail |

---

## Estimated Scope

- **Lines removed:** ~800-1000
- **Lines added:** ~50 (simplified logic)
- **Net reduction:** ~750-950 lines
- **Files modified:** ~12-15
- **Files deleted:** 1 (`segment_efforts.py`)
- **Tests deleted:** ~6-8
- **Tests simplified:** ~8-10

---

## Risks & Considerations

1. **No rollback path** — Once segment efforts API code is removed, re-enabling requires rewriting
2. **Tool naming** — `fetch_runner_segment_efforts.py` name is misleading (it uses activity scan)
3. **Cache compatibility** — Existing cached segment effort responses become orphaned (harmless)

---

## Recommendation

Proceed with this refactor. The segment efforts API code is:

- Never executed (FORCE=True)
- Requires paid Strava subscription
- More complex than activity scan
- Adds maintenance burden

Activity scan is simpler, works for all users, and is already the de facto approach.
