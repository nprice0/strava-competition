# Segment Service Optimisation: Union Prefetch Strategy

## Executive Summary

Refactor `SegmentService` to prefetch all runner activities once across the union date range of all segments, then filter locally per segment. This eliminates redundant API calls and prevents rate limiting.

**Expected impact:**

- Reduce **activities LIST** API calls from `O(runners × segments)` to `O(runners)` — **75% reduction** for 4 segments
- Activity **DETAIL** calls remain at `O(total_activities)` but are made once per activity, not repeated per segment
- Net result: Significant reduction in total API calls and elimination of redundant fetches

---

## 1. Problem Statement

### Current Behaviour

The `SegmentService` processes segments sequentially. For each segment:

1. Iterates through all runners
2. Fetches activities for that segment's date window
3. Clears the activity cache before the next segment

This causes:

- **Redundant API calls**: Same runner's activities fetched multiple times for overlapping windows
- **Rate limit exhaustion**: 34 runners × 4 segments = 136 activity fetches in rapid succession
- **Cache thrashing**: `_clear_runner_activity_cache()` wipes data between segments

### Bug: Rate-Limited Runners Are Silently Skipped

Activity scan is the **default** path (`FORCE_ACTIVITY_SCAN_FALLBACK=True`). When a 429 error occurs during `get_activities()`:

```python
# activities.py line 279 - on HTTPError including 429
return None

# segment_service.py _get_runner_activities
activities = get_activities(runner, start_date, end_date) or []  # None → []
```

The runner gets an empty activity list, so:

1. No activities to scan for segment efforts
2. No `SegmentResult` produced
3. Runner receives `default_time` if configured, otherwise **no result at all**
4. No error logged at the segment service level — it looks like the runner simply had no activities

**Impact:** Segment leaderboards are incomplete. Runners who happened to be fetched during a rate-limit window silently lose their times, with no indication that data is missing.

### Evidence from `log.log`

```
[2026-01-26 10:38:26,005] WARNING SegmentService: Activity scan failed runner=Tom Andrews segment=40422214: ... Rate Limit Exceeded
[2026-01-26 10:39:27,377] INFO root: Segment WS25 - Fancy A Tipple After Mass progress: 34/34 runners fetched
[2026-01-26 10:40:13,330] INFO root: Segment Sheriff of the Dunes progress: 1/34 runners fetched
```

Note: Immediately after finishing one segment, the next segment re-fetches the same runners.

---

## 2. Proposed Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: PREFETCH                           │
├─────────────────────────────────────────────────────────────────────┤
│  1. Compute union date range (earliest start → latest end)          │
│  2. Batch-fetch activities for each runner (controlled pacing)      │
│  3. Fetch activity details with segment efforts (include_all_efforts)│
│  4. Store in shared cache (persists across all segments)            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: LOCAL PROCESSING                        │
├─────────────────────────────────────────────────────────────────────┤
│  For each segment:                                                  │
│    1. Filter cached activities by segment's date window             │
│    2. Match segment efforts by segment ID                           │
│    3. Apply birthday/time bonuses                                   │
│    4. Build SegmentResult (no API calls)                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Single fetch phase**: All API calls happen upfront in a controlled, paced manner
2. **Immutable cache**: Once populated, cache is read-only during segment processing
3. **Fail-fast with graceful degradation**: If prefetch fails for a runner, log and continue
4. **Preserve existing fallback paths**: Keep `get_segment_efforts` API as secondary source

---

## 3. Implementation Steps

### Step 3.0: Fix Silent Skipping of Rate-Limited Runners

**File:** `strava_competition/services/segment_service.py`

The prefetch approach inherently fixes this (one fetch attempt per runner, with retries), but we should also track and report failures:

```python
@dataclass(slots=True)
class RunnerActivityCache:
    """All prefetched activities for a single runner."""
    runner_id: int | str
    activities: List[PrefetchedActivity] = field(default_factory=list)
    fetch_error: str | None = None  # Track WHY fetch failed
    retry_count: int = 0  # How many retries were attempted
```

In `_prefetch_runner_activities`, implement retry with exponential backoff:

```python
def _prefetch_runner_activities(
    self,
    runner: Runner,
    start_date: datetime,
    end_date: datetime,
    cancel_event: threading.Event | None,
    max_retries: int = 3,
) -> RunnerActivityCache:
    """Prefetch with retry on rate limits, falling back to disk cache."""
    last_error: str | None = None

    for attempt in range(1, max_retries + 1):
        if cancel_event and cancel_event.is_set():
            return RunnerActivityCache(
                runner_id=runner.strava_id,
                fetch_error="cancelled",
            )

        try:
            # This already checks disk cache internally before hitting API
            activities_raw = get_activities(runner, start_date, end_date)
            if activities_raw is not None:
                # Success - continue to fetch details
                return self._fetch_activity_details(runner, activities_raw, cancel_event)
        except StravaAPIError as exc:
            last_error = str(exc)
            if "429" in last_error and attempt < max_retries:
                delay = 2 ** attempt  # Exponential: 2s, 4s, 8s
                self._log.warning(
                    "Rate limited fetching activities for %s (attempt %d/%d); "
                    "waiting %ds before retry",
                    runner.name, attempt, max_retries, delay
                )
                time.sleep(delay)
                continue
            break  # Non-retryable error or exhausted retries

    # All retries failed - try disk cache as last resort
    cached_activities = self._load_from_disk_cache(runner, start_date, end_date)
    if cached_activities:
        self._log.warning(
            "Using stale disk cache for runner=%s after rate limit exhaustion",
            runner.name,
        )
        return RunnerActivityCache(
            runner_id=runner.strava_id,
            activities=cached_activities,
            fetch_error=None,  # We have data, just stale
            from_stale_cache=True,  # Flag for reporting
        )

    # No cache, no API - genuine failure
    return RunnerActivityCache(
        runner_id=runner.strava_id,
        fetch_error=last_error or "unknown_error",
        retry_count=max_retries,
    )
```

**Key addition:** `from_stale_cache` flag so results can indicate data staleness.

### Graceful Degradation Hierarchy

When fetching activities for a runner:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Fresh API call                                          │
│     ↓ (429 or error)                                        │
│  2. Retry with exponential backoff (2s, 4s, 8s)             │
│     ↓ (still failing)                                       │
│  3. Use disk cache (strava_cache/) if available             │
│     ↓ (cache miss)                                          │
│  4. Mark runner as FAILED - log explicitly                  │
└─────────────────────────────────────────────────────────────┘
```

### Reporting Stale Data

At the end of processing, report data quality:

```python
def _report_data_quality(self, cache: PrefetchCache) -> None:
    """Report data freshness and failures."""
    stale = [rid for rid, rc in cache.items() if getattr(rc, 'from_stale_cache', False)]
    failed = [rid for rid, rc in cache.items() if rc.fetch_error]

    if stale:
        self._log.warning(
            "%d runners using stale cached data (rate limit): %s",
            len(stale), stale[:5]
        )
    if failed:
        self._log.error(
            "%d runners have NO data (fetch failed, no cache): %s",
            len(failed), failed[:5]
        )
```

**Rationale:**

- Explicit retry logic with backoff gives rate limits time to reset
- `fetch_error` field makes failures visible (not silent)
- Summary log helps diagnose which runners need manual attention

---

### Step 3.1: Create Activity Prefetch Data Structures

**File:** `strava_competition/services/segment_service.py`

Define typed structures for the prefetch cache:

```python
from dataclasses import dataclass, field
from typing import TypeAlias

# Type aliases for clarity
ActivityDetail = Dict[str, Any]
SegmentEffort = Dict[str, Any]

@dataclass(slots=True)
class PrefetchedActivity:
    """An activity with its embedded segment efforts.

    Note: Not frozen because segment_efforts is a mutable list.
    Thread safety is achieved by treating the cache as read-only after prefetch.
    """
    activity_id: int
    start_date: datetime | None
    activity_type: str
    raw_data: ActivityDetail
    segment_efforts: tuple[SegmentEffort, ...] = field(default_factory=tuple)  # Immutable tuple

@dataclass(slots=True)
class RunnerActivityCache:
    """All prefetched activities for a single runner."""
    runner_id: int | str
    activities: List[PrefetchedActivity] = field(default_factory=list)
    fetch_error: str | None = None
    retry_count: int = 0
    from_stale_cache: bool = False  # True if data came from disk cache after API failure

PrefetchCache: TypeAlias = Dict[int | str, RunnerActivityCache]
```

**Rationale:**

- `segment_efforts` is a `tuple` (immutable) not `list` for thread safety
- `slots=True` reduces memory footprint
- `from_stale_cache` flag tracks data freshness
- Explicit `fetch_error` and `retry_count` for observability

---

### Step 3.2: Implement Union Date Range Computation

**File:** `strava_competition/services/segment_service.py`

Add method to compute the union range across all segments or segment groups:

```python
def _compute_union_date_range(
    self,
    segments: Sequence[Segment] | None = None,
    segment_groups: Sequence[SegmentGroup] | None = None,
) -> Tuple[datetime, datetime]:
    """Compute the earliest start and latest end across all segments.

    Args:
        segments: Legacy Segment objects (optional).
        segment_groups: SegmentGroup objects with windows (optional).

    Returns:
        Tuple of (earliest_start, latest_end) datetimes.

    Raises:
        ValueError: If no segments or groups provided.
    """
    all_starts: List[datetime] = []
    all_ends: List[datetime] = []

    if segments:
        for seg in segments:
            all_starts.append(seg.start_date)
            all_ends.append(seg.end_date)

    if segment_groups:
        for group in segment_groups:
            for window in group.windows:
                all_starts.append(window.start_date)
                all_ends.append(window.end_date)

    if not all_starts:
        raise ValueError("No segments or segment groups provided")

    return min(all_starts), max(all_ends)
```

**Rationale:**

- Handles both legacy `Segment` and new `SegmentGroup` models
- Pure function, easy to test
- Clear error for empty input

---

### Step 3.3: Implement Controlled Batch Prefetching

**File:** `strava_competition/services/segment_service.py`

Add the prefetch orchestration method:

```python
def _prefetch_all_activities(
    self,
    runners: Sequence[Runner],
    start_date: datetime,
    end_date: datetime,
    cancel_event: threading.Event | None = None,
) -> PrefetchCache:
    """Prefetch activities and details for all runners in controlled batches.

    Activities are fetched with embedded segment efforts to minimise API calls.
    Uses controlled pacing to avoid rate limiting.

    Args:
        runners: Runners to fetch activities for.
        start_date: Union window start (inclusive).
        end_date: Union window end (inclusive).
        cancel_event: Optional cancellation signal.

    Returns:
        PrefetchCache mapping runner IDs to their cached activities.
    """
    cache: PrefetchCache = {}
    eligible_runners = [r for r in runners if r.segment_team]

    if not eligible_runners:
        return cache

    self._log.info(
        "Prefetching activities for %d runners over window %s → %s",
        len(eligible_runners),
        start_date.date(),
        end_date.date(),
    )

    # Batch processing with controlled pacing
    batch_size = min(self.max_workers, len(eligible_runners))
    total_batches = (len(eligible_runners) + batch_size - 1) // batch_size

    for batch_num, batch_start in enumerate(
        range(0, len(eligible_runners), batch_size), start=1
    ):
        if cancel_event and cancel_event.is_set():
            self._log.info("Prefetch cancelled at batch %d/%d", batch_num, total_batches)
            break

        batch = eligible_runners[batch_start : batch_start + batch_size]
        self._prefetch_batch(
            batch, start_date, end_date, cache, cancel_event
        )

        # Pacing delay between batches (not after the last one)
        if batch_start + batch_size < len(eligible_runners):
            delay = random.uniform(1.0, 2.0)  # nosec B311
            self._log.debug(
                "Prefetch batch %d/%d complete; pacing delay %.1fs",
                batch_num, total_batches, delay
            )
            time.sleep(delay)

    successful = sum(1 for rc in cache.values() if rc.fetch_error is None)
    self._log.info(
        "Prefetch complete: %d/%d runners successful",
        successful, len(eligible_runners)
    )

    return cache
```

**Rationale:**

- Controlled batch size matches existing `max_workers`
- Pacing delay (1-2s) between batches prevents burst rate limiting
- Cancellation support for graceful shutdown
- Comprehensive logging for observability

---

### Step 3.4: Implement Batch Processing and Activity Detail Fetching

**File:** `strava_competition/services/segment_service.py`

```python
def _prefetch_batch(
    self,
    runners: Sequence[Runner],
    start_date: datetime,
    end_date: datetime,
    cache: PrefetchCache,
    cancel_event: threading.Event | None,
) -> None:
    """Prefetch activities for a batch of runners concurrently."""
    with ThreadPoolExecutor(max_workers=len(runners)) as executor:
        future_to_runner = {
            executor.submit(
                self._prefetch_runner_activities,  # Uses Step 3.0's implementation
                runner,
                start_date,
                end_date,
                cancel_event,
            ): runner
            for runner in runners
        }

        for future in as_completed(future_to_runner):
            runner = future_to_runner[future]
            try:
                runner_cache = future.result()
                cache[runner.strava_id] = runner_cache
            except Exception as exc:
                self._log.warning(
                    "Prefetch failed for runner=%s: %s",
                    runner.name, exc
                )
                cache[runner.strava_id] = RunnerActivityCache(
                    runner_id=runner.strava_id,
                    fetch_error=str(exc),
                )

def _fetch_activity_details(
    self,
    runner: Runner,
    activities_raw: List[Dict[str, Any]],
    cancel_event: threading.Event | None,
) -> RunnerActivityCache:
    """Fetch details with segment efforts for each activity.

    Called by _prefetch_runner_activities (Step 3.0) after successful
    activities list fetch.

    Args:
        runner: The runner whose activities we're fetching.
        activities_raw: List of activity summaries from get_activities().
        cancel_event: Event to signal cancellation.

    Returns:
        RunnerActivityCache with all activities and their segment efforts.
    """
    prefetched: List[PrefetchedActivity] = []

    for activity_summary in activities_raw:
        if cancel_event and cancel_event.is_set():
            break

        activity_id = activity_summary.get("id")
        if not activity_id:
            continue

        try:
            detail = get_activity_with_efforts(
                runner, activity_id, include_all_efforts=True
            )
            segment_efforts = detail.get("segment_efforts", []) or []

            start_dt = parse_iso_datetime(
                detail.get("start_date_local") or detail.get("start_date")
            )

            prefetched.append(PrefetchedActivity(
                activity_id=activity_id,
                start_date=start_dt,
                activity_type=detail.get("type", ""),
                raw_data=detail,
                segment_efforts=tuple(segment_efforts),  # Immutable tuple
            ))
        except StravaAPIError as exc:
            # Log but continue - partial data is better than none
            self._log.debug(
                "Failed to fetch detail for activity %s runner=%s: %s",
                activity_id, runner.name, exc
            )

    return RunnerActivityCache(
        runner_id=runner.strava_id,
        activities=prefetched,
    )

def _load_from_disk_cache(
    self,
    runner: Runner,
    start_date: datetime,
    end_date: datetime,
) -> List[PrefetchedActivity]:
    """Attempt to load activities from disk cache as fallback.

    This is called by _prefetch_runner_activities (Step 3.0) when API
    calls fail and retries are exhausted.

    Returns:
        List of PrefetchedActivity objects from cache, or empty list.
    """
    # The existing strava_cache already stores activity responses
    # We need to iterate over cached activities in the date range
    # Implementation depends on cache structure - this is a simplified version

    from strava_competition.services.activity_scan.activities import get_activities

    # get_activities already checks cache first - call it again to hit cache only
    # The 429 handling means fresh API calls will fail, but cache hits succeed
    try:
        cached_raw = get_activities(runner, start_date, end_date)
        if cached_raw is None:
            return []

        # Note: We may not have detailed efforts cached if we only cached the list
        # This is a known limitation - stale data may lack segment_efforts
        return [
            PrefetchedActivity(
                activity_id=act.get("id"),
                start_date=parse_iso_datetime(
                    act.get("start_date_local") or act.get("start_date")
                ),
                activity_type=act.get("type", ""),
                raw_data=act,
                segment_efforts=tuple(act.get("segment_efforts", []) or []),
            )
            for act in cached_raw
            if act.get("id")
        ]
    except Exception as exc:
        self._log.debug("Disk cache lookup failed for %s: %s", runner.name, exc)
        return []
```

**Rationale:**

- `_prefetch_batch`: Concurrent batch processing using `_prefetch_runner_activities` from Step 3.0
- `_fetch_activity_details`: Called by Step 3.0 after successful API response, fetches activity details
- `_load_from_disk_cache`: Called by Step 3.0 as last resort fallback
- Two-phase fetch: list then details (required by Strava API structure)
- Partial success: if one activity detail fails, others still cached
- Uses existing `get_activity_with_efforts` which handles auth/retry
- Extracts `segment_efforts` once, reused for all segment filters

---

### Step 3.5: Implement Cache-Based Effort Lookup

**File:** `strava_competition/services/segment_service.py`

Add method to find segment efforts from the prefetch cache:

```python
def _find_efforts_from_cache(
    self,
    runner: Runner,
    segment_id: int,
    start_date: datetime,
    end_date: datetime,
    cache: PrefetchCache,
) -> List[Dict[str, Any]]:
    """Find segment efforts for a runner from the prefetch cache.

    Filters cached activities by date window and matches segment ID
    in embedded segment_efforts.

    Args:
        runner: The runner to look up.
        segment_id: Target segment ID.
        start_date: Segment window start (inclusive).
        end_date: Segment window end (inclusive).
        cache: The prefetch cache.

    Returns:
        List of matching segment effort dicts (may be empty).
    """
    runner_cache = cache.get(runner.strava_id)
    if runner_cache is None or runner_cache.fetch_error:
        return []

    matching_efforts: List[Dict[str, Any]] = []

    for activity in runner_cache.activities:
        # Filter by activity date within window
        if activity.start_date is None:
            continue

        # Normalize to naive for comparison (segment dates are naive)
        activity_date = (
            activity.start_date.replace(tzinfo=None)
            if activity.start_date.tzinfo
            else activity.start_date
        )
        window_start = (
            start_date.replace(tzinfo=None)
            if hasattr(start_date, 'tzinfo') and start_date.tzinfo
            else start_date
        )
        window_end = (
            end_date.replace(tzinfo=None)
            if hasattr(end_date, 'tzinfo') and end_date.tzinfo
            else end_date
        )

        if activity_date < window_start or activity_date > window_end:
            continue

        # Find matching segment efforts within this activity
        for effort in activity.segment_efforts:
            effort_segment = effort.get("segment", {})
            effort_segment_id = effort_segment.get("id") or effort.get("segment_id")

            if effort_segment_id == segment_id:
                # Validate effort date is also within window
                effort_start = parse_iso_datetime(
                    effort.get("start_date_local") or effort.get("start_date")
                )
                if effort_start:
                    effort_date = (
                        effort_start.replace(tzinfo=None)
                        if effort_start.tzinfo
                        else effort_start
                    )
                    if effort_date < window_start or effort_date > window_end:
                        continue

                matching_efforts.append(effort)

    return matching_efforts
```

**Rationale:**

- Pure function: easy to test
- Handles timezone normalization (segment dates can be naive)
- Double-checks effort date, not just activity date
- Returns raw effort dicts for compatibility with existing validation

---

### Step 3.6: Refactor Process Methods to Use Prefetch

**File:** `strava_competition/services/segment_service.py`

Modify `process` and `process_groups` to:

1. Compute union range first
2. Prefetch all activities
3. Pass cache to processing methods
4. Remove per-segment cache clearing

```python
def process(
    self,
    segments: Sequence[Segment],
    runners: Sequence[Runner],
    cancel_event: threading.Event | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> ResultsMapping:
    """Process all segments for all runners, returning aggregated results."""
    results: ResultsMapping = {}

    if not segments or not runners:
        return results

    # Phase 1: Compute union range and prefetch
    try:
        union_start, union_end = self._compute_union_date_range(segments=segments)
    except ValueError:
        self._log.warning("No valid segments to process")
        return results

    prefetch_cache = self._prefetch_all_activities(
        runners, union_start, union_end, cancel_event
    )

    # Phase 2: Process each segment using cached data
    total_segments = len(segments)
    try:
        for seg_index, segment in enumerate(segments, start=1):
            if cancel_event and cancel_event.is_set():
                self._log.info("Cancellation requested; aborting segment processing")
                break

            if segment.start_date > segment.end_date:
                self._log.warning(
                    "Skipping segment with inverted date range: %s",
                    segment.name,
                )
                continue

            segment_results = self._process_segment_with_cache(
                segment,
                runners,
                prefetch_cache,
                seg_index,
                total_segments,
                cancel_event,
                progress,
            )
            for team_results in segment_results.values():
                team_results.sort(key=lambda r: r.fastest_time)
            results[segment.name] = segment_results
    finally:
        # Clear cache only after ALL segments processed
        self._clear_runner_activity_cache()

    return results
```

---

### Step 3.7: Create Cache-Aware Segment Processing

**File:** `strava_competition/services/segment_service.py`

New method that uses the prefetch cache instead of making API calls:

```python
def _process_segment_with_cache(
    self,
    segment: Segment,
    runners: Sequence[Runner],
    prefetch_cache: PrefetchCache,
    seg_index: int,
    total_segments: int,
    cancel_event: threading.Event | None,
    progress: Callable[[str, int, int], None] | None,
) -> Dict[str, List[SegmentResult]]:
    """Process a segment using prefetched activity data (no API calls)."""
    segment_results: Dict[str, List[SegmentResult]] = {}
    eligible_runners = [r for r in runners if r.segment_team]
    total_runners = len(eligible_runners)

    if total_runners == 0:
        return segment_results

    self._log.debug(
        "Processing segment %s (%d/%d) with %d runners from cache",
        segment.name, seg_index, total_segments, total_runners
    )

    completed = 0

    def notify_progress(count: int) -> None:
        if progress:
            try:
                progress(segment.name, count, total_runners)
            except Exception:
                pass

    for runner in eligible_runners:
        if cancel_event and cancel_event.is_set():
            break

        # Check if runner had a fetch error
        runner_cache = prefetch_cache.get(runner.strava_id)
        if runner_cache and runner_cache.fetch_error:
            self._log.debug(
                "Skipping runner %s for segment %s: prefetch failed (%s)",
                runner.name, segment.name, runner_cache.fetch_error
            )
            # Still inject default result below via _inject_default_segment_results
            completed += 1
            notify_progress(completed)
            continue

        # Look up efforts from cache (no API call)
        efforts = self._find_efforts_from_cache(
            runner,
            segment.id,
            segment.start_date,
            segment.end_date,
            prefetch_cache,
        )

        # Process results using existing validation logic
        result = self._result_from_efforts(runner, segment, efforts)

        # NOTE: We do NOT use activity scan fallback here!
        # Activity scan would make additional API calls, defeating the purpose.
        # If prefetch succeeded but found no matching efforts, that's the answer.
        # If prefetch failed, runner_cache.fetch_error is set and handled above.

        if result:
            team = runner.segment_team
            if team:
                bucket = segment_results.setdefault(team, [])
                bucket.append(result)

        completed += 1
        notify_progress(completed)

    # Inject defaults for runners without results
    self._inject_default_segment_results(segment, eligible_runners, segment_results)

    return segment_results
```

**Rationale:**

- No API calls in the processing loop - all data comes from prefetch cache
- Explicit handling of runners with fetch errors (logged and tracked)
- Reuses existing `_result_from_efforts` for validation/bonus logic
- Does NOT fall back to activity scan (would make more API calls, defeating purpose)
- Default results injected for runners without efforts via `_inject_default_segment_results`
- Maintains progress reporting for UI feedback

---

### Step 3.8: Update `process_groups` Similarly

Apply the same pattern to `process_groups`:

1. Compute union range across all groups/windows
2. Prefetch once
3. Process each group with cached data
4. Clear cache at the end

```python
def process_groups(
    self,
    segment_groups: Sequence[SegmentGroup],
    runners: Sequence[Runner],
    cancel_event: threading.Event | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> ResultsMapping:
    """Process segment groups using prefetched activity data."""
    results: ResultsMapping = {}

    if not segment_groups or not runners:
        return results

    # Phase 1: Compute union range and prefetch
    try:
        union_start, union_end = self._compute_union_date_range(
            segment_groups=segment_groups
        )
    except ValueError:
        self._log.warning("No valid segment groups to process")
        return results

    prefetch_cache = self._prefetch_all_activities(
        runners, union_start, union_end, cancel_event
    )

    # Phase 2: Process each group using cached data
    total_groups = len(segment_groups)
    try:
        for group_index, group in enumerate(segment_groups, start=1):
            if cancel_event and cancel_event.is_set():
                break

            if SEGMENT_SPLIT_WINDOWS_ENABLED:
                group_results = self._process_segment_group_with_cache(
                    group, runners, prefetch_cache,
                    group_index, total_groups, cancel_event, progress
                )
                for team_results in group_results.values():
                    team_results.sort(key=lambda r: r.fastest_time)
                results[group.name] = group_results
            else:
                # Disabled mode: process each window separately
                for window in group.windows:
                    sheet_name = self._get_window_sheet_name(group, window)
                    temp_segment = self._segment_from_group_window(group, window)
                    window_results = self._process_segment_with_cache(
                        temp_segment, runners, prefetch_cache,
                        group_index, total_groups, cancel_event, progress
                    )
                    for team_results in window_results.values():
                        team_results.sort(key=lambda r: r.fastest_time)
                    results[sheet_name] = window_results
    finally:
        self._clear_runner_activity_cache()

    return results

def _process_segment_group_with_cache(
    self,
    group: SegmentGroup,
    runners: Sequence[Runner],
    prefetch_cache: PrefetchCache,
    group_index: int,
    total_groups: int,
    cancel_event: threading.Event | None,
    progress: Callable[[str, int, int], None] | None,
) -> Dict[str, List[SegmentResult]]:
    """Process a segment group using prefetched activity data.

    A segment group can have multiple windows. We process all windows
    and aggregate results by runner (best time across all windows).
    """
    # For segment groups, delegate to existing group processing logic
    # but using cache-based effort lookup instead of API calls

    all_group_results: Dict[str, List[SegmentResult]] = {}

    for window in group.windows:
        temp_segment = self._segment_from_group_window(group, window)
        window_results = self._process_segment_with_cache(
            temp_segment, runners, prefetch_cache,
            group_index, total_groups, cancel_event, progress
        )

        # Merge window results - keep best time per runner
        for team, team_results in window_results.items():
            existing = all_group_results.setdefault(team, [])
            for result in team_results:
                # Check if runner already has a result
                existing_result = next(
                    (r for r in existing if r.runner_name == result.runner_name),
                    None
                )
                if existing_result:
                    # Keep the faster time
                    if result.fastest_time < existing_result.fastest_time:
                        existing.remove(existing_result)
                        existing.append(result)
                else:
                    existing.append(result)

    return all_group_results
```

---

### Step 3.9: Add New Imports

**File:** `strava_competition/services/segment_service.py`

Add required imports at the top of the file:

```python
import random
import time
from dataclasses import dataclass, field
from typing import TypeAlias

# Add get_activity_with_efforts to existing import
from ..strava_api import get_segment_efforts, get_activities, get_activity_with_efforts
```

---

## 4. Testing Strategy

### 4.1 Unit Tests

**File:** `tests/test_segment_service_prefetch.py`

```python
"""Unit tests for SegmentService prefetch optimisation."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from strava_competition.services.segment_service import (
    SegmentService,
    PrefetchedActivity,
    RunnerActivityCache,
)
from strava_competition.models import Runner, Segment, SegmentGroup, SegmentWindow


class TestComputeUnionDateRange:
    """Tests for _compute_union_date_range."""

    def test_single_segment(self):
        """Union of one segment is its own range."""
        service = SegmentService()
        seg = Mock(start_date=datetime(2026, 1, 1), end_date=datetime(2026, 1, 15))

        start, end = service._compute_union_date_range(segments=[seg])

        assert start == datetime(2026, 1, 1)
        assert end == datetime(2026, 1, 15)

    def test_overlapping_segments(self):
        """Union covers earliest to latest across overlapping segments."""
        service = SegmentService()
        seg1 = Mock(start_date=datetime(2026, 1, 1), end_date=datetime(2026, 1, 15))
        seg2 = Mock(start_date=datetime(2026, 1, 10), end_date=datetime(2026, 1, 25))

        start, end = service._compute_union_date_range(segments=[seg1, seg2])

        assert start == datetime(2026, 1, 1)
        assert end == datetime(2026, 1, 25)

    def test_empty_raises(self):
        """Empty input raises ValueError."""
        service = SegmentService()

        with pytest.raises(ValueError, match="No segments"):
            service._compute_union_date_range()


class TestFindEffortsFromCache:
    """Tests for _find_efforts_from_cache."""

    def test_finds_matching_efforts(self):
        """Returns efforts matching segment ID within date window."""
        service = SegmentService()
        runner = Mock(strava_id=123)

        cache = {
            123: RunnerActivityCache(
                runner_id=123,
                activities=[
                    PrefetchedActivity(
                        activity_id=1,
                        start_date=datetime(2026, 1, 10),
                        activity_type="Run",
                        raw_data={},
                        segment_efforts=(  # Tuple, not list
                            {"segment": {"id": 999}, "elapsed_time": 300},
                            {"segment": {"id": 888}, "elapsed_time": 200},
                        ),
                    )
                ],
            )
        }

        efforts = service._find_efforts_from_cache(
            runner, 999, datetime(2026, 1, 1), datetime(2026, 1, 15), cache
        )

        assert len(efforts) == 1
        assert efforts[0]["segment"]["id"] == 999

    def test_filters_by_date_window(self):
        """Excludes activities outside the date window."""
        service = SegmentService()
        runner = Mock(strava_id=123)

        cache = {
            123: RunnerActivityCache(
                runner_id=123,
                activities=[
                    PrefetchedActivity(
                        activity_id=1,
                        start_date=datetime(2026, 2, 1),  # Outside window
                        activity_type="Run",
                        raw_data={},
                        segment_efforts=(  # Tuple, not list
                            {"segment": {"id": 999}, "elapsed_time": 300},
                        ),
                    )
                ],
            )
        }

        efforts = service._find_efforts_from_cache(
            runner, 999, datetime(2026, 1, 1), datetime(2026, 1, 15), cache
        )

        assert len(efforts) == 0

    def test_handles_missing_runner(self):
        """Returns empty list for runner not in cache."""
        service = SegmentService()
        runner = Mock(strava_id=999)
        cache = {}

        efforts = service._find_efforts_from_cache(
            runner, 999, datetime(2026, 1, 1), datetime(2026, 1, 15), cache
        )

        assert efforts == []


class TestPrefetchIntegration:
    """Integration tests for prefetch flow."""

    # Mock where functions are USED (segment_service), not where defined (strava_api)
    @patch("strava_competition.services.segment_service.get_activities")
    @patch("strava_competition.services.segment_service.get_activity_with_efforts")
    def test_prefetch_populates_cache(self, mock_detail, mock_activities):
        """Prefetch retrieves activities and details for all runners."""
        mock_activities.return_value = [{"id": 1}, {"id": 2}]
        mock_detail.return_value = {
            "id": 1,
            "start_date": "2026-01-10T10:00:00Z",
            "type": "Run",
            "segment_efforts": [{"segment": {"id": 999}}],
        }

        service = SegmentService(max_workers=1)
        runner = Mock(strava_id=123, segment_team="Team A", name="Test Runner")

        cache = service._prefetch_all_activities(
            [runner],
            datetime(2026, 1, 1),
            datetime(2026, 1, 31),
        )

        assert 123 in cache
        assert len(cache[123].activities) == 2  # Two activities
        mock_activities.assert_called_once()
        assert mock_detail.call_count == 2  # Detail for each activity

    @patch("strava_competition.services.segment_service.get_activities")
    @patch("strava_competition.services.segment_service.time.sleep")
    def test_prefetch_uses_disk_cache_on_rate_limit(self, mock_sleep, mock_activities):
        """Falls back to disk cache when API returns 429 after retries."""
        from strava_competition.errors import StravaAPIError

        # First 3 calls raise 429, simulating rate limit
        mock_activities.side_effect = [
            StravaAPIError("Rate Limit Exceeded", status_code=429),
            StravaAPIError("Rate Limit Exceeded", status_code=429),
            StravaAPIError("Rate Limit Exceeded", status_code=429),
        ]

        service = SegmentService(max_workers=1)
        runner = Mock(strava_id=123, segment_team="Team A", name="Test Runner")

        # Mock disk cache to return stale data
        with patch.object(service, "_load_from_disk_cache") as mock_cache:
            mock_cache.return_value = [
                PrefetchedActivity(
                    activity_id=1,
                    start_date=datetime(2026, 1, 10),
                    activity_type="Run",
                    raw_data={},
                    segment_efforts=(),
                )
            ]

            cache = service._prefetch_all_activities(
                [runner],
                datetime(2026, 1, 1),
                datetime(2026, 1, 31),
            )

        assert 123 in cache
        assert cache[123].from_stale_cache is True
        assert cache[123].fetch_error is None  # Has data, no error
        assert len(cache[123].activities) == 1
        # Verify retries happened with backoff
        assert mock_activities.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep between retries 1→2 and 2→3

    @patch("strava_competition.services.segment_service.get_activities")
    @patch("strava_competition.services.segment_service.time.sleep")
    def test_prefetch_records_error_when_no_cache(self, mock_sleep, mock_activities):
        """Records fetch_error when API fails and no disk cache available."""
        from strava_competition.errors import StravaAPIError

        mock_activities.side_effect = StravaAPIError("Rate Limit Exceeded", status_code=429)

        service = SegmentService(max_workers=1)
        runner = Mock(strava_id=123, segment_team="Team A", name="Test Runner")

        # Mock disk cache to return empty (no cached data)
        with patch.object(service, "_load_from_disk_cache", return_value=[]):
            cache = service._prefetch_all_activities(
                [runner],
                datetime(2026, 1, 1),
                datetime(2026, 1, 31),
            )

        assert 123 in cache
        assert cache[123].fetch_error is not None
        assert "429" in cache[123].fetch_error or "Rate Limit" in cache[123].fetch_error
        assert len(cache[123].activities) == 0
```

### 4.2 Performance Benchmark

**File:** `tests/test_segment_service_benchmark.py`

```python
"""Benchmark tests comparing old vs new approach."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime


@pytest.mark.benchmark
class TestAPICallReduction:
    """Verify API call reduction."""

    def test_api_calls_scale_with_runners_not_segments(self):
        """New approach: API calls = O(runners), not O(runners × segments)."""
        # Track API calls
        call_count = {"activities": 0, "details": 0}

        def mock_activities(*args, **kwargs):
            call_count["activities"] += 1
            return [{"id": 1}]

        def mock_details(*args, **kwargs):
            call_count["details"] += 1
            return {"segment_efforts": []}

        with patch("strava_competition.services.segment_service.get_activities", mock_activities):
            with patch("strava_competition.services.segment_service.get_activity_with_efforts", mock_details):
                service = SegmentService(max_workers=1)

                # 5 runners, 4 segments
                runners = [Mock(strava_id=i, segment_team="A", name=f"R{i}") for i in range(5)]
                segments = [Mock(id=i, name=f"S{i}", start_date=datetime(2026, 1, 1),
                                end_date=datetime(2026, 1, 31)) for i in range(4)]

                service.process(segments, runners)

        # Old approach would be: 5 runners × 4 segments = 20 activity calls
        # New approach: 5 runners × 1 = 5 activity calls
        assert call_count["activities"] == 5
```

---

## 5. Rollback Plan

### Feature Flag

Add a feature flag to enable/disable the new approach:

**File:** `strava_competition/config.py`

```python
# Enable prefetch optimisation for segment processing
# Set to False to use legacy per-segment fetching
SEGMENT_PREFETCH_ENABLED: bool = _env_bool("SEGMENT_PREFETCH_ENABLED", True)
```

### Conditional Logic

In `process` and `process_groups`:

```python
from ..config import SEGMENT_PREFETCH_ENABLED

def process(self, segments, runners, ...):
    if SEGMENT_PREFETCH_ENABLED:
        return self._process_with_prefetch(segments, runners, ...)
    else:
        return self._process_legacy(segments, runners, ...)
```

### Rollback Procedure

1. Set `SEGMENT_PREFETCH_ENABLED=false` in environment
2. Restart application
3. Monitor logs for rate limiting
4. If resolved, investigate prefetch implementation

---

## 6. Deployment Checklist

- [ ] All unit tests pass
- [ ] Benchmark tests confirm API reduction
- [ ] Feature flag defaults to `True`
- [ ] Logging confirms prefetch phase completes
- [ ] No rate limit errors in test run with 34 runners × 4 segments
- [ ] Memory usage within acceptable bounds
- [ ] Rollback procedure documented and tested

---

## 7. Files to Modify

| File                                             | Changes                                                                   |
| ------------------------------------------------ | ------------------------------------------------------------------------- |
| `strava_competition/services/segment_service.py` | Add prefetch logic, new data classes, refactor `process`/`process_groups` |
| `strava_competition/config.py`                   | Add `SEGMENT_PREFETCH_ENABLED` flag                                       |
| `tests/test_segment_service_prefetch.py`         | New test file for prefetch functionality                                  |
| `tests/test_segment_service_benchmark.py`        | New benchmark tests                                                       |

---

## 8. Estimated Effort

| Task                                     | Estimate     |
| ---------------------------------------- | ------------ |
| Data structures (Step 3.1)               | 30 min       |
| Union date range (Step 3.2)              | 15 min       |
| Batch prefetching (Steps 3.3-3.4)        | 1 hour       |
| Cache lookup (Step 3.5)                  | 30 min       |
| Refactor process methods (Steps 3.6-3.8) | 1.5 hours    |
| Unit tests                               | 1 hour       |
| Integration testing                      | 1 hour       |
| **Total**                                | **~6 hours** |

---

## 9. Success Metrics

1. **API call reduction**: Activities fetch calls reduced by 75%+ (from `runners × segments` to `runners`)
2. **No rate limiting**: Zero 429 errors during full competition run
3. **No regression**: All existing tests pass
4. **Performance**: Total runtime reduced (fewer API calls = faster completion)
