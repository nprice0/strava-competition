# Runner Activity Prefetch & Stream Cache Enhancements

This document outlines a step-by-step implementation plan for improving segment processing efficiency by reducing redundant Strava API calls while controlling memory usage.

## 1. Increase Matcher Stream Cache Capacity

- Evaluate a larger default for `MATCHING_ACTIVITY_STREAM_CACHE_SIZE` (e.g., 512 or 1024) in `strava_competition/config.py`, documenting why the chosen value balances memory versus reuse.
- Ensure the environment variable override continues to work; document the option in the README if needed.

## 2. Analyse Segment Windows per Runner

- Aggregate all segment date ranges for each runner before processing begins.
- Merge overlapping or adjacent windows so each runner has the minimum number of disjoint ranges covering all required dates.

## 3. Introduce Per-Runner Activity Prefetch

- For each runner, fetch activities once per merged window using the existing `get_activities` API helper.
- Immediately reduce each activity dict to only the fields used by downstream logic (id, type, distance, timestamps, elapsed/moving time, etc.).
- Write the reduced list to a disk-backed cache (JSON or similar) keyed by runner ID and window hash.
- Record metadata in the cache file (window start/end, schema version, timestamp) to support validation and future migrations.

## 4. Disk Cache Management

- Store cache files in a dedicated directory (e.g., `.cache/activities/`).
- Implement pruning logic to delete stale files when they fall outside all active segment windows or exceed a retention threshold.
- Guard against concurrent access with per-file locking or atomic writes (e.g., write to temp file then rename).
- Treat cache files as sensitive data; ensure they are ignored by version control and document the path for operators.
- On startup, clear any existing cache files so a partial previous run does not contaminate new processing; prefer verifying files atomically during writes so healthy cache entries persist across runs. Avoid wiping the cache on shutdown to preserve the performance benefit.
- Optionally compress cache payloads (gzip or similar) and enforce a disk usage budget to prevent the cache directory from growing unbounded.

## 5. Consume Prefetched Activities During Segment Processing

- Update `SegmentService` to load a runners cached activities on demand when `_get_runner_activities` is called.
- Slice/filter the loaded list to the requested segment window rather than re-querying Strava (and short-circuit when the cached window exactly matches the request).
- If a cache miss occurs because the requested window extends beyond the stored range, fall back to a live fetch, then refresh the cache file.
- Clear the in-memory list after processing each runner to keep RAM bounded to the largest single runner dataset.

## 6. Maintain Existing Stream Cache Behaviour

- Keep using the matcher stream LRU cache in `matching.fetchers`. Increase its size if needed after profiling, but validate that the disk-backed summary cache already removes most duplicate requests.
- Continue storing segment geometry/GPS data for the active segment in memory; only one segment is processed at a time, so in-memory retention remains safe.

## 7. Instrumentation & Validation

- Add logging around cache hits/misses, file load times, and fallback fetches to quantify benefits and spot regression points.
- Create unit tests covering cache read/write, stale-file detection, and the new `_get_runner_activities` fallback path.
- Run end-to-end tests (existing pytest suites plus a representative `run.py` invocation) to confirm behaviour matches the pre-change baseline while reducing API traffic.

## 8. Operational Guidance

- Document how to clear the cache directory manually if data corruption is suspected.
- Provide configuration knobs (environment variables) for cache directory path, retention limits, and stream cache size.
- Update README or operations guide with new workflow notes for running the application with disk-backed caches.
