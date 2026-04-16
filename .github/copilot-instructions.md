# Strava Competition — Repository Guidelines

Repository-wide guidance for all contributions.

## Instruction Files

The following instruction files extend this baseline and **must be followed**
alongside it. They are auto-applied by Copilot based on `applyTo` globs, but are
listed here so every contributor (human or AI) is aware of them:

- [instructions/python.instructions.md](instructions/python.instructions.md) —
  applies to `**/*.py`. Authoritative source for Python style, typing,
  docstrings, imports, error handling, testing, and quality gates. **Follow
  this file for any Python code generation or edit.**
- [instructions/python-code-review.instructions.md](instructions/python-code-review.instructions.md) —
  applies to `**/*.py`. Three-pass review methodology. **Follow this file
  whenever a code review is explicitly requested.**

If guidance here conflicts with a language-specific instruction file, the
language-specific file wins for files it targets.

## Project Overview

- **Primary language:** Python 3.10+ (see [instructions/python.instructions.md](instructions/python.instructions.md))
- **Package layout:**
  - `strava_competition/` — core library (models, services, API client, Excel I/O)
  - `strava_competition/services/` — orchestration (distance, segment)
  - `strava_competition/strava_client/` — HTTP client, caching, rate limiting
  - `strava_competition/tools/` — standalone CLI utilities (run via `python -m`)
  - `strava_competition/activity_scan/` — activity + segment effort scanning
  - `tests/` — pytest suite with cached API fixtures in `tests/strava_cache/`
- **Threading:** services use `ThreadPoolExecutor` with per-runner token locks
- **Caching:** disk-based response cache in `strava_cache/` (hashed paths, PII redaction)

## Code Quality Standards

All code must be **production-ready**:

- Clean, readable, and self-documenting
- Fully typed where the language supports it
- Covered by tests
- Free of linter, type checker, and security warnings
- Handles failures gracefully—no silent data loss
- Logs meaningful diagnostics without exposing secrets

## Security

- **Never** hardcode secrets, tokens, or credentials. Use environment variables or `.env`.
- Treat Strava tokens as secrets—never log, print, or expose them.
- Validate and sanitise all external inputs (API responses, Excel data).
- Use HTTPS for all external requests; set timeouts on network calls.

## Performance & Reliability

- Cache expensive API responses (this project uses `strava_cache/`).
- Respect Strava rate limits—use backoff and retry logic.
- Prefer batch operations over loops with individual API calls.
- Profile before optimising; don't guess at bottlenecks.
- Handle partial failures—if one runner fails, continue with others.

## Git & Pull Requests

- Keep commits small and atomic—one logical change per commit.
- Write clear commit messages: `Fix token refresh race condition`
- Include tests and documentation updates in the same PR as code changes.
- Link to issues where relevant.

## Documentation

- Update README or docstrings when behaviour changes.
- Document new environment variables, CLI flags, or config options.
- Add inline comments only when the _why_ isn't obvious from the code.
