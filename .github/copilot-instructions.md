---
description: "Python coding conventions for production-ready code"
applyTo: "**/*.py"
---

## Code Quality Standards

All code must be **production-ready**. This means:

- Clean, readable, and self-documenting
- Fully typed with no `Any` escape hatches unless justified
- Covered by tests
- Free of linter, type checker, and security warnings
- Handles failures gracefully—no silent data loss
- Logs meaningful diagnostics without exposing secrets

## Python Style

- Follow PEP 8 strictly. Lines ≤ 88 characters (ruff default).
- 4-space indentation. No tabs.
- Use trailing commas in multi-line structures.
- Separate logical blocks with blank lines.

## Type Hints

- **All** functions, methods, and class attributes must have type hints.
- Use `typing` and `collections.abc` for complex types.
- Prefer `X | None` over `Optional[X]` (Python 3.10+).
- Use `TypedDict`, `NamedTuple`, or dataclasses for structured data—avoid raw dicts.

## Docstrings

Every public function, class, and module must have a docstring:

```python
def fetch_segment_efforts(
    segment_id: int,
    start_date: datetime,
    end_date: datetime,
) -> list[SegmentEffort]:
    """
    Fetch all efforts for a segment within a date window.

    Args:
        segment_id: Strava segment ID.
        start_date: Window start (inclusive).
        end_date: Window end (inclusive).

    Returns:
        List of efforts ordered by elapsed time.

    Raises:
        StravaAPIError: If the API request fails.
        RateLimitError: If rate limits are exceeded.
    """
```

Document:

- What the function does (not how)
- All parameters and return values
- Exceptions that may be raised
- Side effects if any

## Naming Conventions

| Element             | Style              | Example                         |
| ------------------- | ------------------ | ------------------------------- |
| Functions/variables | `snake_case`       | `fetch_activities`              |
| Classes             | `CapWords`         | `SegmentEffort`                 |
| Constants           | `UPPER_CASE`       | `MAX_RETRY_COUNT`               |
| Private members     | Leading underscore | `_parse_response`               |
| Type aliases        | `CapWords`         | `RunnerMap = dict[str, Runner]` |

## Imports

Group imports in this order, separated by blank lines:

1. Standard library
2. Third-party packages
3. Local modules

Use absolute imports. Avoid `from module import *`.

## Functions & Classes

- Keep functions short and focused—one responsibility per function.
- Prefer pure functions where possible; isolate side effects.
- Extract helpers when a block of code can be named and tested in isolation.
- Use dataclasses or `NamedTuple` over plain tuples or dicts for domain objects.
- Limit classes to a single responsibility; split large classes.

## Error Handling

- Raise explicit, descriptive exceptions with context.
- Never use bare `except:`—always catch specific exceptions.
- Use context managers (`with`) for resource cleanup.
- Fail fast: validate inputs early and raise immediately.
- Log exceptions with enough context to debug, but never log secrets.

```python
# Good
if not runner.refresh_token:
    raise ValueError(f"Runner '{runner.name}' is missing a refresh token")

# Bad
try:
    ...
except:
    pass
```

## Security

- **Never** hardcode secrets, tokens, or credentials.
- Load sensitive values from environment variables or `.env`.
- Validate and sanitise all external inputs (API responses, Excel data).
- Run `bandit` before committing; fix all warnings.
- Treat Strava tokens as secrets—never log, print, or expose them.
- Use HTTPS for all external requests.
- Set timeouts on all network calls to prevent hangs.

## Testing

- Write tests for **every** change that affects logic.
- Cover happy paths, edge cases, and error conditions.
- Keep tests deterministic—mock external services, control time, seed randomness.
- Use fixtures and factories to reduce test boilerplate.
- Aim for fast tests; slow tests should be marked and run separately.
- Test error paths explicitly—verify exceptions are raised with correct messages.
- Never commit tests that are skipped or xfailed without a linked issue.

Test file structure:

```
tests/
  test_<module>.py          # Unit tests matching source modules
  conftest.py               # Shared fixtures
```

## Quality Gates

Run these before every commit:

```bash
ruff check .                # Linting
ruff format --check .       # Formatting
mypy .                      # Type checking
bandit -q -r strava_competition  # Security
pytest -q                   # Tests
```

All must pass with zero warnings. No `# type: ignore` without a comment explaining why.

## Git & Pull Requests

- Keep commits small and atomic—one logical change per commit.
- Write clear commit messages: `Fix token refresh race condition`
- Include tests and documentation updates in the same PR as code changes.
- Link to issues where relevant.

## Documentation

- Update README or docstrings when behaviour changes.
- Document new environment variables, CLI flags, or config options.
- Add inline comments only when the _why_ isn't obvious from the code.

## Code Smells to Avoid

- Functions longer than 30 lines
- More than 3 levels of nesting
- Boolean parameters that change behaviour (`def process(data, fast=False)`)
- Magic numbers without named constants
- Mutable default arguments
- Global state
- Comments that repeat what the code says
- Catching exceptions only to re-raise without added context
- Ignoring return values from functions that can fail
- Using `print()` instead of proper logging

## Performance & Reliability

- Cache expensive API responses (this project uses `strava_cache/`).
- Respect Strava rate limits—use backoff and retry logic.
- Prefer batch operations over loops with individual API calls.
- Profile before optimising; don't guess at bottlenecks.
- Handle partial failures—if one runner fails, continue with others.

## API & Data Contracts

- Define clear interfaces with typed models (dataclasses, Pydantic, TypedDict).
- Validate API responses before processing—don't assume structure.
- Version any serialised formats (cache files, Excel output schemas).
- Document breaking changes in commit messages and changelogs.
