---
description: "Python coding conventions for production-ready code"
applyTo: "**/*.py"
---

# Python Conventions

Target: Python 3.10+. Tooling: ruff (lint + format), mypy (strict), bandit, pytest.

## Style

- Follow PEP 8 strictly. Lines ≤ 88 characters (ruff default).
- 4-space indentation. No tabs.
- Use trailing commas in multi-line structures.
- Separate logical blocks with blank lines.

## Type Hints

- **All** functions, methods, and class attributes must have type hints.
- Use `typing` and `collections.abc` for complex types.
- Prefer `X | None` over `Optional[X]` (Python 3.10+).
- Use `TypedDict`, `NamedTuple`, or dataclasses for structured data—avoid raw dicts.
- `dict[str, Any]` is acceptable for raw Strava API JSON responses that are passed through without field-level access.
- No `# type: ignore` without a comment explaining why.

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

Use relative imports within the `strava_competition` package (e.g. `from ..models import Runner`).
Use absolute imports for standard library and third-party packages.
Avoid `from module import *`.

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

## Security (Python specifics)

- Load sensitive values from environment variables or `.env`—never hardcode.
- Validate and sanitise all external inputs (API responses, Excel data).
- Run `bandit` before committing; fix all warnings.
- Set timeouts on all network calls to prevent hangs.
- Use HTTPS for all external requests.

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

All must pass with zero warnings.

## API & Data Contracts

- Define clear interfaces with typed models (dataclasses, Pydantic, TypedDict).
- Validate API responses before processing—don't assume structure.
- Version any serialised formats (cache files, Excel output schemas).
- Document breaking changes in commit messages and changelogs.

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
- Using `print()` for diagnostics instead of `logging` (use `print()` only for user-facing CLI output in tools)
