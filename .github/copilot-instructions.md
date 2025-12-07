---
description: "Python coding conventions and guidelines"
applyTo: "**/*.py"
---

## Python Coding Conventions

- Follow PEP 8 and keep functions focused, descriptive, and well documented.
- Always include type hints (`typing`/`collections.abc`) and docstrings that explain
  inputs, outputs, side effects, and raised exceptions.
- Provide explanatory comments when behaviour is non-obvious or design
  trade-offs require justification.
- Keep lines ≤ 79 characters, use 4-space indentation, and separate logical
  blocks with blank lines.

## General development guidance

- Optimize for readability and maintainability. Extract helpers when a block of
  code can be named and tested in isolation.
- Prefer explicit data structures and control flow over clever or implicit
  constructs.
- Keep modules cohesive. Split large modules into smaller ones with clear
  responsibilities when they grow unwieldy.

## Testing expectations

- Add or update pytest coverage for every change that affects logic, including
  edge cases (empty inputs, invalid data, boundary conditions) and happy paths.
- Keep tests deterministic by controlling randomness, external services, and
  timing. Use fixtures and mocks liberally.
- Run the standard quality gates (`pytest`, `ruff`, `mypy`, `bandit`, etc.)
  before committing. Document any temporary skips or xfails and link to a
  follow-up issue when possible.

## Error handling & resilience

- Raise explicit exceptions with helpful context. Avoid bare `except` blocks
  and always clean up resources using context managers or `try/finally`.
- Log enough information to debug issues without leaking secrets or sensitive
  data. Keep logging consistent with the existing project style.

## Contribution workflow

- Keep commits small and focused. Mention related issues and the tests you ran
  in commit messages or pull requests.
- Document new environment variables, configuration knobs, CLI flags, or file
  formats in both code comments and user-facing docs.
- If a change affects user-visible behaviour, update relevant documentation in
  the same pull request.
- Flag TODOs with context (`TODO(name): reason`) and create follow-up tickets
  for deferred work.

## Example of proper documentation

Follow Google or NumPy docstring style, whichever the project already uses.

```python
import math


def calculate_area(radius: float) -> float:
    """
    Calculate the area of a circle given the radius.

    Parameters:
        radius (float): The radius of the circle.

    Returns:
        float: The area of the circle, calculated as π * radius^2.
    """
    return math.pi * radius ** 2
```

## Naming, imports, and structure

- Use `snake_case` for functions/variables, `CapWords` for classes, and
  `UPPER_CASE` for module-level constants.
- Group imports as standard library, third-party, and local modules—each group
  separated by a blank line. Prefer absolute imports.
- Keep helper functions private unless they are intended entry points. Aim for
  cohesive modules with minimal cross-dependencies.

## Version control & reviews

- Write descriptive commit messages (e.g., `Fix cache eviction race`). Mention
  the tests or tooling executed.
- Keep pull requests focused on a single change set that includes code, tests,
  and docs.
- During reviews, check for adherence to these guidelines and request follow-up
  issues for known gaps instead of silently accepting them.

## Tooling reminders

- Run the project's required linters, type checkers, security scanners, and
  tests before pushing. Capture the output in CI when possible.
- Use `pre-commit` hooks (or equivalent) to keep formatting and linting
  consistent across contributors. If the repo provides a hooks config, run
  `pre-commit install` after cloning.
