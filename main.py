"""Deprecated wrapper entry point.

This file now simply delegates to the packaged application to avoid users
accidentally running the legacy (flat) module versions. Use:
    python -m strava_competition
directly. Left in place for backwards compatibility.
"""
from __future__ import annotations

import logging

from strava_competition.main import main as _pkg_main


def main():  # pragma: no cover - thin wrapper
    logging.warning(
        "Deprecated: running root-level main.py. Delegating to package entry point. Use 'python -m strava_competition' instead."
    )
    _pkg_main()

if __name__ == "__main__":
    main()