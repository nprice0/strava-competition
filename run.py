#!/usr/bin/env python3
"""Entry point for the Strava Segment Competition Tool.

This module configures logging and runs the main application. When packaged
as a standalone executable, logs are written to a 'logs' folder next to
the executable for easy troubleshooting.

Usage:
    python run.py
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

LOG_FORMAT_CONSOLE = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
LOG_FORMAT_FILE = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"


def _get_base_directory() -> Path:
    """Determine the base directory for the application.

    Returns:
        Path to the directory containing the executable or script.
    """
    if getattr(sys, "frozen", False):
        # Running as compiled executable (PyInstaller)
        return Path(sys.executable).parent
    return Path(__file__).parent


def setup_logging() -> Path:
    """Configure dual logging to console and file.

    Creates a timestamped log file in the 'logs' directory. Console output
    shows INFO level, while the file captures DEBUG level for diagnostics.

    Returns:
        Path to the created log file.
    """
    base_dir = _get_base_directory()
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"strava_competition_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console: INFO level for user visibility
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT_CONSOLE))
    root_logger.addHandler(console_handler)

    # File: DEBUG level for full diagnostics
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT_FILE))
    root_logger.addHandler(file_handler)

    return log_file


def _wait_for_keypress() -> None:
    """Wait for user input before exiting (for interactive terminals)."""
    try:
        input("\nPress Enter to exit...")
    except EOFError:
        # Non-interactive environment
        pass


def run() -> int:
    """Run the application with error handling.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Import here to ensure logging is configured first
    from strava_competition.main import main

    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Log file: %s", log_file)

    try:
        main()
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        print(f"\n\nERROR: {exc}")
        print(f"\nFull details saved to: {log_file}")
        _wait_for_keypress()
        return 1


if __name__ == "__main__":
    sys.exit(run())
