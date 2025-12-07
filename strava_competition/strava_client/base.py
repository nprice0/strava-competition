"""Shared Strava client helpers (auth, headers, session utilities)."""

from __future__ import annotations

import logging
import threading
from typing import Dict

from ..auth import get_access_token
from .. import config
from ..models import Runner

LOGGER = logging.getLogger(__name__)

# Per-runner locks to prevent race conditions during token rotation
_token_locks: Dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_runner_lock(runner_id: str) -> threading.Lock:
    """Get or create a lock for a specific runner."""
    with _locks_lock:
        if runner_id not in _token_locks:
            _token_locks[runner_id] = threading.Lock()
        return _token_locks[runner_id]


def ensure_runner_token(runner: Runner) -> None:
    """Ensure the runner has a valid access token, refreshing when needed."""

    if config.STRAVA_OFFLINE_MODE:
        if not getattr(runner, "access_token", None) and not getattr(
            runner, "_skip_token_logged", False
        ):
            LOGGER.info(
                "Skipping Strava token refresh for runner=%s (STRAVA_OFFLINE_MODE)",
                getattr(runner, "name", "?"),
            )
            setattr(runner, "_skip_token_logged", True)
        return

    # Use per-runner locking to prevent race conditions during token rotation
    runner_lock = _get_runner_lock(str(runner.strava_id))
    with runner_lock:
        # Re-check after acquiring lock (another thread may have refreshed)
        if getattr(runner, "access_token", None):
            return

        access_token, new_refresh_token = get_access_token(
            runner.refresh_token, runner_name=runner.name
        )
        runner.access_token = access_token
        if new_refresh_token and new_refresh_token != runner.refresh_token:
            runner.refresh_token = new_refresh_token
            try:
                from ..excel_writer import (
                    update_single_runner_refresh_token,
                )  # local import
                from ..config import INPUT_FILE

                update_single_runner_refresh_token(INPUT_FILE, runner)
            except Exception as exc:  # pragma: no cover - best-effort persistence
                LOGGER.debug(
                    "Failed to persist refresh token for runner %s: %s",
                    runner.name,
                    exc,
                    exc_info=True,
                )


def auth_headers(runner: Runner) -> Dict[str, str]:
    """Return bearer auth headers for the runner (token assumed valid)."""

    token = getattr(runner, "access_token", None) or ""
    return {"Authorization": f"Bearer {token}"}
