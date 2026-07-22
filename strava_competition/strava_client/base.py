"""Shared Strava client helpers (auth, headers, session utilities)."""

from __future__ import annotations

import logging
import threading

from ..auth import get_access_token
from .. import config
from ..models import Runner

LOGGER = logging.getLogger(__name__)

# Per-runner locks to prevent race conditions during token rotation
_token_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_runner_lock(runner_id: str) -> threading.Lock:
    """Get or create a lock for a specific runner."""
    with _locks_lock:
        if runner_id not in _token_locks:
            _token_locks[runner_id] = threading.Lock()
        return _token_locks[runner_id]


def ensure_runner_token(runner: Runner, *, persist: bool = True) -> None:
    """Ensure the runner has a valid access token, refreshing when needed.

    Args:
        runner: Participant whose access token should be ensured.
        persist: When True (default) a rotated refresh token is written back to
            the workbook immediately for crash-safety. Callers that perform
            their own batch persistence afterwards (e.g. the startup pre-warm)
            should pass ``persist=False`` to avoid an O(N^2) rewrite of the
            Runners sheet — one full rewrite per rotated runner.
    """

    if config._cache_mode_offline:
        if not runner.access_token and not runner._skip_token_logged:
            LOGGER.info(
                "Skipping Strava token refresh for runner=%s (STRAVA_API_CACHE_MODE=offline)",
                runner.name,
            )
            runner._skip_token_logged = True
        return

    # Use per-runner locking to prevent race conditions during token rotation
    runner_lock = _get_runner_lock(str(runner.strava_id))
    with runner_lock:
        # Re-check after acquiring lock (another thread may have refreshed)
        if runner.access_token:
            return

        access_token, new_refresh_token = get_access_token(
            runner.refresh_token, runner_name=runner.name
        )
        runner.access_token = access_token
        if new_refresh_token and new_refresh_token != runner.refresh_token:
            runner.refresh_token = new_refresh_token
            if persist:
                _persist_rotated_token(runner)


def _persist_rotated_token(runner: Runner) -> None:
    """Best-effort immediate write-back of a rotated refresh token."""

    try:
        from ..excel_writer import update_single_runner_refresh_token  # local import
        from ..config import INPUT_FILE

        update_single_runner_refresh_token(INPUT_FILE, runner)
    except Exception as exc:  # pragma: no cover - best-effort persistence
        LOGGER.debug(
            "Failed to persist refresh token for runner %s: %s",
            runner.name,
            exc,
            exc_info=True,
        )


def auth_headers(runner: Runner) -> dict[str, str]:
    """Return bearer auth headers for the runner (token assumed valid)."""

    token = runner.access_token or ""
    return {"Authorization": f"Bearer {token}"}
