"""HTTP session factory for Strava API calls.

This module provides thread-safe session management. Each thread receives
its own `requests.Session` instance via thread-local storage, ensuring
concurrent API calls do not share mutable state (connection pools, cookies,
headers) which would otherwise cause race conditions.
"""

from __future__ import annotations

import threading

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import HTTP_POOL_CONNECTIONS, HTTP_POOL_MAXSIZE

__all__ = ["create_default_session", "get_default_session", "close_thread_session"]

_thread_local = threading.local()


def _build_retry() -> Retry:
    return Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )


def create_default_session() -> Session:
    """Create a new configured requests.Session.

    Each session is preconfigured with:
      - Connection pooling (size from config)
      - Automatic retries on transient server errors
      - Sensible default headers (gzip, JSON accept)

    Returns:
        A fresh, fully-configured Session instance.
    """
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=HTTP_POOL_CONNECTIONS,
        pool_maxsize=HTTP_POOL_MAXSIZE,
        max_retries=_build_retry(),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        }
    )
    return session


def get_default_session() -> Session:
    """Return the thread-local Strava session.

    Sessions are created lazily on first access per thread, ensuring
    thread safety while still benefiting from connection pooling within
    each thread's request sequence.

    Returns:
        The Session instance bound to the calling thread.
    """
    session: Session | None = getattr(_thread_local, "session", None)
    if session is None:
        session = create_default_session()
        _thread_local.session = session
    return session


def close_thread_session() -> None:
    """Close and remove the current thread's session.

    Call this to release connection pool resources when a thread is done
    making requests, or during test teardown to ensure clean state.
    """
    session: Session | None = getattr(_thread_local, "session", None)
    if session is not None:
        session.close()
        del _thread_local.session
