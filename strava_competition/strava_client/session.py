"""HTTP session factory for Strava API calls."""

from __future__ import annotations

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import HTTP_POOL_CONNECTIONS, HTTP_POOL_MAXSIZE

__all__ = ["create_default_session", "get_default_session"]


def _build_retry() -> Retry:
    return Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )


def create_default_session() -> Session:
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


_DEFAULT_SESSION = create_default_session()


def get_default_session() -> Session:
    """Return the shared default Strava session."""

    return _DEFAULT_SESSION
