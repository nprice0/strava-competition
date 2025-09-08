import requests
import logging
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from auth import get_access_token
from config import STRAVA_BASE_URL, STRAVA_OAUTH_URL

# Reusable HTTP session with retries and backoff for reliability and performance
_session = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
)
_adapter = HTTPAdapter(max_retries=_retry)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

DEFAULT_TIMEOUT = 15

def get_segment_efforts(runner, segment_id, start_date, end_date):
    """Fetch all efforts for a segment within dates for a runner, with pagination and retry.

    - Reuses cached access_token on the runner.
    - Refreshes token once on 401 and retries.
    - Paginates through all results (per_page=200).
    - Applies retry/backoff and timeouts.
    """
    def auth_headers():
        return {"Authorization": f"Bearer {runner.access_token}"}

    def ensure_token():
        if not runner.access_token:
            access_token, new_refresh_token = get_access_token(runner.refresh_token, runner_name=runner.name)
            runner.access_token = access_token
            if new_refresh_token and new_refresh_token != runner.refresh_token:
                runner.refresh_token = new_refresh_token

    def fetch_page(page: int):
        url = f"{STRAVA_BASE_URL}/segment_efforts"
        params = {
            "segment_id": segment_id,
            "start_date_local": start_date.isoformat(),
            "end_date_local": end_date.isoformat(),
            "per_page": 200,
            "page": page,
        }
        resp = _session.get(url, headers=auth_headers(), params=params, timeout=DEFAULT_TIMEOUT)
        # Rate limit awareness (simple): back off if 429
        if resp.status_code == 429:
            logging.warning("Rate limit hit. Backing off 15 seconds...")
            time.sleep(15)
            resp = _session.get(url, headers=auth_headers(), params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        # Optional: inspect rate limit headers
        usage = resp.headers.get("X-RateLimit-Usage")
        limit = resp.headers.get("X-RateLimit-Limit")
        if usage and limit:
            try:
                short_used = int(usage.split(",")[0])
                short_limit = int(limit.split(",")[0])
                if short_used >= max(short_limit - 5, 0):
                    logging.info("Approaching short-window rate limit; sleeping 15s to avoid throttling.")
                    time.sleep(15)
            except Exception:
                pass
        return resp.json()

    try:
        ensure_token()
        all_efforts = []
        page = 1
        while True:
            try:
                data = fetch_page(page)
            except requests.exceptions.HTTPError as e:
                # One-time 401 retry by refreshing token
                if e.response is not None and e.response.status_code == 401:
                    logging.info(f"401 for runner {runner.name}. Refreshing token and retrying page {page}.")
                    runner.access_token = None
                    ensure_token()
                    data = fetch_page(page)
                else:
                    raise

            if not data:
                break
            all_efforts.extend(data)
            if len(data) < 200:
                break
            page += 1
        return all_efforts
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            logging.warning(f"Skipping runner {runner.name}: Unauthorized (invalid/expired token after retry)")
            return None
        elif e.response is not None and e.response.status_code == 402:
            logging.warning(f"Runner {runner.name}: Payment required for segment efforts. Skipping.")
            return None
        else:
            logging.error(f"Error for runner {runner.name}: {e}")
            return None