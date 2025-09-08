import logging
import requests
from config import CLIENT_ID, CLIENT_SECRET, STRAVA_OAUTH_URL

def get_access_token(refresh_token: str, runner_name: str | None = None):
    url = STRAVA_OAUTH_URL
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }
    # Mask sensitive values in logs
    masked_refresh = (refresh_token[-4:] if refresh_token else "")
    if runner_name:
        logging.info(
            f"Requesting access token for runner '{runner_name}' with refresh token ending: ****{masked_refresh}"
        )
    else:
        logging.info(f"Requesting access token with refresh token ending: ****{masked_refresh}")
    logging.debug(f"Token URL: {url}")
    # Do not log full payload with secrets; log safe subset
    logging.debug({
        "client_id": CLIENT_ID,
        "grant_type": payload["grant_type"],
    })
    response = requests.post(url, data=payload, timeout=15)
    logging.info(f"Response status: {response.status_code}")
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTPError: {e}")
        logging.error(f"Response body: {response.text}")
        raise
    data = response.json()
    # Do not log full tokens; only lengths and endings for traceability
    at = data.get("access_token")
    rt = data.get("refresh_token")
    logging.info(
        "Token response received: access_token_len=%s refresh_token_len=%s",
        len(at) if at else 0,
        len(rt) if rt else 0,
    )
    return data.get("access_token"), data.get("refresh_token")