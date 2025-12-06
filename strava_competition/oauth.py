"""Local Strava OAuth helper with optional token printing."""

from __future__ import annotations

import argparse
import logging
import secrets
import socket
import threading
import time
import urllib.parse
import webbrowser
from typing import Optional

from flask import Flask, abort, request
from flask.typing import ResponseReturnValue
import requests
from werkzeug.serving import BaseWSGIServer, make_server

from .config import CLIENT_ID, CLIENT_SECRET

# Configure logging for this script if not already configured by the host app
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )

# Configuration
OAUTH_PORT = 5000
REDIRECT_URI = (
    f"http://localhost:{OAUTH_PORT}/callback"  # Must match Strava app callback domain
)
SCOPE = "read,activity:read_all"
PRINT_TOKENS_DEFAULT = False

# Flask app and server state
app = Flask(__name__)
auth_code: Optional[str] = None
auth_event = threading.Event()
expected_state = secrets.token_urlsafe(16)
_server: BaseWSGIServer | None = None


def _mask_token(token: str, visible: int = 4) -> str:
    """Return ``token`` with all but the trailing ``visible`` chars masked."""

    if not token:
        return ""
    visible = max(0, visible)
    if visible == 0:
        return "*" * len(token)
    hidden_length = max(len(token) - visible, 0)
    if hidden_length == 0:
        return token
    return ("*" * hidden_length) + token[-visible:]


@app.route("/callback")
def callback() -> ResponseReturnValue:
    global auth_code
    state = request.args.get("state")
    if not state or state != expected_state:
        logging.error("Invalid OAuth state received; possible CSRF. Aborting.")
        abort(400, description="Invalid state")
    auth_code = request.args.get("code")
    logging.info("Authorisation code received via callback.")
    # Signal that the code is received
    auth_event.set()
    return "Authorisation received! You can close this window now."


def run_flask() -> None:
    global _server
    _server = make_server("localhost", OAUTH_PORT, app)
    _server.serve_forever()


def _build_auth_url() -> str:
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "approval_prompt": "force",
        "state": expected_state,
    }
    return "https://www.strava.com/oauth/authorize?" + urllib.parse.urlencode(params)


def wait_for_port(port: int, host: str = "localhost", timeout: int = 10) -> bool:
    """Return True once ``host:port`` accepts TCP connections or timeout elapses."""

    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _reset_oauth_state() -> None:
    """Reset state/flags so the OAuth helper can be rerun in one process."""

    global auth_code, expected_state
    auth_event.clear()
    auth_code = None
    expected_state = secrets.token_urlsafe(16)


def _exchange_code_for_tokens(print_tokens: bool) -> None:
    """Exchange the captured code for tokens and log according to policy."""

    if not auth_code:
        raise RuntimeError(
            "OAuth exchange attempted without an authorisation code present"
        )
    # Bandit B105 false positive: this is the documented Strava OAuth endpoint.
    token_url = "https://www.strava.com/oauth/token"  # nosec B105
    try:
        logging.info("Exchanging authorisation code for tokens...")
        response = requests.post(
            token_url,
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code": auth_code,
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI,
            },
            timeout=15,
        )
        response.raise_for_status()
        tokens = response.json()
        if not all(
            key in tokens for key in ("access_token", "refresh_token", "expires_at")
        ):
            logging.error("Token response missing expected keys: %s", tokens)
            return
        access_token = tokens.get("access_token") or ""
        refresh_token = tokens.get("refresh_token") or ""
        expires_at = tokens.get("expires_at")
        if print_tokens:
            logging.warning("Printing raw Strava tokens to stdout. Handle with care!")
            logging.info("Access Token: %s", access_token)
            logging.info("Refresh Token: %s", refresh_token)
            logging.info("Expires At: %s", expires_at)
        else:
            logging.info(
                "Token exchange succeeded: access_token=%s refresh_token=%s expires_at=%s",
                _mask_token(access_token),
                _mask_token(refresh_token),
                expires_at,
            )
    except (
        requests.exceptions.RequestException,
        ValueError,
        KeyError,
    ) as exc:  # pragma: no cover - network failures
        logging.error("Failed to exchange code for tokens: %s", exc)


def start_oauth_flow(*, print_tokens: bool, wait_timeout: int = 60) -> None:
    """Run the OAuth flow end-to-end, optionally printing tokens."""

    _reset_oauth_state()
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    logging.info("Waiting for Flask server to start on port %s...", OAUTH_PORT)
    if not wait_for_port(OAUTH_PORT):
        logging.error("Flask server did not start on port %s", OAUTH_PORT)
        if _server:
            _server.shutdown()
        flask_thread.join(timeout=5)
        raise SystemExit(1)

    auth_url = _build_auth_url()
    logging.info("Opening browser for authorisation...")
    webbrowser.open(auth_url)

    if not auth_event.wait(timeout=wait_timeout):
        logging.error("Timeout waiting for authorisation code.")
        if _server:
            _server.shutdown()
        flask_thread.join(timeout=5)
        raise SystemExit(1)

    if not auth_code:
        logging.error("Authorisation code was not received. Exiting.")
        if _server:
            _server.shutdown()
        flask_thread.join(timeout=5)
        raise SystemExit(1)

    logging.info("Authorisation code received.")
    try:
        _exchange_code_for_tokens(print_tokens)
    finally:
        if _server:
            logging.info("Shutting down local OAuth server.")
            _server.shutdown()
        flask_thread.join(timeout=5)


def _parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Strava OAuth helper")
    parser.add_argument(
        "--print-tokens",
        action="store_true",
        default=PRINT_TOKENS_DEFAULT,
        help="Print raw access/refresh tokens once exchanged (defaults to masked logging)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Seconds to wait for browser authorisation before exiting",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point when executing ``python -m strava_competition.oauth``."""

    args = _parse_args()
    start_oauth_flow(print_tokens=args.print_tokens, wait_timeout=args.timeout)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
