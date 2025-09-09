import logging
import secrets
import socket
import threading
import time
import urllib.parse
import webbrowser

from flask import Flask, abort, request
import requests
from werkzeug.serving import make_server

from .config import CLIENT_ID, CLIENT_SECRET

# Configure logging for this script if not already configured by the host app
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# Configuration
OAUTH_PORT = 5000
REDIRECT_URI = f"http://localhost:{OAUTH_PORT}/callback"  # Must match Strava app callback domain
SCOPE = "read,activity:read_all"
PRINT_TOKENS = True  # Set to True to print full tokens after exchange

# Flask app and server state
app = Flask(__name__)
auth_code = None
auth_event = threading.Event()
expected_state = secrets.token_urlsafe(16)
_server = None


@app.route("/callback")
def callback():
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


def run_flask():
    global _server
    _server = make_server("localhost", OAUTH_PORT, app)
    _server.serve_forever()


# Build the Strava authorisation URL
params = {
    "client_id": CLIENT_ID,
    "response_type": "code",
    "redirect_uri": REDIRECT_URI,
    "scope": SCOPE,
    "approval_prompt": "force",
    "state": expected_state,
}
auth_url = "https://www.strava.com/oauth/authorize?" + urllib.parse.urlencode(params)


# Helper to wait for port to be open
def wait_for_port(port, host="localhost", timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


# Reset event and auth_code for repeated runs
auth_event.clear()
auth_code = None

# Start the local server and open the authorisation URL
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

logging.info(f"Waiting for Flask server to start on port {OAUTH_PORT}...")
if not wait_for_port(OAUTH_PORT):
    logging.error(f"Flask server did not start on port {OAUTH_PORT}")
    exit(1)

logging.info("Opening browser for authorisation...")
webbrowser.open(auth_url)

# Wait for the authorisation code (with timeout)
if not auth_event.wait(timeout=60):
    logging.error("Timeout waiting for authorisation code.")
    if _server:
        _server.shutdown()
    flask_thread.join(timeout=5)
    exit(1)

# Ensure auth_code is set before proceeding
if not auth_code:
    logging.error("Authorisation code was not received. Exiting.")
    if _server:
        _server.shutdown()
    flask_thread.join(timeout=5)
    exit(1)

logging.info("Authorisation code received.")

# Exchange code for tokens
token_url = "https://www.strava.com/oauth/token"
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
    if not all(k in tokens for k in ("access_token", "refresh_token", "expires_at")):
        logging.error(f"Token response missing expected keys: {tokens}")
    else:
        # Masked logging
        at = tokens.get("access_token") or ""
        rt = tokens.get("refresh_token") or ""
        if PRINT_TOKENS:
            logging.info("Access Token: %s", at)
            logging.info("Refresh Token: %s", rt)
            logging.info("Expires At: %s", tokens.get("expires_at"))
        else:
            logging.info(
                "Token exchange succeeded: access_token_len=%s refresh_token_len=%s",
                len(at),
                len(rt),
            )
except Exception as e:
    logging.error(f"Failed to exchange code for tokens: {e}")
finally:
    if _server:
        logging.info("Shutting down local OAuth server.")
        _server.shutdown()
    flask_thread.join(timeout=5)
