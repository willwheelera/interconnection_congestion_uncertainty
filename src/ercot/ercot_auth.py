"""Authenticate to the ERCOT public API and cache a short-lived bearer token.

Run this yourself: it prompts for your ERCOT password (never stored), exchanges
it for a ~1-hour bearer token, and writes that token to a file for query scripts
to read. The password is discarded after authentication.

Username and subscription key are read from the environment:
    ERCOT_API_USERNAME
    ERCOT_PUBLIC_API_SUBSCRIPTION_KEY

Usage:
    uv run python src/grid_data/ercot_auth.py
"""

import getpass
import json
import os

from gridstatus.ercot_api.ercot_api import ErcotAPI

TOKEN_PATH = os.path.expanduser("~/.ercot_token.json")


def authenticate():
    """Prompt for the password and exchange it for a bearer token."""
    password = getpass.getpass("ERCOT API password: ")
    api = ErcotAPI(password=password)
    api.get_token()
    return api.token, api.token_expiry


def write_token(token, expiry):
    """Write token and expiry to TOKEN_PATH with owner-only permissions."""
    with open(TOKEN_PATH, "w") as f:
        json.dump({"token": token, "expiry": expiry}, f)
    os.chmod(TOKEN_PATH, 0o600)


def main():
    token, expiry = authenticate()
    write_token(token, expiry)
    print(f"Token cached to {TOKEN_PATH} (expires in ~1 hour).")


if __name__ == "__main__":
    main()
