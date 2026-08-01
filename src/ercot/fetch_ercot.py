"""Fetch ERCOT real-time price and shadow-price datasets via the public API.

Uses the short-lived bearer token written by `ercot_auth.py` (run that first), so
no password is needed. If the token is missing or expired, this errors and asks
you to re-authenticate.

Datasets fetched (saved as parquet under OUT_DIR):
    /np6-787-cd/lmp_electrical_bus         LMP by electrical bus (SCED, 5-min)
    /np6-788-cd/lmp_node_zone_hub          LMP by node/zone/hub (SCED, 5-min)
    /np6-86-cd/shdw_prices_bnd_trns_const  Shadow prices on binding constraints (SCED)
    /np6-905-cd/spp_node_zone_hub          Settlement point prices (RT 15-min)

Username and subscription key are read from the environment:
    ERCOT_API_USERNAME
    ERCOT_PUBLIC_API_SUBSCRIPTION_KEY

Usage:
    uv run python src/grid_data/fetch_ercot.py [--dry-run]
"""

import argparse
import json
import os
import time

import pandas as pd

from gridstatus.ercot_api.ercot_api import ErcotAPI

from ercot.data_paths import ERCOT_DATA

TOKEN_PATH = os.path.expanduser("~/.ercot_token.json")
OUT_DIR = str(ERCOT_DATA)

# Each dataset: a save name, its endpoint, and which date filter the endpoint
# accepts. "sced" uses SCEDTimestampFrom/To (ISO datetimes); "delivery" uses
# deliveryDateFrom/To (calendar dates).
DATASETS = [
    ("lmp_electrical_bus", "/np6-787-cd/lmp_electrical_bus", "sced"),
    ("lmp_node_zone_hub", "/np6-788-cd/lmp_node_zone_hub", "sced"),
    ("shadow_prices", "/np6-86-cd/shdw_prices_bnd_trns_const", "sced"),
    ("spp_node_zone_hub", "/np6-905-cd/spp_node_zone_hub", "delivery"),
]

# Phase 1 scan: shadow prices are the congestion signal and small enough to pull
# for a whole year at once. The LMP/SPP tables (per-bus/point, 5-15 min) are
# drill-down-scale and reserved for targeted critical-day windows.
SCAN_DATASETS = [d for d in DATASETS if d[0] == "shadow_prices"]

# Phase 2 drill-down: settled/SCED prices for individual flagged days. Cheap
# node/zone/hub LMP + settlement-point prices; the ~19k-bus np6-787 feed is left
# out (reserved for targeted spatial-gradient work, not whole days).
PHASE2_DATASETS = [d for d in DATASETS
                   if d[0] in ("lmp_node_zone_hub", "spp_node_zone_hub")]


def load_token():
    """Read (token, expiry) from TOKEN_PATH; error if the token is expired."""
    with open(TOKEN_PATH) as f:
        cached = json.load(f)
    if time.time() >= cached["expiry"]:
        raise RuntimeError("Cached token expired; re-run src/grid_data/ercot_auth.py.")
    return cached["token"], cached["expiry"]


def build_client():
    """Build an ErcotAPI client backed by the cached token (no password used)."""
    token, expiry = load_token()
    client = ErcotAPI(password="unused")
    client.token = token
    client.token_expiry = expiry
    return client


def date_filters(time_kind, start, end):
    """Build the endpoint-specific date filter params for [start, end]."""
    format_str = "%Y-%m-%dT%H:%M:%S" if time_kind == "sced" else "%Y-%m-%d"
    key_prefix = "SCEDTimestamp" if time_kind == "sced" else "deliveryDate"
    return {
        f"{key_prefix}From": start.strftime(format_str),
        f"{key_prefix}To": end.strftime(format_str),
    }


def fetch_dataset(client, endpoint, time_kind, start, end, max_pages=None):
    """Fetch one dataset over [start, end] and return it as a DataFrame.

    If max_pages is set, pagination stops early (the result may be partial).
    """
    params = date_filters(time_kind, start, end)
    return client.hit_ercot_api(
        endpoint=endpoint, verbose=True, max_pages=max_pages, **params
    )


def save_dataset(df, name, tag):
    """Write a fetched dataset to OUT_DIR as parquet, tagged by the caller."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{tag}.parquet")
    df.to_parquet(path)
    print(f"  saved {len(df)} rows -> {path}")


def fetch_window(client, datasets, start, end, tag):
    """Fetch the given datasets over [start, end], saving one parquet per
    dataset tagged by `tag`. Skips datasets already saved for this tag.
    """
    for name, endpoint, time_kind in datasets:
        path = os.path.join(OUT_DIR, f"{name}_{tag}.parquet")
        if os.path.exists(path):
            print(f"Skipping {name} {tag} (already saved)")
            continue
        print(f"Fetching {name} [{start} .. {end}]")
        df = fetch_dataset(client, endpoint, time_kind, start, end)
        save_dataset(df, name, tag)


def fetch_window_dryrun(client, datasets, start, end, max_pages=1):
    """Fetch a capped slice of each dataset and report sizes; saves nothing.

    A plumbing check for auth, the date filter, and rough row counts before a
    real fetch.
    """
    for name, endpoint, time_kind in datasets:
        df = fetch_dataset(client, endpoint, time_kind, start, end, max_pages)
        print(f"  dry run {name}: {len(df)} rows in <= {max_pages} page(s)")


def window(date, duration):
    """[start, end] Timestamps spanning one calendar `duration` from `date`.

    duration is "day" or "year"; the window runs from midnight US/Central to one
    second before the next period begins.
    """
    start = pd.Timestamp(date, tz="US/Central")
    span = pd.DateOffset(days=1) if duration == "day" else pd.DateOffset(years=1)
    return start, start + span - pd.Timedelta(seconds=1)


def fetch_days(client, datasets, dates):
    """Fetch `datasets` for each calendar date in `dates`.

    One parquet per dataset per day, tagged by ISO date ({name}_{YYYY-MM-DD}),
    skipping days already on disk.
    """
    for date in dates:
        start, end = window(date, "day")
        fetch_window(client, datasets, start, end, str(date))


def scan_year(year=2025, dry_run=False):
    """Fetch the Phase 1 scan datasets for a full year in a single window.

    Shadow prices are small enough to pull in one call and save as one file per
    dataset, tagged by year. dry_run reports sizes without writing.
    """
    client = build_client()
    start, end = window(f"{year}-01-01", "year")
    if dry_run:
        fetch_window_dryrun(client, SCAN_DATASETS, start, end)
    else:
        fetch_window(client, SCAN_DATASETS, start, end, str(year))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report sizes from a capped fetch without writing parquet.",
    )
    return parser.parse_args()


def example(dry_run=False):
    """Pull a one-hour window across all four endpoints to exercise the plumbing."""
    client = build_client()
    # Example historical window: a summer afternoon hour in 2025 (US/Central).
    start = pd.Timestamp("2025-07-15 15:00", tz="US/Central")
    end = start + pd.Timedelta(hours=1)
    if dry_run:
        fetch_window_dryrun(client, DATASETS, start, end)
    else:
        fetch_window(client, DATASETS, start, end, start.strftime("%Y_%m_%d_%H%M"))


if __name__ == "__main__":
    scan_year(2025, dry_run=parse_args().dry_run)
