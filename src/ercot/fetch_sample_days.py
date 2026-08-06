"""Download Phase 2 settled/SCED prices for the suggested high-congestion days.

`select_high_congestion_periods.py` writes ercot_data/sample_days.csv: delivery
dates chosen to isolate base-case vs contingency congestion. This pulls the
node/zone/hub LMP (np6-788) and settlement-point prices (np6-905) for each of
those days so we can check whether a high shadow price shows up as a high settled
price, and under which regime. One parquet per dataset per day; days already on
disk are skipped.

Run ercot_auth.py first for the bearer token.

Usage:
    uv run python src/grid_data/fetch_sample_days.py
"""

from pathlib import Path

import pandas as pd

from src.ercot.fetch_ercot import PHASE2_DATASETS, build_client, fetch_days

from src.ercot.data_paths import SAMPLE_DAYS_CSV


def load_sample_days(path):
    """Read the suggested delivery dates (one per row) from the selection CSV."""
    return pd.read_csv(path)["date"].tolist()


def main():
    dates = load_sample_days(SAMPLE_DAYS_CSV)
    client = build_client()
    print(f"Fetching {len(PHASE2_DATASETS)} datasets for {len(dates)} days")
    fetch_days(client, PHASE2_DATASETS, dates)


if __name__ == "__main__":
    main()
