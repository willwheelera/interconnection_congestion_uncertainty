"""Fetch ERCOT actual load for a small collection of high-congestion snapshots.

The snapshot hours are read from the cached high-congestion table
(data/ercot/high_congestion_hours_base.csv), which marks the
highest base-case congestion hours of 2025. For each selected hour we pull
ERCOT actual system load by weather zone (endpoint /np6-345-cd/act_sys_load_by_wzn)
and keep the matching HourEnding row. The clock hour h in the congestion table
is treated as hour-beginning, so it maps to HourEnding h+1.

Requires a valid ERCOT token (run src/grid_data/ercot_auth.py) and the API env
vars (source gridstatus_api_key.sh).

Output: data/activsg2000_load_snapshots.csv with one row per snapshot holding the
zonal loads, the system Total (MW), and the congestion price/constraint.

Usage:
    source gridstatus_api_key.sh
    uv run python src/texas/fetch_load_snapshots.py --n 8
"""

import argparse
from pathlib import Path

import pandas as pd

from src.paths import REPO_ROOT
from src.ercot.data_paths import ERCOT_DATA
from src.ercot.fetch_ercot import build_client

CONGESTION_CSV = ERCOT_DATA / "high_congestion_hours_base.csv"
OUT = REPO_ROOT / "data" / "activsg2000_load_snapshots.csv"
LOAD_ENDPOINT = "/np6-345-cd/act_sys_load_by_wzn"
ZONE_COLS = ["Coast", "East", "FarWest", "North", "NorthC", "Southern", "SouthC", "West", "Total"]


def select_snapshots(n: int) -> pd.DataFrame:
    """Return the n highest-congestion hours with a mapped HourEnding label."""
    df = pd.read_csv(CONGESTION_CSV).sort_values("max_price", ascending=False).head(n).copy()
    df["hour_ending"] = ((df["hour_of_day"] + 1)).map(lambda h: f"{h:02d}:00")
    return df.reset_index(drop=True)


def fetch_day_load(client, date: str) -> pd.DataFrame:
    """Fetch the 24-hour weather-zone actual load table for one operating day."""
    return client.hit_ercot_api(
        endpoint=LOAD_ENDPOINT, verbose=False,
        operatingDayFrom=date, operatingDayTo=date,
    )


def collect_loads(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Fetch load for each snapshot's date and keep the matching HourEnding row."""
    client = build_client()
    day_cache: dict[str, pd.DataFrame] = {}
    rows = []
    for _, s in snapshots.iterrows():
        date = s["date"]
        if date not in day_cache:
            day_cache[date] = fetch_day_load(client, date)
        day = day_cache[date]
        match = day[day["HourEnding"] == s["hour_ending"]].iloc[0]
        row = {"timestamp": s["hour"], "date": date, "hour_ending": s["hour_ending"],
               "max_price": s["max_price"], "max_constraint": s["max_constraint"]}
        row.update({c: float(match[c]) for c in ZONE_COLS})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    snapshots = select_snapshots(args.n)
    loads = collect_loads(snapshots)
    OUT.parent.mkdir(exist_ok=True)
    loads.to_csv(OUT, index=False)
    print(loads[["timestamp", "max_constraint", "max_price", "Total"]].to_string(index=False))
    print(f"\nwrote {len(loads)} snapshots -> {OUT}")


if __name__ == "__main__":
    main()
