"""Fetch ERCOT actual wind and solar output for the load-snapshot hours.

Each load snapshot (data/activsg2000_load_snapshots.csv) is a specific ERCOT hour.
To make renewable output consistent with that timestamp, this pulls the hourly
wind and solar production reports for the same hours:

    /np4-742-cd/wpp_hrly_actual_fcast_geo   wind, by geographic region
    /np4-745-cd/spp_hrly_actual_fcast_geo   solar, by geographic region

Both reports give, per region, the actual generation (GEN) and the available
capacity ceiling (COP HSL) for the hour. COP HSL -- not GEN -- is the quantity we
keep for the model: it is the weather-driven availability envelope before any
congestion curtailment, so the DCOPF is free to rediscover congestion rather than
inherit ERCOT's realized curtailment. Regional columns are kept for a later
region-resolved refinement; the current builder uses only the system-wide value.

Requires a valid ERCOT token (run src/grid_data/ercot_auth.py) and the API env
vars (source gridstatus_api_key.sh).

Output: data/activsg2000_renewable_snapshots.csv, one row per snapshot.

Usage:
    source gridstatus_api_key.sh
    uv run python src/texas/fetch_renewable_snapshots.py
"""

from pathlib import Path

import pandas as pd

from paths import REPO_ROOT
from ercot.fetch_ercot import build_client

SNAPSHOTS = REPO_ROOT / "data" / "activsg2000_load_snapshots.csv"
OUT = REPO_ROOT / "data" / "activsg2000_renewable_snapshots.csv"

WIND_ENDPOINT = "/np4-742-cd/wpp_hrly_actual_fcast_geo"
SOLAR_ENDPOINT = "/np4-745-cd/spp_hrly_actual_fcast_geo"

# Report regions (system-wide first). ERCOT names wind and solar regions
# differently; both include a SystemWide aggregate used for the current CF.
WIND_REGIONS = ["SystemWide", "Panhandle", "Coastal", "South", "West", "North"]
SOLAR_REGIONS = ["SystemWide", "CenterWest", "NorthWest", "FarWest", "FarEast",
                 "SouthEast", "CenterEast"]


def norm(name) -> str:
    """Uppercase, strip spaces and underscores -- for matching report columns."""
    return "".join(str(name).split()).upper().replace("_", "")


def pick(row: pd.Series, target: str) -> float:
    """Value of the row column whose normalized name equals `target` (normalized)."""
    want = norm(target)
    for col in row.index:
        if norm(col) == want:
            return float(row[col])
    raise KeyError(f"no column matching {target!r} in {list(row.index)}")


def fetch_day(client, endpoint: str, date: str) -> pd.DataFrame:
    """Fetch one operating day of an hourly wind/solar production report."""
    return client.hit_ercot_api(
        endpoint=endpoint, verbose=False,
        deliveryDateFrom=date, deliveryDateTo=date,
    )


def settled_row(day: pd.DataFrame, hour_ending: int) -> pd.Series:
    """The realized posting for an hour: latest PostedDatetime with an actual GEN.

    Each day carries ~200 rolling postings per hour (week-ahead forecasts with a
    null GEN, then the realized values once the hour passes). The last posting
    holding a non-null system-wide GEN is the settled actual.
    """
    sub = day[(day["HourEnding"] == hour_ending) & day["GenSystemWide"].notna()]
    return sub.sort_values("PostedDatetime").iloc[-1]


def region_values(row: pd.Series, fuel: str, regions: list[str]) -> dict:
    """Per-region GEN and COP HSL for one fuel, plus system-wide HSL."""
    out = {}
    for r in regions:
        out[f"{fuel}_gen_{r.lower()}"] = pick(row, f"Gen{r}")
        out[f"{fuel}_cophsl_{r.lower()}"] = pick(row, f"COPHSL{r}")
    out[f"{fuel}_hsl_systemwide"] = pick(row, "HSLSystemWide")
    return out


def snapshot_row(client, snap, day_cache: dict) -> dict:
    """Assemble one output row: wind and solar region values for a snapshot hour."""
    date, he = snap["date"], snap["hour_ending"]
    hour_ending = int(str(he).split(":")[0])
    row = {"timestamp": snap["timestamp"], "date": date, "hour_ending": he}
    for fuel, endpoint, regions in (("wind", WIND_ENDPOINT, WIND_REGIONS),
                                    ("solar", SOLAR_ENDPOINT, SOLAR_REGIONS)):
        key = (fuel, date)
        if key not in day_cache:
            day_cache[key] = fetch_day(client, endpoint, date)
        match = settled_row(day_cache[key], hour_ending)
        row.update(region_values(match, fuel, regions))
    return row


def main() -> None:
    snapshots = pd.read_csv(SNAPSHOTS)
    client = build_client()
    day_cache: dict = {}
    rows = [snapshot_row(client, s, day_cache) for _, s in snapshots.iterrows()]
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out[["timestamp", "wind_cophsl_systemwide", "solar_cophsl_systemwide"]]
          .to_string(index=False))
    print(f"\nwrote {len(out)} snapshots -> {OUT}")


if __name__ == "__main__":
    main()
