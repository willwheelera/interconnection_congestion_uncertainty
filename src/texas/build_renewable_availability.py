"""Attach a per-fuel renewable capacity factor to each load snapshot.

The load collection (data/activsg2000_load_collection.json) records, per snapshot,
the per-zone load scale. This adds the matching renewable side: for the same hour,
a capacity factor for wind and for solar,

    cf = COP_HSL_system_wide(hour) / ERCOT_installed_capacity(fuel),

where the numerator is ERCOT's available generation for that hour (from
fetch_renewable_snapshots.py) and the denominator is the full 2025 operational
installed capacity of that fuel, summed from the latest CDR unit-detail sheet.
The result is a weather-and-time capacity factor in [0, 1]: near zero for solar in
the evening, a fluctuating mid-range value for wind. It is written back into each
snapshot as `renewable_cf` so the target solve can derate the tagged renewables.

This is system-wide (one cf per fuel per hour). The regional columns in the
renewable-snapshot CSV support a later per-zone refinement without a re-fetch.

Usage:
    uv run python src/texas/build_renewable_availability.py
"""

import json
from pathlib import Path

import openpyxl
import pandas as pd

from compile_resource_additions import CDR, fuel_group, read_table

REPO_ROOT = Path(__file__).parents[2]
RENEWABLES = REPO_ROOT / "data" / "activsg2000_renewable_snapshots.csv"
COLLECTION = REPO_ROOT / "data" / "activsg2000_load_collection.json"

FUELS = ["Wind", "Solar"]


def ercot_installed_mw() -> dict[str, float]:
    """Total operational installed MW by fuel group, from the latest CDR."""
    wb = openpyxl.load_workbook(CDR, read_only=True, data_only=True)
    fleet = read_table(wb["Unit Details"])
    wb.close()
    totals = {f: 0.0 for f in FUELS}
    for r in fleet:
        if not str(r["status"]).startswith("OPER"):
            continue
        grp = fuel_group(r["fuel"] or "")
        if grp in totals:
            totals[grp] += r["installed_mw"]
    return totals


def snapshot_cfs(renewables: pd.DataFrame, installed: dict[str, float]) -> dict[str, dict]:
    """Map snapshot timestamp -> {fuel: capacity factor} from COP HSL / installed."""
    cfs = {}
    for _, row in renewables.iterrows():
        cf = {}
        for fuel in FUELS:
            avail = float(row[f"{fuel.lower()}_cophsl_systemwide"])
            cf[fuel] = min(avail / installed[fuel], 1.0)
        cfs[row["timestamp"]] = cf
    return cfs


def main() -> None:
    installed = ercot_installed_mw()
    renewables = pd.read_csv(RENEWABLES)
    cfs = snapshot_cfs(renewables, installed)

    collection = json.loads(COLLECTION.read_text())
    for snap in collection["snapshots"]:
        snap["renewable_cf"] = cfs[snap["timestamp"]]
    collection["ercot_installed_mw"] = installed
    COLLECTION.write_text(json.dumps(collection, indent=2))

    print("ERCOT installed MW:", {f: round(v) for f, v in installed.items()})
    for snap in collection["snapshots"]:
        cf = snap["renewable_cf"]
        print(f"  {snap['timestamp']}  wind cf {cf['Wind']:.2f}  solar cf {cf['Solar']:.2f}")
    print(f"\nwrote renewable_cf into {len(collection['snapshots'])} snapshots -> {COLLECTION}")


if __name__ == "__main__":
    main()
