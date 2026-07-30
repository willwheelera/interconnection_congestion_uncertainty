"""Turn fetched ERCOT load snapshots into per-zone ACTIVSg2000 load scale factors.

Each snapshot carries ERCOT actual load for the eight weather zones. Dividing by
the model's nominal load in each zone gives a per-zone scale factor; applying it
to that zone's buses reproduces the snapshot's spatial loading pattern (not just
its total). Different snapshots load the zones differently, congesting different
constraints.

Zone scales are capped at ZONE_SCALE_CAP. In practice only FarWest hits it: the
2018 model's FarWest load/infrastructure is far below 2025 Permian levels, so its
uncapped ~5-6x scale forces radial load-shedding that would pin the max-over-lines
congestion target at the value of lost load. Other zones stay at true ERCOT ratios.

Output: data/activsg2000_load_collection.json holding the nominal per-zone load
and, per snapshot, its total MW and per-zone scale factors.

Usage:
    uv run python src/texas/build_load_snapshots.py
"""

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parents[2]
CASE = REPO_ROOT / "data" / "activsg2000.json"
SNAPSHOTS = REPO_ROOT / "data" / "activsg2000_load_snapshots.csv"
OUT = REPO_ROOT / "data" / "activsg2000_load_collection.json"

ZONES = ["Coast", "East", "FarWest", "North", "NorthC", "Southern", "SouthC", "West"]

# Largest per-zone load scale allowed; above this the 2018 topology sheds load.
ZONE_SCALE_CAP = 1.5


def nominal_zone_load(case_path: Path) -> dict[str, float]:
    """Sum nominal bus loads within each ERCOT weather zone (MW)."""
    md = json.loads(case_path.read_text())
    buses = md["elements"]["bus"]
    totals = {z: 0.0 for z in ZONES}
    for load in md["elements"]["load"].values():
        totals[buses[load["bus"]]["zone"]] += load["p_load"]
    return totals


def build_collection(nominal: dict[str, float], snapshots: pd.DataFrame) -> list[dict]:
    """One record per snapshot with per-zone scale factors and system total."""
    out = []
    for _, s in snapshots.iterrows():
        scales = {z: min(float(s[z]) / nominal[z], ZONE_SCALE_CAP) for z in ZONES}
        out.append({
            "timestamp": s["timestamp"],
            "total_mw": float(s["Total"]),
            "zone_scales": scales,
            "max_price": float(s["max_price"]),
            "max_constraint": s["max_constraint"],
        })
    return out


def main():
    nominal = nominal_zone_load(CASE)
    snapshots = pd.read_csv(SNAPSHOTS)
    collection = build_collection(nominal, snapshots)
    OUT.write_text(json.dumps(
        {"nominal_zone_mw": nominal, "snapshots": collection}, indent=2))
    print("nominal zone load MW:", {z: round(v) for z, v in nominal.items()})
    for s in collection:
        top = max(s["zone_scales"].items(), key=lambda kv: kv[1])
        print(f"  {s['timestamp']}  total {s['total_mw']:.0f} MW  "
              f"peak-zone scale {top[0]} x{top[1]:.2f}")
    print(f"\nwrote {len(collection)} snapshots -> {OUT}")


if __name__ == "__main__":
    main()
