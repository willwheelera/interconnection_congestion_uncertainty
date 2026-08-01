"""Add the 2017-2025 ERCOT resource additions to ACTIVSg2000 as new generators.

The synthetic ACTIVSg2000 case represents the ERCOT fleet of its ~2016 vintage,
so every resource with an in-service year 2017-2025 in the latest CDR is a unit
present in 2025 but absent from the TAMU model. This script injects each of those
units (from data/ercot_resource_additions_2017_2025.csv) as a generator at an
appropriate node.

Placement is geographic, matching build_queue_distribution.py: the synthetic bus
numbers do not correspond to real interconnection points, so each unit is assigned
to the network bus nearest its county centroid by great-circle distance. County
centroids come from the ERCOT queue projects file, with a small supplement for the
few counties absent there.

Each new generator uses the same minimal DCOPF schema as add_queue_generators:
p_min 0, p_max = nameplate MW, and a by-fuel marginal cost placing renewables and
storage at the bottom of the merit order and new thermal above the existing fleet.
Descriptive fields (fuel, county, zone, in-service year) are kept for downstream
filtering; EGRET ignores them when solving.

Usage:
    uv run python src/texas/add_cdr_resources.py
"""

import csv
import json
from pathlib import Path

import numpy as np

from paths import REPO_ROOT
CASE = REPO_ROOT / "data" / "activsg2000.json"
ADDITIONS = REPO_ROOT / "data" / "ercot_resource_additions_2017_2025.csv"
PROJECTS = REPO_ROOT / "texas_model" / "ercot_queue" / "projects.json"
OUT = REPO_ROOT / "data" / "activsg2000_2025.json"

EARTH_RADIUS_KM = 6371.0

# Marginal cost ($/MWh) by fuel group: renewables/storage at zero so they clear
# first; new thermal above the existing fleet's ~$17-19 so it runs last.
GROUP_COST = {"Solar": 0.0, "Wind": 0.0, "Storage": 0.0, "Hydro": 0.0,
              "Nuclear": 8.0, "Coal": 18.0, "Gas": 25.0, "Biomass": 30.0,
              "Diesel": 40.0}

# County centroids (lat, lon) absent from the queue projects file.
SUPPLEMENTAL_CENTROIDS = {
    "IRION": (31.30, -100.98), "PARMER": (34.53, -102.78),
    "OLDHAM": (35.40, -102.60), "FLOYD": (34.07, -101.30),
    "DONLEY": (34.96, -100.81), "RANDALL": (34.96, -101.90),
    "MASON": (30.72, -99.23), "TITUS": (33.22, -94.96),
}


def load_bus_coords(case: dict) -> tuple[list[str], np.ndarray]:
    """Return (bus_ids, Nx2 array of [lat, lon]) from the case elements."""
    buses = case["elements"]["bus"]
    ids = list(buses.keys())
    coords = np.array([[buses[b]["lat"], buses[b]["lon"]] for b in ids])
    return ids, coords


def county_centroids() -> dict[str, tuple[float, float]]:
    """Map upper-cased Texas county name -> centroid, from the queue file."""
    projects = json.loads(PROJECTS.read_text())["projects"]
    centroids = dict(SUPPLEMENTAL_CENTROIDS)
    for p in projects:
        c, la, lo = p.get("county"), p.get("county_lat"), p.get("county_lng")
        if c and la is not None and lo is not None:
            centroids.setdefault(c.strip().upper(), (la, lo))
    return centroids


def resolve_centroid(county: str, centroids: dict) -> tuple[float, float] | None:
    """Look up a county centroid, splitting compound names like 'HILL & LIMESTONE'."""
    for part in county.replace("/", "&").replace(",", "&").split("&"):
        key = part.strip().upper()
        if key in centroids:
            return centroids[key]
    return None


def haversine_km(lat, lon, lats, lons) -> np.ndarray:
    """Great-circle distance (km) from one point to arrays of points."""
    lat, lon, lats, lons = map(np.radians, (lat, lon, lats, lons))
    a = (np.sin((lats - lat) / 2) ** 2
         + np.cos(lat) * np.cos(lats) * np.sin((lons - lon) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def make_generator(unit: dict, bus: str, dist_km: float) -> dict:
    """Build a generator element for one CDR unit at its matched bus."""
    return {
        "bus": bus, "generator_type": "thermal", "in_service": True,
        "p_min": 0.0, "p_max": float(unit["installed_mw"]), "pg": 0.0,
        "p_cost": {"data_type": "cost_curve", "cost_curve_type": "polynomial",
                   "values": {"0": 0.0, "1": GROUP_COST[unit["fuel_group"]], "2": 0.0}},
        "fuel": unit["fuel"], "fuel_group": unit["fuel_group"],
        "unit_code": unit["unit_code"], "county": unit["county"],
        "cdr_zone": unit["zone"], "in_service_year": int(unit["in_service"][:4]),
        "match_dist_km": round(dist_km, 1),
    }


def add_units(case: dict, units: list[dict], centroids: dict) -> list[float]:
    """Inject each unit as a generator at its nearest bus; return match distances."""
    gens = case["elements"]["generator"]
    bus_ids, bus_coords = load_bus_coords(case)
    dists = []
    for i, u in enumerate(units):
        centroid = resolve_centroid(u["county"], centroids)
        if centroid is None:
            raise KeyError(f"no centroid for county {u['county']!r}")
        d = haversine_km(centroid[0], centroid[1], bus_coords[:, 0], bus_coords[:, 1])
        j = int(np.argmin(d))
        key = f"cdr_{u['unit_code']}" if u["unit_code"] else f"cdr_row{i}"
        while key in gens:
            key += "_d"
        gens[key] = make_generator(u, bus_ids[j], float(d[j]))
        dists.append(float(d[j]))
    return dists


def print_summary(case: dict, units: list[dict], dists: list[float],
                  base_gen_mw: float) -> None:
    """Report what was added and the resulting model totals."""
    from collections import defaultdict
    grp = defaultdict(float)
    for u in units:
        grp[u["fuel_group"]] += float(u["installed_mw"])
    added_mw = sum(float(u["installed_mw"]) for u in units)
    gens = case["elements"]["generator"]
    print(f"Added {len(units)} generators, {added_mw:,.0f} MW nameplate")
    for g in sorted(grp, key=lambda k: -grp[k]):
        print(f"  {g.ljust(9)} {grp[g]:8.0f} MW")
    print(f"\nMatch distance to bus (km): median {np.median(dists):.1f}  "
          f"mean {np.mean(dists):.1f}  max {np.max(dists):.1f}")
    print(f"units matched >75 km away: {sum(1 for d in dists if d > 75)}")
    print(f"\nGenerators: {544} -> {len(gens)}   "
          f"Capacity: {base_gen_mw:,.0f} -> {base_gen_mw + added_mw:,.0f} MW")


def main() -> None:
    case = json.loads(CASE.read_text())
    base_gen_mw = sum(g["p_max"] for g in case["elements"]["generator"].values())
    units = list(csv.DictReader(ADDITIONS.open()))
    centroids = county_centroids()

    dists = add_units(case, units, centroids)
    OUT.write_text(json.dumps(case, indent=2))
    print_summary(case, units, dists, base_gen_mw)
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
