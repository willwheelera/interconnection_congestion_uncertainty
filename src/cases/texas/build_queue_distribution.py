"""Select 20 ERCOT queue projects, match them to ACTIVSg2000 buses, and build
the Bernoulli distribution over which of them come online.

Each candidate generator becomes operational independently with probability
p = completion_probability_calibrated / 100 (the ercotqueue.com calibrated build
chance). The joint distribution over the 20 generators is the product of these
Bernoullis; a configuration is a length-20 binary vector.

Matching is geographic: the synthetic ACTIVSg2000 bus numbers do not correspond
to real ERCOT points of interconnection, so each project (located by its county
centroid) is assigned to the nearest network bus by great-circle distance.

Candidates are drawn (seeded) from active projects with a non-degenerate build
chance and a known capacity/location, stratified by fuel (see FUEL_QUOTA) so the
set mixes thermal and different renewables -- an arbitrary draw, not the most likely.

Usage:
    uv run python src/texas/build_queue_distribution.py
    uv run python src/texas/build_queue_distribution.py --seed 0
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.paths import REPO_ROOT
PROJECTS = REPO_ROOT / "texas_model" / "ercot_queue" / "projects.json"
CASE = REPO_ROOT / "data" / "activsg2000.json"
OUT = REPO_ROOT / "data" / "activsg2000_queue20.json"

EARTH_RADIUS_KM = 6371.0

# How many candidates to draw from each fuel type, for a mix of thermal and
# different renewables. Must sum to the desired candidate count.
FUEL_QUOTA = {"Gas": 5, "Wind": 5, "Solar": 5, "Battery": 5}


def load_bus_coords(case_path: Path) -> tuple[list[str], np.ndarray]:
    """Return (bus_ids, Nx2 array of [lat, lon]) from the converted EGRET JSON."""
    md = json.loads(case_path.read_text())
    buses = md["elements"]["bus"]
    ids = list(buses.keys())
    coords = np.array([[buses[b]["lat"], buses[b]["lon"]] for b in ids])
    return ids, coords


def eligible_projects(projects: list[dict]) -> list[dict]:
    """Keep active projects with a non-degenerate build chance and known site."""
    keep = []
    for p in projects:
        prob = p.get("completion_probability_calibrated")
        if p.get("is_commissioned"):
            continue
        if prob is None or not (0 < prob < 100):
            continue
        if p.get("county_lat") is None or p.get("county_lng") is None:
            continue
        if not p.get("capacity_mw"):
            continue
        keep.append(p)
    return keep


def haversine_km(lat, lon, lats, lons) -> np.ndarray:
    """Great-circle distance (km) from one point to arrays of points."""
    lat, lon, lats, lons = map(np.radians, (lat, lon, lats, lons))
    dlat = lats - lat
    dlon = lons - lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lats) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def match_to_bus(project: dict, bus_ids, bus_coords) -> tuple[str, float]:
    """Return (nearest_bus_id, distance_km) for a project's county centroid."""
    d = haversine_km(project["county_lat"], project["county_lng"],
                     bus_coords[:, 0], bus_coords[:, 1])
    i = int(np.argmin(d))
    return bus_ids[i], float(d[i])


def build_candidates(projects, bus_ids, bus_coords, quota, seed) -> list[dict]:
    """Draw the quota of eligible projects per fuel (seeded), match each to a bus."""
    pool = eligible_projects(projects)
    rng = random.Random(seed)
    chosen = []
    for fuel, k in quota.items():
        group = [p for p in pool if p.get("fuel_display") == fuel]
        chosen += rng.sample(group, k)
    out = []
    for p in chosen:
        bus, dist = match_to_bus(p, bus_ids, bus_coords)
        out.append({
            "inr": p["inr"],
            "name": p["name"],
            "fuel": p.get("fuel_display"),
            "capacity_mw": p["capacity_mw"],
            "p": p["completion_probability_calibrated"] / 100.0,
            "county": p.get("county"),
            "county_lat": p["county_lat"],
            "county_lng": p["county_lng"],
            "bus": bus,
            "match_dist_km": round(dist, 2),
        })
    return out


def print_summary(candidates: list[dict]) -> None:
    """Print the candidate list and the expected number/capacity coming online."""
    print(f"{'inr':<12} {'fuel':<8} {'MW':>7} {'p':>5} {'bus':>6} {'dist_km':>8}  name")
    for c in candidates:
        print(f"{c['inr']:<12} {str(c['fuel']):<8} {c['capacity_mw']:>7.1f} "
              f"{c['p']:>5.2f} {c['bus']:>6} {c['match_dist_km']:>8.1f}  {c['name']}")
    exp_n = sum(c["p"] for c in candidates)
    exp_mw = sum(c["p"] * c["capacity_mw"] for c in candidates)
    tot_mw = sum(c["capacity_mw"] for c in candidates)
    print(f"\n{len(candidates)} candidates  total {tot_mw:.0f} MW nameplate")
    print(f"expected online: {exp_n:.1f} generators, {exp_mw:.0f} MW")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    projects = json.loads(PROJECTS.read_text())["projects"]
    bus_ids, bus_coords = load_bus_coords(CASE)
    candidates = build_candidates(projects, bus_ids, bus_coords, FUEL_QUOTA, args.seed)

    OUT.write_text(json.dumps({"seed": args.seed, "candidates": candidates}, indent=2))
    print_summary(candidates)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
