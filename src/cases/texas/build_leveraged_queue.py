"""Draw a queue of real ERCOT projects restricted to sites that can move congestion.

Same source list, same nearest-county-centroid bus matching, and same stratified-by-fuel
quota as build_queue_distribution.py.  What changes is which projects are in the pool: a
project qualifies only if its bus has shift-factor leverage on one of the constraints that
actually carry mu (`screen_bus_leverage.py`), and only if its capacity-factor derated size
turns that leverage into enough MW to matter.  Target constraints are the highest-priced
ones the real queue can reach with every fuel, and slots are dealt round-robin over them,
so the queue loads several congestion modes with several candidates each.

No project is moved, resized, or repriced.  `--aggregate K` optionally merges the largest
K same-fuel projects sharing a bus into one candidate, representing the combined effect of
neighboring projects at a single point of interconnection.

Usage:
    uv run python src/cases/texas/build_leveraged_queue.py --seed 0
    uv run python src/cases/texas/build_leveraged_queue.py --seed 0 --aggregate 3
"""

import argparse
import json
import random

import pandas as pd

from src.cases.registry import CASES
from src.cases.texas.build_queue_distribution import (
    CASE, PROJECTS, FUEL_QUOTA, eligible_projects, load_bus_coords, match_to_bus)
from src.paths import DATA


def capacity_factors(case) -> dict:
    """Mean derating per fuel across snapshots; a fuel absent from the snapshots runs flat."""
    cfs = [s["renewable_cf"] for s in case.snapshots()]
    return {f: sum(cf[f] for cf in cfs) / len(cfs) for cf in cfs for f in cf}


def load_leverage(case) -> tuple[pd.DataFrame, dict]:
    """Read the bus x constraint leverage table, keeping one constraint per line pair.

    A double-circuit outage prices both directions -- `ctg:329:330` and `ctg:330:329` are
    the same corridor -- so only the higher-priced one is kept as a target.
    """
    lev = pd.read_csv(case.results_dir / "bus_leverage.csv", index_col="bus")
    lev.index = lev.index.astype(str)
    mu = json.loads((case.results_dir / "bus_leverage_mu.json").read_text())
    seen, keep = set(), []
    for key in sorted(mu, key=mu.get, reverse=True):
        pair = frozenset(key.split(":")[1:])
        if pair not in seen:
            seen.add(pair)
            keep.append(key)
    return lev[keep], mu


def build_pool(case) -> pd.DataFrame:
    """Every eligible queue project with its matched bus and capacity-factor derated size."""
    cf = capacity_factors(case)
    bus_ids, bus_coords = load_bus_coords(CASE)
    rows = []
    for p in eligible_projects(json.loads(PROJECTS.read_text())["projects"]):
        bus, dist = match_to_bus(p, bus_ids, bus_coords)
        fuel = p.get("fuel_display")
        rows.append({"inr": p["inr"], "name": p["name"], "fuel": fuel,
                     "capacity_mw": p["capacity_mw"],
                     "eff_mw": cf.get(fuel, 1.0) * p["capacity_mw"],
                     "p": p["completion_probability_calibrated"] / 100.0,
                     "county": p.get("county"), "county_lat": p["county_lat"],
                     "county_lng": p["county_lng"], "bus": bus,
                     "match_dist_km": round(dist, 2)})
    return pd.DataFrame(rows)


def aggregate_sites(pool: pd.DataFrame, k: int) -> pd.DataFrame:
    """Merge the largest k same-fuel projects at each bus into one aggregate candidate.

    Capacity adds; the build probability is the capacity-weighted mean of the merged
    projects, so an aggregate is as likely to be built as its constituents are on average.
    """
    out = []
    for (bus, fuel), group in pool.groupby(["bus", "fuel"], sort=False):
        top = group.nlargest(k, "capacity_mw")
        head = top.iloc[0].to_dict()
        if len(top) > 1:
            head |= {"name": f"{len(top)}x {fuel} at {head['county']}",
                     "inr": f"{head['inr']}+{len(top) - 1}",
                     "capacity_mw": float(top.capacity_mw.sum()),
                     "eff_mw": float(top.eff_mw.sum()),
                     "p": float((top.p * top.capacity_mw).sum() / top.capacity_mw.sum())}
        out.append(head)
    return pd.DataFrame(out)


def screen_matrix(pool: pd.DataFrame, lev: pd.DataFrame) -> pd.DataFrame:
    """MW each project could push onto each priced constraint: derated size x |shift factor|."""
    sf = lev.loc[pool.bus.values].set_index(pool.index)
    return sf.mul(pool.eff_mw, axis=0)


def target_modes(screen, pool, mu, fuels, min_screen, nmodes, per_fuel) -> list:
    """The highest-priced constraints every fuel can reach with `min_screen` MW of leverage.

    A constraint no queue project can load is not a usable target however expensive it is,
    and one only batteries can reach would break the fuel quota.
    """
    ok = []
    for mode in screen.columns:
        counts = pool.fuel[screen[mode] >= min_screen].value_counts()
        if all(counts.get(f, 0) >= per_fuel for f in fuels):
            ok.append(mode)
    return sorted(ok, key=mu.get, reverse=True)[:nmodes]


def assign_modes(pool: pd.DataFrame, screen: pd.DataFrame, modes: list) -> pd.DataFrame:
    """Label each project with the target constraint it loads hardest, and by how much."""
    sub = screen[modes]
    return pool.assign(mode=sub.idxmax(axis=1), screen_mw=sub.max(axis=1))


def draw(pool: pd.DataFrame, quota: dict, modes: list, min_screen: float,
         topn: int, seed: int) -> pd.DataFrame:
    """Deal each fuel's slots round-robin over the target modes, sampling among the best.

    Within a (fuel, mode) slot the pick is uniform over the `topn` qualifying projects with
    the most leverage on that mode, which is what makes different seeds give genuinely
    different queues rather than reshuffles of one ranking.
    """
    rng = random.Random(seed)
    pool = pool[pool.screen_mw >= min_screen]
    chosen, turn = [], 0
    for fuel, k in quota.items():
        for _ in range(k):
            mode = modes[turn % len(modes)]
            turn += 1
            free = pool[(pool.fuel == fuel) & ~pool.inr.isin(chosen)]
            bucket = free[free["mode"] == mode]
            top = (bucket if len(bucket) else free).nlargest(topn, "screen_mw")
            chosen.append(top.inr.iloc[rng.randrange(len(top))])
    return pool[pool.inr.isin(chosen)]


def to_candidates(picks: pd.DataFrame) -> list[dict]:
    """The queue-file schema the case registry and the solve builder read."""
    keep = ["inr", "name", "fuel", "capacity_mw", "p", "county", "county_lat",
            "county_lng", "bus", "match_dist_km"]
    return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            for row in picks[keep].to_dict("records")]


def print_summary(picks: pd.DataFrame, modes: list, baseline: pd.DataFrame) -> None:
    """Print the drawn queue, then the totals that make two draws comparable."""
    show = picks.sort_values(["mode", "screen_mw"], ascending=[True, False])
    print(f"{'fuel':<8} {'MW':>8} {'eff':>7} {'p':>5} {'bus':>6} {'screen':>8}  "
          f"{'mode':<14} name")
    for _, r in show.iterrows():
        print(f"{r.fuel:<8} {r.capacity_mw:>8.1f} {r.eff_mw:>7.1f} {r.p:>5.2f} "
              f"{r.bus:>6} {r.screen_mw:>8.1f}  {r['mode']:<14} {r['name'][:36]}")
    counts = picks["mode"].value_counts()
    print("\nmode coverage: " + "  ".join(f"{m}={counts.get(m, 0)}" for m in modes))
    print(f"{len(picks)} candidates  {picks.capacity_mw.sum():.0f} MW nameplate  "
          f"{picks.eff_mw.sum():.0f} MW effective  "
          f"{(picks.p * picks.eff_mw).sum():.0f} MW expected online  "
          f"{picks.bus.nunique()} distinct buses")
    print(f"screen leverage on target modes: total {picks.screen_mw.sum():.0f} MW, "
          f"median {picks.screen_mw.median():.1f} MW  "
          f"(current queue: total {baseline.screen_mw.sum():.0f} MW, "
          f"median {baseline.screen_mw.median():.1f} MW)")


def current_queue(case, lev: pd.DataFrame, modes: list) -> pd.DataFrame:
    """The queue the case ships with, screened the same way, as a comparison baseline."""
    cf = capacity_factors(case)
    pool = pd.DataFrame([{"fuel": c["fuel"], "bus": str(c["bus"]),
                          "eff_mw": cf.get(c["fuel"], 1.0) * c["capacity_mw"]}
                         for c in case.candidates()])
    return assign_modes(pool, screen_matrix(pool, lev), modes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="activsg2000_original_v12")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nmodes", type=int, default=5)
    parser.add_argument("--min-screen-mw", type=float, default=20.0)
    parser.add_argument("--per-fuel", type=int, default=2)
    parser.add_argument("--topn", type=int, default=8)
    parser.add_argument("--aggregate", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    case = CASES[args.case]
    lev, mu = load_leverage(case)
    pool = build_pool(case)
    if args.aggregate > 1:
        pool = aggregate_sites(pool, args.aggregate)
    screen = screen_matrix(pool, lev)
    modes = target_modes(screen, pool, mu, FUEL_QUOTA, args.min_screen_mw,
                         args.nmodes, args.per_fuel)
    pool = assign_modes(pool, screen, modes)
    picks = draw(pool, FUEL_QUOTA, modes, args.min_screen_mw, args.topn, args.seed)

    tag = f"lev{args.seed}" + (f"a{args.aggregate}" if args.aggregate > 1 else "")
    out = DATA / (args.out or f"activsg2000_queue20_{tag}.json")
    out.write_text(json.dumps({"seed": args.seed, "nmodes": args.nmodes,
                               "aggregate": args.aggregate, "modes": modes,
                               "candidates": to_candidates(picks)}, indent=2))
    print("target modes: " + "  ".join(f"{m} (mu={mu[m]:.0f})" for m in modes))
    print_summary(picks, modes, current_queue(case, lev, modes))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
